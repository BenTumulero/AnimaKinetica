"""Cell-culture simulation orchestration and process KPI calculation.

:class:`CellCultureSimulation` assembles parameter, stoichiometric, kinetic,
gas-control, pH-control, feed, perfusion, and operating-mode ODE models. It
integrates successive phases with :func:`scipy.integrate.solve_ivp`, applies
terminal process events, handles repeated-batch inoculum carryover, and stores
results in :class:`~animakinetica.core.simulation_data.SimulationData`.

By default, :meth:`CellCultureSimulation.run_simulation` evaluates one
maximum-culture-age window and projects additive totals to the configured yearly
uptime. Disable projection or provide an explicit time limit to retain literal
simulation-window totals.
"""
# Imports
# Standard library
import os
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

# Third-party
import numpy as np
from scipy.integrate import solve_ivp

# Local
from animakinetica.core.parameters import SimulationParameters, Parameter
if TYPE_CHECKING:
    from animakinetica.analysis.optimization import OptimizationManager
from animakinetica.core.io_utils import (
    load_pickle_object,
    resolve_pickle_load_path,
    resolve_pickle_save_path,
    save_pickle_object,
)
from animakinetica.core.simulation_data import (
    SimulationData, BatchData, BatchDataAccumulator, EXPORT_DIR,
    BATCH_TOTAL_ACCUMULATION_MAP
    )
from animakinetica.models.events import *
from animakinetica.models.feed_strategies import FeedStrategy, create_feed_model
from animakinetica.models.gas_controllers import GasController, create_gas_controller_model
from animakinetica.models.kinetics import Kinetics, create_kinetic_model
from animakinetica.models.odesystems import create_ode_systems, OperatingMode, BatchODESystem, FedBatchODESystem, PerfusionODESystem, ChemostatODESystem
from animakinetica.models.perfusion_strategies import PerfusionStrategy, create_perfusion_strategy
from animakinetica.models.stoichiometry import StoichiometricModel, StoichiometryMode

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = ["CellCultureSimulation", "timing_decorator"]

# Operating modes that should restart from a fresh inoculum after each repeated run.
# For these long-duration operations, max_culture_age is a per-run time limit rather than a
# cumulative vessel-age limit across repeated runs.
CONTINUOUS_MODES = {
    OperatingMode.CHEMOSTAT,
    OperatingMode.CONTINUOUS_PERFUSION,
}

# Modes that run a startup fed-batch only at the beginning of each fresh
# vessel lifetime, then continue with repeated production-only cycles.
STARTUP_FEDBATCH_MODES = {
    OperatingMode.BATCH_WITH_STARTUP_FEDBATCH,
}

# One-to-one mapping for run-level totals accumulation from per-batch totals.
# Derived from the _TRACKED_QUANTITIES registry in simulation_data.py.
# Note: total_medium_in is NOT in this map — it is accumulated with a special
# line that sums batch_batch_medium_in + batch_feed_medium_in.
_BATCH_TOTAL_ACCUMULATION_MAP: dict[str, str] = dict(BATCH_TOTAL_ACCUMULATION_MAP)

# Keys in the `totals` accumulator that are additive quantities and should be
# scaled when projecting a short simulation window to a yearly horizon.
_ADDITIVE_TOTAL_KEYS: tuple = (
    'total_cultivation_time',
    'total_medium_in',
    *_BATCH_TOTAL_ACCUMULATION_MAP.keys(),
    'number_of_batches',
)

_DERIVED_SUMMARY_KEYS: tuple = (
    'total_DCM_harvested_kg',
    'total_WCM_harvested_kg',
    'STY_kg_m3_day',
    'STY_WCM_kg_m3_day',
    'medium_conversion_efficiency_m3_kg',
    'medium_conversion_efficiency_m3_kg_WCM',
    'protein_conversion_efficiency_kg_kg',
    'feed_conversion_efficiency_kg_kg',
    'feed_conversion_efficiency_kg_kg_WCM',
)

_TOTAL_DIAGNOSTIC_KEYS: tuple = (
    'number_of_batches',
    'phase_active_runtime_h',
    'batch_phase_active_runtime_h',
    'last_batch_termination_reason',
    'unique_batch_termination_reasons',
    'batch_time_min_h',
    'batch_time_max_h',
    'batch_time_range_h',
    'last_batch_final_volume_L',
    'last_batch_final_volume_fraction',
    'batch_final_volume_min_L',
    'batch_final_volume_max_L',
    'batch_final_volume_min_fraction',
    'batch_final_volume_max_fraction',
)

def timing_decorator(func):
    """Optionally print elapsed wall-clock time after a function call.

    Set ``ANIMAKINETICA_TIMING_OUTPUT=0`` to suppress timing diagnostics for
    a process, such as when a campaign runner executes many child scripts.
    Direct script execution keeps timing output enabled by default.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    callable
        Wrapped function with unchanged return value.
    """
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        if os.environ.get("ANIMAKINETICA_TIMING_OUTPUT", "1") != "0":
            print(f"{func.__name__} execution time: {(end_time - start_time) * 1000:.3f} ms")
        return result
    return wrapper

class CellCultureSimulation:
    """Configure, run, serialize, and analyze cell-culture simulations.

    The object owns the model instances and the resulting
    :class:`~animakinetica.core.simulation_data.SimulationData`. ``EXCESS``
    kinetics always uses ``EXCESS_COUPLED`` stoichiometry internally. Set
    ``performance_mode=True`` for optimization or sensitivity sweeps; use the
    default detailed mode when plots and trace arrays are required.
    """

    def __init__(self, 
                 parameters: SimulationParameters,
                 kinetics: Kinetics = Kinetics.STOICHIOMETRIC,
                 gas_controller: GasController = GasController.CASCADE_N_FG_XO2,
                 operating_mode: OperatingMode = OperatingMode.BATCH,
                 feed_strategy: FeedStrategy = None,
                 perfusion_strategy: Optional[PerfusionStrategy] = None,
                 stoichiometry_mode: StoichiometryMode = StoichiometryMode.CONSTANT,
                 performance_mode: bool = False,
                 solver_method: str = 'LSODA',
                 **kwargs
                ):
        """Initialize a simulation and its coupled process models.

        Parameters
        ----------
        parameters : SimulationParameters
            Fully configured simulation parameters. Batch-mode volume defaults
            are applied to an internal copy when appropriate.
        kinetics : Kinetics, default=Kinetics.STOICHIOMETRIC
            Kinetic-model family. ``Kinetics.EXCESS`` forces
            :attr:`StoichiometryMode.EXCESS_COUPLED`.
        gas_controller : GasController, default=GasController.CASCADE_N_FG_XO2
            Dissolved-oxygen gas-control strategy.
        operating_mode : OperatingMode, default=OperatingMode.BATCH
            Bioreactor operating mode used to construct ODE phases.
        feed_strategy : FeedStrategy, optional
            Feed-rate strategy for feed-requiring modes.
        perfusion_strategy : PerfusionStrategy, optional
            Dilution-rate strategy for perfusion-requiring modes.
        stoichiometry_mode : StoichiometryMode, default=StoichiometryMode.CONSTANT
            Constant or dynamic yield policy for stoichiometric kinetics;
            overridden for excess kinetics.
        performance_mode : bool, default=False
            Whether to relax solver tolerances and suppress detailed ODE logging
            for high-throughput evaluations.
        solver_method : str, default="LSODA"
            Integration method passed to :func:`scipy.integrate.solve_ivp`.
            ``"BDF"`` can be appropriate for persistently stiff systems.
        **kwargs
            Kinetic-model options, such as inhibition-model selections.
        """
        # Apply mode-specific parameter defaults before any dependent models cache values.
        self.parameters = parameters.create_mode_adjusted_copy(operating_mode)
        
        # Extract initial conditions from parameters for easy access in the simulation loop.
        self.base = {
            'X_init':        float(self.parameters.initial_conditions.X_inoc),
            'V_init':        float(self.parameters.reactor.V_batch_init),
            'C_Glc_init':    float(self.parameters.initial_conditions.C_Glc_init),
            'C_Gln_init':    float(self.parameters.initial_conditions.C_Gln_init),
            'C_Lac_init':    float(self.parameters.initial_conditions.C_Lac_init),
            'C_Amm_init':    float(self.parameters.initial_conditions.C_Amm_init),
            'C_O2_init':     float(self.parameters.initial_conditions.C_O2_init),
            'C_CT_init':     float(self.parameters.initial_conditions.C_CT_init),
            'C_H_init':      float(self.parameters.initial_conditions.C_H_init),
            'C_Na_init':     float(self.parameters.initial_conditions.C_Na_init),
            'C_AA_init':     float(self.parameters.initial_conditions.C_AA_init),
            'C_Osm_bg_init': float(self.parameters.initial_conditions.osm_bg_init),
            'yearly_uptime': float(self.parameters.process_schedule.yearly_uptime),
        }

        # Append non-carbonate buffer initial concentrations (one per buffer).
        # When no buffers are configured this adds nothing, keeping the state
        # vector at 11 elements (backward compatible).
        self._buffer_defs = self.parameters.acid_base_properties.buffers or []
        for i, buf in enumerate(self._buffer_defs):
            self.base[f'C_buffer_{i}_init'] = float(buf.C_init)

        # Store passed kwargs for temporary simulations
        self._kwargs = kwargs

        # Initialize data storage and batch counter
        self.data = SimulationData()
        self.data.parameters = self.parameters # Also store parameters in data object
        self._current_batch_number = 1 # Start batch numbering at 1

        # When using excess kinetics the catabolic stoichiometry is derived each ODE
        # step by back-calculation from the excess rates, so no catabolic parameter
        # file is needed. Force EXCESS_COUPLED regardless of what was passed.
        if kinetics == Kinetics.EXCESS:
            stoichiometry_mode = StoichiometryMode.EXCESS_COUPLED

        # Create and store
        self.stoichiometry_mode = stoichiometry_mode
        self.stoichiometric_model = StoichiometricModel(self.parameters, mode=stoichiometry_mode)
        self.kinetics = kinetics
        self.kinetic_model = create_kinetic_model(
            self.kinetics,
            self.parameters,
            self.stoichiometric_model,
            **kwargs,
        )
        self.gas_controller = gas_controller
        self.gas_control_model = create_gas_controller_model(self.gas_controller, self.parameters)
        self.feed_strategy = feed_strategy
        self.feed_strategy_model = create_feed_model(self.feed_strategy, self.parameters) if self.feed_strategy is not None else None
        self.perfusion_strategy = perfusion_strategy
        self.perfusion_strategy_model = create_perfusion_strategy(self.perfusion_strategy, self.parameters) if self.perfusion_strategy is not None else None
        self.operating_mode = operating_mode

        # Performance mode settings
        self.performance_mode = performance_mode
        self.solver_method = solver_method

        # Set solve_ivp solver tolerances based on performance mode
        # Keep pH and gas-related states tighter to improve event stability.
        self.rtol = 3e-6 if self.performance_mode else 1e-7
        _buf_tol = 5e-4 if self.performance_mode else 1e-5  # buffer conc (mmol/L)
        self.atol = [1e3 if self.performance_mode else 1e2,     # Xv 
                5e-4 if self.performance_mode else 1e-5,    # C_Glc
                5e-4 if self.performance_mode else 1e-5,    # C_Gln
                5e-4 if self.performance_mode else 1e-5,    # C_Lac
                5e-4 if self.performance_mode else 1e-5,    # C_Amm
                5e-5 if self.performance_mode else 1e-4,    # V     1 ml per L
                5e-5 if self.performance_mode else 1e-5,    # C_O2
                5e-5 if self.performance_mode else 1e-5,    # C_CT
                2e-8 if self.performance_mode else 5e-10,   # C_H: tight for pH accuracy
                5e-4 if self.performance_mode else 1e-5,    # C_Na
                5e-4 if self.performance_mode else 1e-5,    # C_AA
                5e-4 if self.performance_mode else 1e-5]    # C_Osm_bg
        # Append one tolerance per non-carbonate buffer state variable
        self.atol += [_buf_tol] * len(self._buffer_defs)


        # Set up maximum integration time (used for t_span in solve_ivp)
        self.max_culture_age = float(self.parameters.process_schedule.max_culture_age) * 24 # Convert from days to hours
        self.yearly_uptime = float(self.parameters.process_schedule.yearly_uptime) * 24 # Convert from days to hours
        
        # Create ODE systems for the selected operating mode
        self.ode_systems = create_ode_systems(self.parameters, self.operating_mode, self.stoichiometric_model,
                                              self.kinetic_model, self.gas_control_model, 
                                              self.feed_strategy_model, self.perfusion_strategy_model
                                              ) # Returns dict of ODE systems based on mode

        # Pre-build per-phase event lists once so _run_single_operation never re-evaluates them.
        # Dict keys match self.ode_systems keys; each value is a frozen list of event functions.
        self._phase_events: dict[str, list] = self._build_phase_events()

        # Initialize active ODE system. Active ODE is called by solve_ivp. It is activated and deactivated in the methods based on mode it requires for that part of the code.
        self.active_ODE = None 

        # Optimization attributes
        self.optimized: bool = False  # Flag to indicate if the simulation has been run in optimization mode
        self.optimization_manager: Optional['OptimizationManager'] = None  # Placeholder for optimization manager object

        # Cashed parameters for nutrient use calculations (to avoid repeated attribute access in loops)
        self.MW_glucose = 180.1572 # g/mol
        self.MW_glutamine = 146.1456 # g/mol
        self.MW_AAs_ex_glutamine = self.parameters.cell_properties.M_AAs_ex_glutamine # g/mol
        self.MW_lactate = 89.07 # g/mol, lactate equivalent
        self.MW_ammonium = 17.031 # g/mol, ammonium equivalent
        self.MW_CO2 = 44.01 # g/mol
        self.MW_O2 = 31.998 # g/mol
        self.MW_water = 18.01528 # g/mol

### Initialization helper methods ###
    def _build_phase_events(self) -> dict[str, list]:
        """Build static terminal-event lists for all configured ODE phases.

        Each phase begins with common safety and depletion events and receives
        only its applicable volume or phase-switch triggers.

        Returns
        -------
        dict[str, list]
            ODE phase names mapped to event callables compatible with
            :func:`scipy.integrate.solve_ivp`.
        """
        base_events = [
            glucose_depletion_event, glutamine_depletion_event, AA_depletion_event,
            oxygen_depletion_event,
            pH_deviation_event, viscosity_limit_event, cell_minimum_event,
            ammonia_IC100_event, lactate_IC100_event,
        ]

        phase_events: dict[str, list] = {}
        for phase_name, ode_system in self.ode_systems.items():
            events = list(base_events)  # fresh copy per phase

            # Add max volume event only for fed-batch-derived modes, since batch and perfusion modes operate at +- constant volume except for base and acid additions.
            if isinstance(ode_system, FedBatchODESystem):
                events.append(max_volume_event)

            # Fedbatch phase that precedes the chemostat phase:
            # add the cell-density trigger only when the parameter is provided.
            if isinstance(ode_system, FedBatchODESystem):
                if self.operating_mode == OperatingMode.CHEMOSTAT:

                    if (self.parameters.chemostat is not None
                            and self.parameters.chemostat.Xv_chemostat_trig is not None):
                        events.append(celldensity_chemostat_trigger_event)

                elif self.operating_mode in (OperatingMode.CONCENTRATED_FEDBATCH,
                                             OperatingMode.CONTINUOUS_PERFUSION):
                # Fedbatch phase that precedes the perfusion phase:
                # add each perfusion trigger only when the corresponding parameter is set.
                    if self.parameters.perfusion is not None:
                        if self.parameters.perfusion.C_Lac_pf_trig is not None:
                            events.append(lactate_pf_trigger_event)
                        if self.parameters.perfusion.C_Amm_pf_trig is not None:
                            events.append(ammonia_pf_trigger_event)
                        if self.parameters.perfusion.Xv_pf_trig is not None:
                            events.append(celldensity_pf_trigger_event)

            # ChemostatODESystem, PerfusionODESystem:
            # no additional trigger events — base_events only.
            if (isinstance(ode_system, PerfusionODESystem)
                    and self.operating_mode == OperatingMode.CONCENTRATED_FEDBATCH
                    and self.parameters.perfusion is not None
                    and self.parameters.perfusion.Xv_harvest_stop is not None):
                min_runtime = self.parameters.perfusion.min_perfusion_runtime_h
                events.append(
                    make_perfusion_harvest_stop_event(
                        min_perfusion_runtime_h=float(min_runtime) if min_runtime is not None else 0.0
                    )
                )

            phase_events[phase_name] = events

        return phase_events

    def _get_phase_events_for_batch_phase(self, phase_name: str, phase_names: list[str], phase_idx: int) -> list:
        """Return events for a phase in the current logical batch.

        Uses the pre-built static phase event list as a base, then applies
        sequence-dependent events that depend on where this phase appears in the
        current batch schedule.
        """
        events = list(self._phase_events[phase_name])

        # Add the glucose batch->fedbatch transition trigger only when the
        # current BATCH phase is immediately followed by a FEDBATCH phase in
        # this batch's phase schedule.
        next_phase = phase_names[phase_idx + 1] if phase_idx + 1 < len(phase_names) else None
        if (
            phase_name == 'BATCH'
            and next_phase == 'FEDBATCH'
            and self.parameters.fed_batch is not None
            and self.parameters.fed_batch.C_Glc_batch_trigger is not None
        ):
            events.append(glucose_batch_trigger_event)

        return events

### Single operation level methods ###
    def _calculate_medium_and_nutrient_use(self, sol, is_final_phase: bool = True, include_initial_charge: bool = False):
        """Calculate incremental process-boundary flows for one ODE phase.

        Initial charge is recorded at index zero, feed and gas flows are
        integrated by the trapezoidal rule, and terminal emptying is recorded
        only for a final phase. Internal carryover corrections are applied later
        by :class:`BatchDataAccumulator`.

        Parameters
        ----------
        sol : scipy.integrate.OdeResult
            Solution returned by :func:`scipy.integrate.solve_ivp`.
        is_final_phase : bool, default=True
            Whether terminal reactor emptying contributes external outputs.
        include_initial_charge : bool, default=False
            Whether the initial reactor charge is an external process input.

        Returns
        -------
        tuple[numpy.ndarray, ...]
            Thirty-four per-step incremental arrays for volume, nutrients,
            products, gases, biomass, water, and heat. Use :func:`numpy.sum`
            for a total and :func:`numpy.cumsum` for a cumulative profile.
        """
        params = self.parameters
        n_points = len(sol.t)
        dt_array = np.diff(sol.t)

        def _aligned_log_or_zeros(values) -> np.ndarray:
            """Return a logged trajectory aligned to the accepted ODE solution.

            A terminal event can end an optimization-phase solve before an RHS
            evaluation is retained in the intermediate logs.  In that case the
            corresponding flow has no resolvable trajectory and contributes
            zero over the (typically degenerate) phase, rather than causing a
            shape error during process-boundary bookkeeping.
            """
            trajectory = np.asarray(values, dtype=float)
            if trajectory.shape == (n_points,):
                return trajectory
            return np.zeros(n_points, dtype=float)

        D_list = _aligned_log_or_zeros(self.active_ODE.D_list)
        D_base_list = _aligned_log_or_zeros(self.active_ODE.D_base_list)
        D_acid_list = _aligned_log_or_zeros(self.active_ODE.D_acid_list)

        # Extract initial conditions for easy access
        init_conds_array = sol.y[:,0]
        Xv_init = init_conds_array[0]
        C_Glc_init = init_conds_array[1]
        C_Gln_init = init_conds_array[2]
        C_Lac_init = init_conds_array[3]
        C_Amm_init = init_conds_array[4]
        V_init = init_conds_array[5]
        C_O2_init = init_conds_array[6]
        C_CT_init = init_conds_array[7]
        C_H_init = init_conds_array[8]
        C_Na_init = init_conds_array[9]
        C_AA_init = init_conds_array[10]

        # Extract final conditions for easy access
        final_conds_array = sol.y[:,-1]
        Xv_final = final_conds_array[0]
        C_Glc_final = final_conds_array[1]
        C_Gln_final = final_conds_array[2]
        C_Lac_final = final_conds_array[3]
        C_Amm_final = final_conds_array[4]
        V_final = final_conds_array[5]
        C_O2_final = final_conds_array[6]
        C_CT_final = final_conds_array[7]
        C_H_final = final_conds_array[8]
        C_Na_final = final_conds_array[9]
        C_AA_final = final_conds_array[10]

        # Other solution extractions
        reactor_volume_array = sol.y[5]

        # Hardcoded feed concentrations for lactate and ammonium as placeholders for potential future changes
        C_Lac_feed = 0.0 # mmol/L, assuming no lactate in the feed
        C_Amm_feed = 0.0 # mmol/L, assuming no ammonium in the feed

        def _incremental_trapezoid(rate_array: np.ndarray) -> np.ndarray:
            """Return per-timestep amounts via trapezoidal integration (no cumsum)."""
            incremental = np.zeros(n_points)
            if n_points > 1:
                incremental[1:] = 0.5 * (rate_array[:-1] + rate_array[1:]) * dt_array
            return incremental

        cells_in_array = np.zeros(n_points)
        cells_out_array = np.zeros(n_points)
        batch_medium_in_array = np.zeros(n_points)
        feed_medium_in_array = np.zeros(n_points)
        base_in_array = np.zeros(n_points)
        acid_in_array = np.zeros(n_points)
        glucose_in_array = np.zeros(n_points)
        glutamine_in_array = np.zeros(n_points)
        AA_in_array = np.zeros(n_points)
        lactate_in_array = np.zeros(n_points)
        ammonium_in_array = np.zeros(n_points)
        outlet_volume_array = np.zeros(n_points)
        permeate_volume_array = np.zeros(n_points)
        glucose_out_array = np.zeros(n_points)
        glutamine_out_array = np.zeros(n_points)
        AA_out_array = np.zeros(n_points)
        glucose_net_array = np.zeros(n_points)
        glutamine_net_array = np.zeros(n_points)
        AA_net_array = np.zeros(n_points)
        lactate_out_array = np.zeros(n_points)
        ammonium_out_array = np.zeros(n_points)
        oxygen_in_array = np.zeros(n_points)
        oxygen_out_array = np.zeros(n_points)
        CT_in_array = np.zeros(n_points)
        CT_out_array = np.zeros(n_points)
        lactate_net_array = np.zeros(n_points)
        ammonium_net_array = np.zeros(n_points)
        oxygen_net_array = np.zeros(n_points)
        CT_net_array = np.zeros(n_points)
        water_produced_array = np.zeros(n_points)
        heat_generated_array = np.zeros(n_points)
        harvest_volume_array = np.zeros(n_points)
        cells_net_array = np.zeros(n_points)
        cells_lost_array = np.zeros(n_points)  # Biomass leaving the boundary as non-harvested loss (e.g. perfusion separator leak)

        # Get the OTR, CTR, q_H2O, and heat_rate lists from the ODE system if they exist and have the expected length; otherwise use zero arrays. This allows the cumulative trapezoidal integration to proceed without errors even if these lists are not defined for the current kinetic model or operating mode. 
        if len(getattr(self.active_ODE, 'OTR_list', np.array([]))) == n_points:
            OTR_list = np.asarray(self.active_ODE.OTR_list, dtype=float)
        else:
            OTR_list = np.zeros(n_points)
        if len(getattr(self.active_ODE, 'CTR_list', np.array([]))) == n_points:
            CTR_list = np.asarray(self.active_ODE.CTR_list, dtype=float)
        else:
            CTR_list = np.zeros(n_points)
        if len(getattr(self.active_ODE, 'q_H2O_list', np.array([]))) == n_points:
            q_H2O_list = np.asarray(self.active_ODE.q_H2O_list, dtype=float)
        else:
            q_H2O_list = np.zeros(n_points)
        if len(getattr(self.active_ODE, 'heat_rate_list', np.array([]))) == n_points:
            heat_rate_list = np.asarray(self.active_ODE.heat_rate_list, dtype=float)
        else:
            heat_rate_list = np.zeros(n_points)

        # Incremental gas transfer per timestep. OTR/CTR results are stored in
        # temp vars; oxygen_in_array/CT_in_array receive their initial-charge
        # entries in the mode-specific blocks below, then the incremental gas
        # transfer is added on top.
        oxygen_transfer_rate = OTR_list * sol.y[5]  # mmol/L/h * L = mmol/h
        ottr_incremental = _incremental_trapezoid(oxygen_transfer_rate)

        co2_transfer_rate = CTR_list * sol.y[5] # mmol/L/h * L = mmol/h
        ct_in_incremental = _incremental_trapezoid(np.maximum(co2_transfer_rate, 0.0))
        CT_out_array[:] = _incremental_trapezoid(np.maximum(-co2_transfer_rate, 0.0)) # Account for CO2 leaving the system when CTR is negative.

        if n_points > 1:
            # Water production by cell metabolism
            water_production_rate = q_H2O_list * sol.y[0] * sol.y[5] # mmol/cell/h * cells/L * L = mmol/h
            heat_generation_rate = heat_rate_list * sol.y[5] # J/L/h * L = J/h
            water_produced_array[:] = _incremental_trapezoid(water_production_rate)
            heat_generated_array[:] = _incremental_trapezoid(heat_generation_rate)

            # Base addition
            base_flow_rate = D_base_list * sol.y[5]
            base_in_array[:] = _incremental_trapezoid(base_flow_rate)

            # Acid addition
            acid_flow_rate = D_acid_list * sol.y[5]
            acid_in_array[:] = _incremental_trapezoid(acid_flow_rate)

        if isinstance(self.active_ODE, BatchODESystem):
            if include_initial_charge:
                # Count inputs: initial conditions (incremental at t=0)
                cells_in_array[0] = Xv_init * V_init
                batch_medium_in_array[0] = V_init
                glucose_in_array[0] = C_Glc_init * V_init
                glutamine_in_array[0] = C_Gln_init * V_init
                AA_in_array[0] = C_AA_init * V_init
                lactate_in_array[0] = C_Lac_init * V_init
                ammonium_in_array[0] = C_Amm_init * V_init
                oxygen_in_array[0] = C_O2_init * V_init
                oxygen_in_array += ottr_incremental
                CT_in_array[0] = C_CT_init * V_init
                CT_in_array += ct_in_incremental
            # Count outputs: harvest at the end of the batch phase
            if is_final_phase:
                outlet_volume_array[-1] = V_final
                glucose_out_array[-1] = V_final * C_Glc_final
                glutamine_out_array[-1] = V_final * C_Gln_final
                AA_out_array[-1] = V_final * C_AA_final
                lactate_out_array[-1] = V_final * C_Lac_final
                ammonium_out_array[-1] = V_final * C_Amm_final
                oxygen_out_array[-1] = V_final * C_O2_final
                CT_out_array[-1] += V_final * C_CT_final
                harvest_volume_array[-1] = V_final
                cells_out_array[-1] = V_final * Xv_final

        elif isinstance(self.active_ODE, FedBatchODESystem):
            # Count inputs: initial conditions at t=0 (only when this is the first phase)
            if include_initial_charge:
                cells_in_array[0] = Xv_init * V_init
                batch_medium_in_array[0] = V_init
                glucose_in_array[0] = C_Glc_init * V_init
                glutamine_in_array[0] = C_Gln_init * V_init
                AA_in_array[0] = C_AA_init * V_init
                lactate_in_array[0] = C_Lac_init * V_init
                ammonium_in_array[0] = C_Amm_init * V_init
                oxygen_in_array[0] = C_O2_init * V_init
                CT_in_array[0] = C_CT_init * V_init
            # Feed additions: incremental per timestep
            if n_points > 1:
                feed_flow_rate = D_list * reactor_volume_array      # 
                feed_medium_in_array[:] = _incremental_trapezoid(feed_flow_rate)
                glucose_in_array += params.fed_batch.C_Glc_feed * feed_medium_in_array
                glutamine_in_array += params.fed_batch.C_Gln_feed * feed_medium_in_array
                AA_in_array += params.fed_batch.C_AA_feed * feed_medium_in_array
                lactate_in_array += C_Lac_feed * feed_medium_in_array
                ammonium_in_array += C_Amm_feed * feed_medium_in_array
                oxygen_in_array += ottr_incremental
                oxygen_in_array += float(params.shared_medium_properties.C_O2) * feed_medium_in_array
                CT_in_array += ct_in_incremental
                CT_in_array += float(params.shared_medium_properties.C_CT) * feed_medium_in_array
            # Count outputs: harvest at the end of the fed-batch phase
            if is_final_phase: # Harvest the entire reactor contents at the end of the final batch phase.
                outlet_volume_array[-1] = V_final
                glucose_out_array[-1] = V_final * C_Glc_final
                glutamine_out_array[-1] = V_final * C_Gln_final
                AA_out_array[-1] = V_final * C_AA_final
                lactate_out_array[-1] = V_final * C_Lac_final
                ammonium_out_array[-1] = V_final * C_Amm_final
                oxygen_out_array[-1] = V_final * C_O2_final
                CT_out_array[-1] += V_final * C_CT_final
                harvest_volume_array[-1] = V_final
                cells_out_array[-1] = V_final * Xv_final

        # In continuous operating modes such as chemostat and perfusion modes, the input and output is only what comes in and out during the culture time in that mode.
        # The initial charge is only counted on the process boundary when the continuous mode is the first, or only phase of a cultivation.
        # If it is not the first phase, the initial charge was already counted in the preceding phase.
        elif isinstance(self.active_ODE, ChemostatODESystem):
            # Count inputs: initial conditions at t=0
            if include_initial_charge: 
                cells_in_array[0] = Xv_init * V_init
                batch_medium_in_array[0] = V_init
                glucose_in_array[0] = C_Glc_init * V_init
                glutamine_in_array[0] = C_Gln_init * V_init
                AA_in_array[0] = C_AA_init * V_init
                lactate_in_array[0] = C_Lac_init * V_init
                ammonium_in_array[0] = C_Amm_init * V_init
                oxygen_in_array[0] = C_O2_init * V_init
                CT_in_array[0] = C_CT_init * V_init
            # Incremental gas transfer + continuous feed additions
            if n_points > 1:
                feed_flow_rate = D_list * reactor_volume_array
                feed_medium_in_array[:] = _incremental_trapezoid(feed_flow_rate)
                glucose_in_array += params.chemostat.C_Glc_feed * feed_medium_in_array
                glutamine_in_array += params.chemostat.C_Gln_feed * feed_medium_in_array
                AA_in_array += params.chemostat.C_AA_feed * feed_medium_in_array
                lactate_in_array += C_Lac_feed * feed_medium_in_array
                ammonium_in_array += C_Amm_feed * feed_medium_in_array
                oxygen_in_array += float(params.shared_medium_properties.C_O2) * feed_medium_in_array
                oxygen_in_array += ottr_incremental
                CT_in_array += float(params.shared_medium_properties.C_CT) * feed_medium_in_array
                CT_in_array += ct_in_incremental

            # Count outputs: incremental continuous outflow per timestep
                outlet_flow_rate = (D_list + D_base_list + D_acid_list) * reactor_volume_array
                outlet_volume_array[:] = _incremental_trapezoid(outlet_flow_rate)
                harvest_volume_array[:] = _incremental_trapezoid(outlet_flow_rate)
                glucose_out_array[:] = _incremental_trapezoid(sol.y[1] * outlet_flow_rate)
                glutamine_out_array[:] = _incremental_trapezoid(sol.y[2] * outlet_flow_rate)
                AA_out_array[:] = _incremental_trapezoid(sol.y[10] * outlet_flow_rate)
                lactate_out_array[:] = _incremental_trapezoid(sol.y[3] * outlet_flow_rate)
                ammonium_out_array[:] = _incremental_trapezoid(sol.y[4] * outlet_flow_rate)
                oxygen_out_array[:] = _incremental_trapezoid(sol.y[6] * outlet_flow_rate)
                CT_out_array[:] += _incremental_trapezoid(sol.y[7] * outlet_flow_rate)
                cells_out_array[:] = _incremental_trapezoid(sol.y[0] * outlet_flow_rate)

            # Count outputs: harvest at the end of the chemostat phase.
            if is_final_phase:
                outlet_volume_array[-1] += V_final
                glucose_out_array[-1] += V_final * C_Glc_final
                glutamine_out_array[-1] += V_final * C_Gln_final
                AA_out_array[-1] += V_final * C_AA_final
                lactate_out_array[-1] += V_final * C_Lac_final
                ammonium_out_array[-1] += V_final * C_Amm_final
                oxygen_out_array[-1] += V_final * C_O2_final
                CT_out_array[-1] += V_final * C_CT_final
                harvest_volume_array[-1] += V_final
                cells_out_array[-1] += Xv_final * V_final

        elif isinstance(self.active_ODE, PerfusionODESystem):
            # Count inputs: initial conditions at t=0
            if include_initial_charge:
                cells_in_array[0] = Xv_init * V_init
                batch_medium_in_array[0] = V_init
                glucose_in_array[0] = C_Glc_init * V_init
                glutamine_in_array[0] = C_Gln_init * V_init
                AA_in_array[0] = C_AA_init * V_init
                lactate_in_array[0] = C_Lac_init * V_init
                ammonium_in_array[0] = C_Amm_init * V_init
                oxygen_in_array[0] = C_O2_init * V_init
                CT_in_array[0] = C_CT_init * V_init

            # Incremental gas transfer + continuous feed additions
            if n_points > 1:
                oxygen_in_array += ottr_incremental
                CT_in_array += ct_in_incremental
                feed_flow_rate = D_list * reactor_volume_array
                feed_medium_in_array[:] = _incremental_trapezoid(feed_flow_rate)
                glucose_in_array += params.perfusion.C_Glc_feed * feed_medium_in_array
                glutamine_in_array += params.perfusion.C_Gln_feed * feed_medium_in_array
                AA_in_array += params.perfusion.C_AA_feed * feed_medium_in_array
                lactate_in_array += C_Lac_feed * feed_medium_in_array
                ammonium_in_array += C_Amm_feed * feed_medium_in_array
                oxygen_in_array += float(params.shared_medium_properties.C_O2) * feed_medium_in_array
                CT_in_array += float(params.shared_medium_properties.C_CT) * feed_medium_in_array

            # Count outputs: incremental continuous outflow and bleed flow per timestep
                B_list = np.full(n_points, self.active_ODE.B)
                permeate_flow_rate = (D_list + D_base_list + D_acid_list - B_list) * reactor_volume_array # For future use if permeate tracking is desired
                bleed_flow_rate = B_list * reactor_volume_array
                outlet_flow_rate = permeate_flow_rate + bleed_flow_rate
                outlet_volume_array[:] = _incremental_trapezoid(outlet_flow_rate)
                permeate_volume_array[:] = _incremental_trapezoid(permeate_flow_rate)
                harvest_volume_array[:] = _incremental_trapezoid(bleed_flow_rate)
                glucose_out_array[:] = _incremental_trapezoid(sol.y[1] * outlet_flow_rate)
                glutamine_out_array[:] = _incremental_trapezoid(sol.y[2] * outlet_flow_rate)
                AA_out_array[:] = _incremental_trapezoid(sol.y[10] * outlet_flow_rate)
                lactate_out_array[:] = _incremental_trapezoid(sol.y[3] * outlet_flow_rate)
                ammonium_out_array[:] = _incremental_trapezoid(sol.y[4] * outlet_flow_rate)
                oxygen_out_array[:] = _incremental_trapezoid(sol.y[6] * outlet_flow_rate)
                CT_out_array[:] += _incremental_trapezoid(sol.y[7] * outlet_flow_rate)
                # Cells cross the boundary via the bleed (harvested) and via the imperfect
                # separator leaking into the permeate (lost). This mirrors the ODE removal
                # dXv/dt = Xv*(mu - B - eff_loss) with eff_loss = H*(1 - eff_separator) and
                # H*V = permeate_flow_rate, so the incremental biomass output stays consistent
                # with the integrated growth and the elemental closure balances.
                eff_separator = float(self.active_ODE.eff_separator)
                cells_lost_rate = sol.y[0] * permeate_flow_rate * (1.0 - eff_separator)
                cells_harvested_rate = sol.y[0] * bleed_flow_rate
                cells_lost_array[:] = _incremental_trapezoid(cells_lost_rate)
                cells_out_array[:] = _incremental_trapezoid(cells_harvested_rate + cells_lost_rate)
           
           # Count outputs: harvest at the end of the perfusion phase.
            if is_final_phase:  
                outlet_volume_array[-1] += V_final
                glucose_out_array[-1] += V_final * C_Glc_final
                glutamine_out_array[-1] += V_final * C_Gln_final
                AA_out_array[-1] += V_final * C_AA_final
                lactate_out_array[-1] += V_final * C_Lac_final
                ammonium_out_array[-1] += V_final * C_Amm_final
                oxygen_out_array[-1] += V_final * C_O2_final
                CT_out_array[-1] += V_final * C_CT_final
                harvest_volume_array[-1] += V_final
                cells_out_array[-1] += Xv_final * V_final

        else:
            raise ValueError("Unknown error occurred. Please check the code.")

        # Calculate net production/consumption. Convention: production positive,
        # consumption negative.  All arrays use net = out − in.
        cells_net_array[:] = cells_out_array - cells_in_array
        glucose_net_array[:] = glucose_out_array - glucose_in_array
        glutamine_net_array[:] = glutamine_out_array - glutamine_in_array
        AA_net_array[:] = AA_out_array - AA_in_array
        lactate_net_array[:] = lactate_out_array - lactate_in_array
        ammonium_net_array[:] = ammonium_out_array - ammonium_in_array
        oxygen_net_array[:] = oxygen_out_array - oxygen_in_array
        CT_net_array[:] = CT_out_array - CT_in_array

        return (
            batch_medium_in_array,
            feed_medium_in_array,
            base_in_array,
            acid_in_array,
            outlet_volume_array,
            permeate_volume_array,
            harvest_volume_array,
            glucose_in_array,
            glucose_out_array,
            glucose_net_array,
            glutamine_in_array,
            glutamine_out_array,
            glutamine_net_array,
            AA_in_array,
            AA_out_array,
            AA_net_array,
            lactate_in_array,
            lactate_out_array,
            lactate_net_array,
            ammonium_in_array,
            ammonium_out_array,
            ammonium_net_array,
            oxygen_in_array,
            oxygen_out_array,
            oxygen_net_array,
            CT_in_array,
            CT_out_array,
            CT_net_array,
            water_produced_array,
            heat_generated_array,
            cells_in_array,
            cells_out_array,
            cells_net_array,
            cells_lost_array
        )

    def _determine_termination(self, sol, event_functions: list) -> tuple[float, str]:
        """Determine a phase end time and human-readable termination reason.

        Parameters
        ----------
        sol : scipy.integrate.OdeResult
            ODE integration result.
        event_functions : list[callable]
            Events supplied to the solver, used to map triggered events to
            readable reason text.

        Returns
        -------
        tuple[float, str]
            End time in h and termination description.
        """
        termination_reason = "Unknown termination"
        if sol.t_events and any(len(e) > 0 for e in sol.t_events):
            event_times = [e[0] for e in sol.t_events if len(e) > 0]
            t_end = min(event_times)

            # Identify which event terminated the simulation
            triggered_event_idx = None
            for idx, event_time_list in enumerate(sol.t_events):
                if len(event_time_list) > 0 and event_time_list[0] == t_end:
                    triggered_event_idx = idx
                    break

            # Get termination reason from event function name
            if triggered_event_idx is not None and triggered_event_idx < len(event_functions):
                event_func = event_functions[triggered_event_idx]
                event_name = event_func.__name__ if hasattr(event_func, '__name__') else str(event_func)
                # Convert event function name to readable message
                termination_reason = event_name.replace('_', ' ').replace('event', '').strip().capitalize()
        else:
            t_end = sol.t[-1]
            if sol.status == 0:
                termination_reason = "Maximum time reached"
            else:
                termination_reason = f"Integration terminated: {sol.message}"
        return t_end, termination_reason

    def _build_batch_data(self, sol, termination_reason: str,
                          is_final_phase: bool = True,
                          include_initial_charge: bool = False) -> BatchData:
        """Build :class:`BatchData` from a solved phase and logged signals.

        Parameters
        ----------
        sol : scipy.integrate.OdeResult
            ODE integration result.
        termination_reason : str
            Human-readable phase stop reason.
        is_final_phase : bool, default=True
            Whether final reactor emptying is recorded as an external output.
        include_initial_charge : bool, default=False
            Whether the initial reactor state is recorded as an external input.

        Returns
        -------
        BatchData
            Phase result with boundary-flow increments and available traces.
        """

        (
            batch_medium_in, feed_medium_in, base_in, acid_in,
            outlet_volume, permeate_volume, harvest_volume,
            glucose_in, glucose_out, glucose_net,
            glutamine_in, glutamine_out, glutamine_net,
            AA_in, AA_out, AA_net,
            lactate_in, lactate_out, lactate_net,
            ammonium_in, ammonium_out, ammonium_net,
            oxygen_in, oxygen_out, oxygen_net,
            CT_in, CT_out, CT_net,
            water_produced, heat_generated,
            cells_in, cells_out, cells_net, cells_lost) = self._calculate_medium_and_nutrient_use(
            sol, is_final_phase=is_final_phase, include_initial_charge=include_initial_charge,
        )
        cumulative_arrays = {
            'batch_medium_in': batch_medium_in,
            'feed_medium_in':  feed_medium_in,
            'base_in':          base_in,
            'acid_in':          acid_in,
            'outlet_volume':    outlet_volume,
            'permeate_volume':  permeate_volume,
            'glucose_in':      glucose_in,
            'glucose_out':      glucose_out,
            'glucose_net':      glucose_net,
            'glutamine_in':    glutamine_in,
            'glutamine_out':    glutamine_out,
            'glutamine_net':    glutamine_net,
            'AA_in':           AA_in,
            'AA_out':           AA_out,
            'AA_net':           AA_net,
            'lactate_in':      lactate_in,
            'lactate_out':      lactate_out,
            'lactate_net':      lactate_net,
            'ammonium_in':     ammonium_in,
            'ammonium_out':     ammonium_out,
            'ammonium_net':     ammonium_net,
            'oxygen_in':        oxygen_in,
            'oxygen_out':       oxygen_out,
            'oxygen_net':       oxygen_net,
            'CT_in':           CT_in,
            'CT_out':          CT_out,
            'CT_net':          CT_net,
            'water_produced':   water_produced,
            'heat_generated':   heat_generated,
            'harvest_volume':   harvest_volume,
            'cells_in':       cells_in,
            'cells_out':      cells_out,
            'cells_net':      cells_net,
            'cells_lost':     cells_lost,
        }

        n_points = len(sol.t)

        # Pull all ODE-logged arrays generically using the field list from BatchDataAccumulator,
        # excluding the cumulative fields which are computed above instead.
        ode_arrays = {
            field: getattr(self.active_ODE, field)[:n_points].copy()
            for field in BatchDataAccumulator._1D_FIELDS
            if field not in BatchDataAccumulator._PROCESS_CUMULATIVE_FIELDS
        }
        # 2D buffer concentration array (n_buffers × n_points)
        C_buffer_arr = getattr(self.active_ODE, 'C_buffer_list', np.array([]))
        if C_buffer_arr.size > 0:
            C_buffer_arr = C_buffer_arr[:, :n_points].copy()

        return BatchData(
            batch_number=self._current_batch_number,
            initial_conditions=sol.y[:, 0].tolist(),
            t_end=sol.t[-1],
            Xv_final=sol.y[0][-1],
            V_final=sol.y[5][-1],
            t_points=sol.t,
            y_points=sol.y,
            termination_reason=termination_reason,
            C_buffer_list=C_buffer_arr if C_buffer_arr.size > 0 else None,
            **cumulative_arrays,
            **ode_arrays,
        )

    def _solve_ode(self, init_conds: list, event_functions: list, t_limit: float = None):
        """Integrate the active ODE system with terminal process events.

        Parameters
        ----------
        init_conds : list[float]
            Initial state in ODE state-vector order.
        event_functions : list[callable]
            Terminal events passed to :func:`scipy.integrate.solve_ivp`.
        t_limit : float, optional
            Integration limit in h. Defaults to ``max_culture_age``.

        Returns
        -------
        scipy.integrate.OdeResult
            Adaptive integration result.
        """
        return solve_ivp(
            fun=self.active_ODE.rhs,
            t_span=(0, t_limit if t_limit is not None else self.max_culture_age),
            y0=init_conds,
            dense_output=False,
            args=(self.parameters,), # Required for event functions
            events=event_functions,
            method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=1 # every step is logged. This prevents large errors in trapezoidal integration of the logged rates. The ODE solver will still choose smaller steps if needed for accuracy.
        )

    def _has_invalid_initial_pH(self, init_conds: list[float]) -> bool:
        """Return whether a phase starts outside the inclusive pH tolerance.

        ``solve_ivp`` only detects an event after a directional root crossing.
        A phase that starts beyond the pH band therefore requires this explicit
        preflight check. The existing event remains the single source of truth
        for the configured hydrogen-ion limits; a zero event value is valid.
        """
        return pH_deviation_event(0.0, init_conds, self.parameters) < 0.0

    def _build_initial_pH_rejection(self, init_conds: list[float]) -> BatchData:
        """Create a zero-duration result without evaluating the ODE RHS.

        Rejected entry states are retained in normal simulation reporting so a
        campaign can audit the cause without treating the state as a physical
        zero-time charge, harvest, or controller action.
        """
        initial_state = np.asarray(init_conds, dtype=float).reshape(-1, 1)
        zeros = np.zeros(1, dtype=float)
        n_buffers = max(initial_state.shape[0] - 12, 0)
        buffer_state = initial_state[12:, :].copy() if n_buffers else None
        arrays = {
            field: zeros.copy()
            for field in BatchDataAccumulator._1D_FIELDS
        }
        return BatchData(
            batch_number=self._current_batch_number,
            initial_conditions=list(init_conds),
            t_end=0.0,
            Xv_final=float(initial_state[0, 0]),
            V_final=float(initial_state[5, 0]),
            t_points=zeros.copy(),
            y_points=initial_state,
            termination_reason="Invalid initial pH",
            initial_state_rejected=True,
            C_buffer_list=buffer_state,
            **arrays,
        )
    
    def _run_single_operation(self, init_conds: list[float], store_batch: bool = True,
                               t_limit: float = None, event_functions: list = None,
                               is_final_phase: bool = True,
                               include_initial_charge: bool = False) -> BatchData:
        """Run one ODE phase and construct its :class:`BatchData` result.

        Parameters
        ----------
        init_conds : list[float]
            ODE initial state, including optional buffer states after
            ``C_Osm_bg``.
        store_batch : bool, default=True
            Whether to append the result directly to :attr:`data.batches`.
        t_limit : float, optional
            Phase integration limit in h.
        event_functions : list[callable]
            Pre-built terminal events. This argument is required.
        is_final_phase : bool, default=True
            Whether to record terminal reactor emptying as an external output.
        include_initial_charge : bool, default=False
            Whether to record the initial state as an external input.

        Returns
        -------
        BatchData
            Completed phase result.

        Raises
        ------
        ValueError
            If no event list is provided.
        """
        if event_functions is None:
            raise ValueError(
                "_run_single_operation requires event_functions to be provided explicitly. "
                "Pass self._phase_events[phase_name] from _run_batch_phases."
            )

        # Reset logging arrays in ODE system for new simulation
        self.active_ODE.reset_logging()

        # Terminal events only identify directional crossings. Reject a phase
        # that already starts outside the pH band before it reaches solve_ivp.
        if self._has_invalid_initial_pH(init_conds):
            batch_data = self._build_initial_pH_rejection(init_conds)
            if store_batch:
                self.data.batches.append(batch_data)
                self._current_batch_number += 1
            return batch_data

        # Run simulation
        sol = self._solve_ode(init_conds, event_functions, t_limit=t_limit)

        # Clamp state variables in-place to prevent tiny numerical
        # overshoots from solve_ivp's interpolation.  This is the single
        # source of truth — all downstream methods (_determine_termination,
        # _calculate_medium_and_nutrient_use, _build_batch_data) read
        # sol.y directly, so clamping here keeps everything consistent.
        np.clip(sol.y, 0, None, out=sol.y)
        sol.y[5] = np.maximum(sol.y[5], 1e-6)  # volume floor

        # Determine simulation stop time and termination reason
        t_end, termination_reason = self._determine_termination(sol, event_functions)

        # Finalize logging: interpolate all logged values to solution timepoints
        self.active_ODE.finalize_logging(sol.t)
        

        # Create and store batch data object
        batch_data = self._build_batch_data(
            sol, termination_reason=termination_reason,
            is_final_phase=is_final_phase,
            include_initial_charge=include_initial_charge,
        )

        # Store in simulation data and increment batch counter
        if store_batch:
            self.data.batches.append(batch_data)
            self._current_batch_number += 1
        
        return batch_data

### Phase level methods ###
    @staticmethod
    def _build_next_phase_init_conds(phase_data: BatchData) -> list[float]:
        """Return the terminal state of one phase as the initial state of the next.

        Carries forward all state variables including any non-carbonate buffer
        states (indices >= 11) so that buffer concentrations are continuous
        across phase switches within a batch.
        """
        y = phase_data.y_points
        init = [
            phase_data.Xv_final,
            y[1, -1],
            y[2, -1],
            y[3, -1],
            y[4, -1],
            phase_data.V_final,
            y[6, -1],
            y[7, -1],
            y[8, -1],
            y[9, -1],
            y[10, -1],
            y[11, -1],
        ]
        # Append any buffer state variables (rows 12+)
        if y.shape[0] > 12:
            for i in range(12, y.shape[0]):
                init.append(y[i, -1])
        return init

    def _run_batch_phases(
        self,
        init_conds: list,
        phase_names: list,
        inoculum: dict | None = None,
        t_limit: float = None,
    ) -> BatchDataAccumulator:
        """Run and combine all ODE phases for one logical batch.

        Parameters
        ----------
        init_conds : list[float]
            Initial state for the first phase, with optional buffer states.
        phase_names : list[str]
            Ordered keys in :attr:`ode_systems`.
        inoculum : dict, optional
            Carryover data from the prior batch. When provided, its internal
            input contribution is removed from external process accounting.
        t_limit : float, optional
            Remaining vessel-lifetime limit in h.

        Returns
        -------
        BatchDataAccumulator
            Concatenated phase trajectories and per-batch totals.
        """
        accumulator = BatchDataAccumulator()

        for phase_idx, phase_name in enumerate(phase_names):
            self.active_ODE = self.ode_systems[phase_name]
            is_final_phase = (phase_idx == len(phase_names) - 1)
            phase_events = self._get_phase_events_for_batch_phase(phase_name, phase_names, phase_idx)
            # The first phase always counts the initial reactor charge on the
            # process boundary.  Subsequent phases (e.g. FedBatch after Batch,
            # or Chemostat after FedBatch) do not, because the charge was
            # already accounted for in the preceding phase.
            include_initial_charge = phase_idx == 0
            # Run the phase simulation and get the BatchData result
            phase_data = self._run_single_operation(
                init_conds, store_batch=False, t_limit=t_limit,
                event_functions=phase_events,
                is_final_phase=is_final_phase,
                include_initial_charge=include_initial_charge,
            )

            accumulator.add_phase(phase_data, phase_name=phase_name)

            # A failed entry-state validation is not a phase transition. Do
            # not propagate its state into a later FEDBATCH/CHEMOSTAT/PERFUSION
            # phase within this logical batch.
            if phase_data.initial_state_rejected:
                break

            # Apply inoculum carryover correction if there is an inoculum and this is the first phase of a batch.
            if inoculum is not None and include_initial_charge:
                accumulator.apply_internal_inoculum_input_correction(inoculum)

            # Pass final state of this phase as initial conditions for the next phase
            if phase_idx < len(phase_names) - 1:
                init_conds = self._build_next_phase_init_conds(phase_data)

            # Subtract time spent in this phase from total limit time
            if t_limit is not None:
                t_limit -= phase_data.t_end

        return accumulator

    def _get_phase_names_for_current_batch(self, fresh_vessel_lifetime: bool) -> list[str]:
        """Return the ordered ODE phases for the next logical batch.

        Most modes keep the static factory order.  Startup-fedbatch modes run a
        one-time FEDBATCH fill at the beginning of each fresh vessel lifetime,
        then continue as BATCH-only cycles until the lifetime resets.
        """
        if self.operating_mode in STARTUP_FEDBATCH_MODES:
            if fresh_vessel_lifetime:
                return ['BATCH','FEDBATCH', 'BATCH']
            return ['BATCH']
        return list(self.ode_systems.keys())

    def _get_next_batch_base_conditions(self, fresh_vessel_lifetime: bool) -> dict:
        """Return base initial conditions for preparing the *next* batch.

        For startup-fedbatch modes, once the startup FEDBATCH has filled the
        reactor in a given vessel lifetime, subsequent BATCH-only cycles should
        start at full working volume.  A fresh vessel lifetime resets to the
        configured fed-batch start volume (self.base['V_init']).
        """
        next_base = dict(self.base)
        if self.operating_mode in STARTUP_FEDBATCH_MODES and not fresh_vessel_lifetime:
            next_base['V_init'] = float(self.parameters.reactor.V_max)
        return next_base

### Full simulation methods ###
 
    # Determines the inoculum volume in case the mode is a repeated mode, or determines that the simulation should end.
    def _calculate_next_inoculum(
        self,
        accumulator: BatchDataAccumulator,
        params: SimulationParameters,
        V_init: float,
        X_init: float,
    ) -> tuple[float, bool, float]:
        """Determine whether a viable carryover inoculum can start the next batch.

        Parameters
        ----------
        accumulator : BatchDataAccumulator
            Completed current batch.
        params : SimulationParameters
            Parameters supplying the minimum viable cell density and reactor volume limits.
        V_init : float
            Next-batch starting volume in L.
        X_init : float
            Base target inoculum density in cells/L (defined at full reactor working volume V_max).

        Returns
        -------
        tuple[float, bool, float]
            Required carryover volume in L, a flag indicating that no next
            batch should be attempted, and the target initial cell density
            X_next_init for the next batch. Batches stop when final density is too
            low or required carryover exceeds 30% of final volume.
        """
        Xv_final = accumulator.y_points[0, -1]
        V_final  = accumulator.y_points[5, -1]

        if Xv_final < params.cell_properties.Xv_min * 1.5:  # Too few cells for a viable inoculum
            return 0.0, True, 0.0

        # Anchor baseline cell charge to total cell inventory N_target = X_init * V_max
        V_max = float(params.reactor.V_max)
        X_target_init = X_init * (V_max / V_init)

        # 1. Nominal step (5% minimum carryover for industrial draw and fill feasability)
        V_inoc_min = 0.05 * V_final
        X_nominal = (Xv_final * V_inoc_min) / V_init

        # 2. Healthy batch: nominal carryover meets or exceeds inventory target
        if X_nominal >= X_target_init:
            return V_inoc_min, False, X_nominal

        # 3. Sub-optimal batch: expand carryover up to 30% to hit inventory target
        V_inoc_needed = (X_target_init * V_init) / Xv_final
        if V_inoc_needed <= 0.30 * V_final:
            return V_inoc_needed, False, X_target_init

        # Required carryover exceeds 30% cap -> abort simulation
        return 0.0, True, 0.0
    # If simulation does not end, prepare the initial conditions for the next batch by mixing the inoculum carryover with fresh medium.
    def _prepare_next_batch_conditions(
        self,
        accumulator: BatchDataAccumulator,
        V_inoc: float,
        base_conditions: dict,
        X_next_init: float,
    ) -> tuple[dict, dict]:
        """Mix carryover inoculum with fresh medium for the next batch.

        Parameters
        ----------
        accumulator : BatchDataAccumulator
            Completed current batch.
        V_inoc : float
            Retained inoculum volume in L.
        base_conditions : dict
            Fresh-medium initial-state values.
        X_next_init : float
            Effective target or nominal initial cell density for the next batch in cells/L.

        Returns
        -------
        tuple[dict, dict]
            Raw inoculum data for process-boundary correction and mixed next
            initial conditions. Buffer concentrations are included when present.
        """
        V_init = base_conditions['V_init']
        V_medium_add = V_init - V_inoc
        y = accumulator.y_points

        inoculum_data = {
            'V_inoc':     V_inoc,
            'Xv_inoc':    y[0, -1],   # Cell density in inoculum (end-of-batch value)
            'C_Glc_inoc': y[1, -1],
            'C_Gln_inoc': y[2, -1],
            'C_Lac_inoc': y[3, -1],
            'C_Amm_inoc': y[4, -1],
            'C_O2_inoc':  y[6, -1],
            'C_CT_inoc':  y[7, -1],
            'C_H_inoc':   y[8, -1],
            'C_Na_inoc':  y[9, -1],
            'C_AA_inoc':  y[10, -1],
            'C_Osm_bg_inoc': y[11, -1],
        }
        # Carry over non-carbonate buffer concentrations from the inoculum
        n_buf = y.shape[0] - 12
        for i in range(n_buf):
            inoculum_data[f'C_buffer_{i}_inoc'] = y[12 + i, -1]

        def _mix(c_inoc: float, c_base: float) -> float:
            return (c_inoc * V_inoc + c_base * V_medium_add) / V_init

        next_init_conds = {
            'X_init':     X_next_init,
            'V_init':     V_init, 
            'C_Glc_init': _mix(inoculum_data['C_Glc_inoc'], base_conditions['C_Glc_init']),
            'C_Gln_init': _mix(inoculum_data['C_Gln_inoc'], base_conditions['C_Gln_init']),
            'C_Lac_init': _mix(inoculum_data['C_Lac_inoc'], base_conditions['C_Lac_init']),
            'C_Amm_init': _mix(inoculum_data['C_Amm_inoc'], base_conditions['C_Amm_init']),
            'C_O2_init':  _mix(inoculum_data['C_O2_inoc'],  base_conditions['C_O2_init']),
            'C_CT_init':  _mix(inoculum_data['C_CT_inoc'],  base_conditions['C_CT_init']),
            'C_H_init':   _mix(inoculum_data['C_H_inoc'],   base_conditions['C_H_init']),
            'C_Na_init':  _mix(inoculum_data['C_Na_inoc'],  base_conditions['C_Na_init']),
            'C_AA_init':  _mix(inoculum_data['C_AA_inoc'],  base_conditions['C_AA_init']),
            'C_Osm_bg_init': _mix(inoculum_data['C_Osm_bg_inoc'], base_conditions['C_Osm_bg_init']),
        }
        # Mix non-carbonate buffer concentrations (inoculum + fresh medium)
        for i in range(n_buf):
            next_init_conds[f'C_buffer_{i}_init'] = _mix(
                inoculum_data[f'C_buffer_{i}_inoc'],
                base_conditions[f'C_buffer_{i}_init'],
            )
        return inoculum_data, next_init_conds

    # Main simulation loop with batch repetition and yearly projection logic
    @timing_decorator
    def run_simulation(self, store_batches: bool = True, number_of_batches: Optional[int] = None,
                       t_max_simulation: Optional[float] = None,
                       project_yearly: bool = True) -> SimulationData:
        """Run repeated multi-phase operations and compute process KPIs.

        By default (``project_yearly=True``) the simulation runs for a single
        *max_culture_age* window and then scales all additive totals to the full
        yearly uptime.  This delivers the same KPIs as a year-long run at a
        fraction of the compute cost.  Pass ``project_yearly=False`` to disable
        projection and simulate the full yearly_uptime directly.

        Parameters
        ----------
        store_batches : bool, default=True
            Whether to retain trajectories and detailed trace arrays. Disable for
            optimization and sensitivity sweeps.
        number_of_batches : int, optional
            Explicit number of logical batches. Supplying it disables yearly
            projection.
        t_max_simulation : float, optional
            Explicit elapsed-time limit in h. Supplying it disables yearly
            projection and reports the literal window.
        project_yearly : bool, default=True
            Whether to scale additive totals from the default
            ``min(max_culture_age, yearly_uptime)`` window to yearly uptime.
            Ratio KPIs remain unchanged.

        Returns
        -------
        SimulationData
            Results, derived KPIs, literal-window totals, and projected totals.
        """
        if self.parameters is None:
            raise ValueError("Simulation parameters must be set before running the simulation.")
        params = self.parameters

        # Projection is only appropriate for the default repeated-run horizon.
        # Explicit batch counts or time caps should report the literal simulated window.
        _explicit_t_max = t_max_simulation is not None
        effective_project_yearly = project_yearly and not _explicit_t_max and number_of_batches is None

        if t_max_simulation is None:
            if effective_project_yearly:
                t_max_simulation = min(self.max_culture_age, self.yearly_uptime)
            else:
                t_max_simulation = self.yearly_uptime  # Full-year run

        # Clear previous data and reset batch counter
        self.data = SimulationData()
        self.data.simulation_type = f"{str(self.operating_mode)}"
        self.data.feed_strategy = str(self.feed_strategy) if self.feed_strategy is not None else "NA"
        self.data.perfusion_strategy = (
            str(self.perfusion_strategy) if self.perfusion_strategy is not None else "NA"
        )
        self.data.parameters = params
        self._current_batch_number = 0

        # Propagate store_batches flag to ODE systems so they skip detailed
        # logging (30+ list appends per rhs() call) when not needed.
        for ode in self.ode_systems.values():
            ode.log_details = store_batches

        # Pre-allocate batch storage if storing data
        if store_batches:
            batch_storage = []

        # Initialise running totals
        day = 0
        culture_age = 0.0  # Tracks cumulative cultivation time within the current vessel (h)
        totals = {
            **{key: 0.0 for key in _ADDITIVE_TOTAL_KEYS if key != 'number_of_batches'},
            'phase_active_runtime_h': {phase_name: 0.0 for phase_name in self.ode_systems},
            # Preserve phase durations per logical batch as well as their
            # run-wide totals. Optimization validity checks need this to
            # reject a final partial multi-phase batch even when the same
            # downstream phase ran during earlier batches.
            'batch_phase_active_runtime_h': [],
            'last_batch_termination_reason': 'Unknown termination',
            'unique_batch_termination_reasons': set(),
            'batch_time_min_h': float('inf'),
            'batch_time_max_h': 0.0,
            'last_batch_final_volume_L': 0.0,
            'last_batch_final_volume_fraction': 0.0,
            'batch_final_volume_min_L': float('inf'),
            'batch_final_volume_max_L': 0.0,
            'batch_final_volume_min_fraction': float('inf'),
            'batch_final_volume_max_fraction': 0.0,
        }

        # Current initial conditions (start from base, updated after each batch)
        current = dict(self.base)
        final_batch_flag = False
        inoculum_data = None  # None for the first batch (no carryover correction)
        fresh_vessel_lifetime = True

        # Main simulation loop
        while not final_batch_flag and (number_of_batches == None or self._current_batch_number < number_of_batches):
            self._current_batch_number += 1  # Start batch numbering at 1
            current_batch_inoculum_volume = float(inoculum_data['V_inoc']) if inoculum_data is not None else 0.0

            init_conds = [
                current['X_init'],
                current['C_Glc_init'],
                current['C_Gln_init'],
                current['C_Lac_init'],
                current['C_Amm_init'],
                current['V_init'],
                current['C_O2_init'],
                current['C_CT_init'],
                current['C_H_init'],
                current['C_Na_init'],
                current['C_AA_init'],
                current['C_Osm_bg_init'],
            ]
            # Append non-carbonate buffer initial concentrations
            for i in range(len(self._buffer_defs)):
                init_conds.append(current[f'C_buffer_{i}_init'])

            # Reset the shared pH-controller cached state for this new batch so
            # that stale D_base / C_H_current / V_current from the previous batch
            # do not leak into the first RHS evaluation.  Buffer total
            # concentrations are ODE state variables (reset via init_conds above),
            # so this only clears scalar controller cache.  Each ODE system has
            # its own pHControllerModel instance, so reset all of them.
            for ode in self.ode_systems.values():
                ode.pH_controller.reset_buffer_state(current['V_init'])

            # Limit the remaining time for the next batch to the maximum culture age minus the current culture age.
            t_limit = self.max_culture_age - culture_age # Culture age of preceding batches
          

            # Set the ordered list of ODE phases for this batch based on the operating mode and whether this is a fresh vessel lifetime.
            phase_names = self._get_phase_names_for_current_batch(
                fresh_vessel_lifetime=fresh_vessel_lifetime
            )

            # Run all ODE phases for this batch
            accumulator = self._run_batch_phases(
                init_conds, phase_names, inoculum=inoculum_data, t_limit=t_limit
            )
            batch_totals = accumulator.to_batch_totals()
            for phase_name, phase_duration_h in accumulator.phase_durations_h.items():
                totals['phase_active_runtime_h'][phase_name] = (
                    totals['phase_active_runtime_h'].get(phase_name, 0.0)
                    + phase_duration_h
                )
            totals['batch_phase_active_runtime_h'].append({
                'batch_number': self._current_batch_number,
                **{
                    phase_name: float(accumulator.phase_durations_h.get(phase_name, 0.0))
                    for phase_name in phase_names
                },
            })

            # Advance day and culture-age counters
            batch_time = float(batch_totals['batch_time'])
            day += batch_time / 24
            culture_age += batch_time

            ###  Determine starting conditions for the next batch.
            pending_inoculum_data = None
            pending_current = None
            pending_carryover_volume = 0.0
            inoculum_problem = False
           
            # A rejected initial state must end the run. In particular, do not
            # prepare another carryover batch with the same invalid pH state.
            if accumulator.initial_state_rejected:
                inoculum_problem = True
            # Continuous operations always restart from a fresh inoculum.
            # if it is not a continuous mode, check if culture age has been reached
            elif culture_age >= self.max_culture_age: # or self.operating_mode in CONTINUOUS_MODES 
                culture_age = 0.0 # Reset culture age for the next batch, which will start with a fresh inoculum
                inoculum_data = None
                current = dict(self.base) # Reset inoculum conditions to base for the next batch
                fresh_vessel_lifetime = True
            # For batch and fed-batch modes, calculate the next inoculum volume and conditions based on the current batch's final state.
            # or abort the simulation if the next inoculum would be non-viable or would require an excessively large carryover fraction of the current batch's final volume.
            else: 
                next_base_conditions = self._get_next_batch_base_conditions(
                    fresh_vessel_lifetime=False
                )
                V_inoc_next, inoculum_problem, X_next_init = self._calculate_next_inoculum(
                    accumulator, params, next_base_conditions['V_init'], float(self.parameters.initial_conditions.X_inoc)
                )
                if inoculum_problem:
                    pass # Next batch will be flagged as final and the loop will exit, so no need to prepare conditions for the next batch.
                else:
                    # Do not assign the initial conditions yet as we still need to check for other conditions
                    pending_carryover_volume = V_inoc_next
                    pending_inoculum_data, pending_current = self._prepare_next_batch_conditions(
                        accumulator, V_inoc_next, next_base_conditions, X_next_init
                    )
                    fresh_vessel_lifetime = False

            # Determine whether another batch may start based on time limits
            # and batch count. Yearly uptime is the available production
            # horizon, so no turnaround time is inserted between batches.
            reached_batch_limit = (
                number_of_batches is not None
                and self._current_batch_number >= number_of_batches
            )

            batch_end_total_time = totals['total_cultivation_time'] + batch_time
            time_tolerance_h = 1e-9
            within_time_cap = batch_end_total_time < (t_max_simulation - time_tolerance_h)

            # Final decision on whether the next batch starts
            next_batch_starts = (
                (not inoculum_problem) 
                and (not accumulator.initial_state_rejected)
                and (not reached_batch_limit)
                and within_time_cap
            )

            if not next_batch_starts: # Final batch
                final_batch_flag = True # exit loop after this iteration, no need to prepare conditions for the next batch
            elif next_batch_starts: # Prepare carryover corrections for the next batch.
                if pending_carryover_volume > 0.0:
                    accumulator.apply_internal_inoculum_output_correction(pending_carryover_volume)
                    batch_totals = accumulator.to_batch_totals()
                    inoculum_data = pending_inoculum_data
                    current = pending_current

            # Accumulate totals
            totals['total_cultivation_time'] += batch_time
            totals['total_medium_in'] += float(batch_totals['batch_batch_medium_in']) + float(batch_totals['batch_feed_medium_in'])
            for total_key, batch_total_key in _BATCH_TOTAL_ACCUMULATION_MAP.items():
                totals[total_key] += float(batch_totals[batch_total_key])
            totals['last_batch_termination_reason'] = str(batch_totals['batch_termination_reason'])
            totals['unique_batch_termination_reasons'].add(str(batch_totals['batch_termination_reason']))
            totals['batch_time_min_h'] = min(totals['batch_time_min_h'], batch_time)
            totals['batch_time_max_h'] = max(totals['batch_time_max_h'], batch_time)
            batch_final_volume_L = float(batch_totals['batch_final_volume'])
            batch_final_volume_fraction = batch_final_volume_L / float(params.reactor.V_max)
            totals['last_batch_final_volume_L'] = batch_final_volume_L
            totals['last_batch_final_volume_fraction'] = batch_final_volume_fraction
            totals['batch_final_volume_min_L'] = min(totals['batch_final_volume_min_L'], batch_final_volume_L)
            totals['batch_final_volume_max_L'] = max(totals['batch_final_volume_max_L'], batch_final_volume_L)
            totals['batch_final_volume_min_fraction'] = min(
                totals['batch_final_volume_min_fraction'], batch_final_volume_fraction
            )
            totals['batch_final_volume_max_fraction'] = max(
                totals['batch_final_volume_max_fraction'], batch_final_volume_fraction
            )

            # Store batch data if requested.
            if store_batches:
                final_batch_data = accumulator.build(
                    batch_number=self._current_batch_number,
                    init_conds=init_conds.copy(),
                    t_end=batch_time,
                )
                final_batch_data.internal_inoculum_volume_L = current_batch_inoculum_volume
                batch_storage.append(final_batch_data)

        # Deactivate ODE system
        self.active_ODE = None

        # Store batch data if requested
        if store_batches:
            self.data.batches = batch_storage

        # --- Yearly projection ---
        # Record the raw simulated-window values before any scaling.
        totals['number_of_batches'] = self._current_batch_number
        simulated_total_time = totals['total_cultivation_time']
        simulated_n_batches  = self._current_batch_number

        # Apply projection only when the caller did not supply an explicit time cap,
        # projection was requested, and a non-trivial window was actually simulated.
        if effective_project_yearly and simulated_total_time > 0:
            projection_scale_factor = self.yearly_uptime / simulated_total_time
        else:
            projection_scale_factor = 1.0

        simulated_totals = dict(totals)
        projected_totals = dict(totals)
        projection_mode = 'yearly_projection' if projection_scale_factor != 1.0 else 'full_simulation'

        # Scale every additive total from the simulated window to the yearly horizon.
        if projection_scale_factor != 1.0:
            for key in _ADDITIVE_TOTAL_KEYS:
                if key in projected_totals:
                    projected_totals[key] *= projection_scale_factor

        projection_metadata = {
            'projection_scale_factor': projection_scale_factor,
            'projection_mode': projection_mode,
            'projection_window_h': simulated_total_time,
            'projection_target_time_h': (
                self.yearly_uptime if projection_scale_factor != 1.0 else simulated_total_time
            ),
        }

        projected_stats_all = self._compute_summary_statistics_from_totals(projected_totals)
        simulated_stats_all = self._compute_summary_statistics_from_totals(simulated_totals)

        # Compute and store summary statistics
        self.data.summary_stats = self._compute_summary_statistics(projected_stats_all)
        self.data.projected_totals_stats = self._extract_total_stats(projected_stats_all)
        self.data.simulated_window_totals_stats = self._extract_total_stats(simulated_stats_all)
        self.data.projected_totals_stats.update(projection_metadata)
        self.data.simulated_window_totals_stats.update(projection_metadata)
        return self.data

    @classmethod
    #@timing_decorator
    def run_temporary_simulation(cls,
                                 params: SimulationParameters, 
                                 kinetics: Kinetics,
                                 gas_controller: GasController,
                                 operating_mode: OperatingMode,
                                 feed_strategy: FeedStrategy,
                                 perfusion_strategy: PerfusionStrategy,
                                 stoichiometry_mode: StoichiometryMode,
                                 performance_mode: bool,
                                 solver_method: str = 'LSODA',
                                 return_only_summary_stats: bool = False,
                                 wall_timeout: float = 0.0,
                                 debug_failures: bool = False,
                                 **kwargs) -> Union[Dict[str, Any], Literal[0], SimulationData]:
        """Run an isolated simulation for optimization or sensitivity analysis.

        Exceptions are intentionally converted to ``0`` so callers can treat
        failed parameter evaluations as invalid objective-function results.

        Parameters
        ----------
        params : SimulationParameters
            Parameter configuration for the temporary run.
        kinetics : Kinetics
            Kinetic-model family.
        gas_controller : GasController
            Gas-control strategy.
        operating_mode : OperatingMode
            Operating mode.
        feed_strategy : FeedStrategy
            Feed strategy, when applicable.
        perfusion_strategy : PerfusionStrategy
            Perfusion strategy, when applicable.
        stoichiometry_mode : StoichiometryMode
            Stoichiometric yield update policy.
        performance_mode : bool
            Performance setting passed to the temporary simulation.
        solver_method : str, default="LSODA"
            ODE solver method.
        return_only_summary_stats : bool, default=False
            Whether to return totals dictionaries rather than full result data.
        wall_timeout : float, default=0.0
            ODE wall-clock timeout in seconds; zero disables it.
        debug_failures : bool, default=False
            Whether to print caught evaluation failures.
        **kwargs
            Additional kinetic-model options.

        Returns
        -------
        dict[str, dict] or SimulationData or Literal[0]
            Summary payload, full results, or ``0`` if the temporary run fails.
        """
        try:
            # Create temporary simulation with modified parameters
            temp_sim = cls(
                parameters=params,
                kinetics=kinetics,
                gas_controller=gas_controller,
                operating_mode=operating_mode,
                feed_strategy=feed_strategy,
                perfusion_strategy=perfusion_strategy,
                stoichiometry_mode=stoichiometry_mode,
                performance_mode=performance_mode,
                solver_method=solver_method,
                **kwargs
            )
            
            # Arm wall-clock timeout on all ODE systems (0 = disabled).
            # This aborts integrations that stall due to threshold chattering
            # (e.g. TBS/TBP perfusion strategies near metabolite thresholds).
            if wall_timeout > 0:
                for ode_sys in temp_sim.ode_systems.values():
                    ode_sys.set_wall_timeout(wall_timeout)

            # Run simulation without storing batches.  project_yearly=True caps the
            # run at min(max_culture_age, yearly_uptime) and then scales additive
            # totals to the yearly horizon — giving accurate yearly KPIs at the cost
            # of one max-culture-age window rather than a full-year run.
            sim_data = temp_sim.run_simulation(store_batches=False, project_yearly=True)
            
            # Return based on flag
            if return_only_summary_stats:
                return {
                    'summary_stats': sim_data.summary_stats,
                    'projected_totals_stats': sim_data.projected_totals_stats,
                    'simulated_window_totals_stats': sim_data.simulated_window_totals_stats,
                }
            else:
                return sim_data

        except Exception as e:
            if debug_failures:
                print(
                    "Temporary simulation failed in run_temporary_simulation: "
                    f"{type(e).__name__}: {e}"
                )
            # Return 0 for failed simulations (optimization/sensitivity context)
            return 0

### Summary statistics calculation ###

    def _compute_summary_statistics_from_totals(self, totals: dict) -> dict:
        """Compute process-level totals and KPIs from accumulated boundary flows.

        Parameters
        ----------
        totals : dict
            Accumulated per-batch process-boundary totals for one literal or
            projected time window.

        Returns
        -------
        dict
            Process totals, efficiencies, biomass values, timing diagnostics,
            and space-time yield.
        """
        dry_cell_mass    = self.parameters.cell_properties.dry_cell_mass   # g/cell
        wet_cell_mass    = self.parameters.cell_properties.wet_cell_mass   # g/cell
        protein_fraction = self.parameters.cell_properties.protein_fraction
        cells_per_mol = float(self.parameters.cell_properties.cells_per_mol)

        V_max            = self.parameters.reactor.V_max / 1000            # m³

        total_cells_in = totals['total_cells_in']
        total_cells_out  = totals['total_cells_out']  # Total biomass crossing the boundary (harvested + lost)
        total_cells_net = totals['total_cells_net']  # Net biomass: positive = produced
        total_cells_lost = totals.get('total_cells_lost', 0.0)  # Non-harvested biomass loss (e.g. perfusion separator leak)
        total_cells_harvested = total_cells_out - total_cells_lost  # Productive harvested biomass used for KPIs
        total_cultivation_time = totals['total_cultivation_time']
        total_medium_in       = totals['total_medium_in']
        total_base_in         = totals['total_base_in']
        total_acid_in         = totals.get('total_acid_in', 0.0)
        total_glucose_in      = totals['total_glucose_in']
        total_glutamine_in    = totals['total_glutamine_in']
        total_AA_in           = totals['total_AA_in']
        total_outlet_volume    = totals.get('total_outlet_volume', 0.0)
        total_permeate_volume  = totals.get('total_permeate_volume', 0.0)
        total_glucose_out      = totals.get('total_glucose_out', 0.0)
        total_glutamine_out    = totals.get('total_glutamine_out', 0.0)
        total_AA_out           = totals.get('total_AA_out', 0.0)
        # Net arrays: production positive, consumption negative.
        # Use abs() when converting substrate nets to mass (they are negative).
        total_glucose_net      = totals.get('total_glucose_net', 0.0)
        total_glutamine_net    = totals.get('total_glutamine_net', 0.0)
        total_AA_net           = totals.get('total_AA_net', 0.0)
        total_lactate_in       = totals.get('total_lactate_in', 0.0)
        total_lactate_out      = totals.get('total_lactate_out', 0.0)
        total_ammonium_in      = totals.get('total_ammonium_in', 0.0)
        total_ammonium_out     = totals.get('total_ammonium_out', 0.0)
        total_O2_in            = totals.get('total_O2_in', 0.0)
        total_O2_out           = totals.get('total_O2_out', 0.0)
        total_CT_in           = totals.get('total_CT_in', 0.0)
        total_CT_out          = totals.get('total_CT_out', 0.0)
        total_lactate_net      = totals.get('total_lactate_net', 0.0)
        total_ammonium_net     = totals.get('total_ammonium_net', 0.0)
        total_O2_net           = totals.get('total_O2_net', 0.0)
        total_CT_net          = totals.get('total_CT_net', 0.0)
        total_water_produced   = totals.get('total_water_produced', 0.0)
        total_heat_generated   = totals.get('total_heat_generated', 0.0)
        last_batch_termination_reason = totals.get('last_batch_termination_reason', 'Unknown termination')
        unique_batch_termination_reasons = sorted(totals.get('unique_batch_termination_reasons', set()))
        batch_time_min_h = totals.get('batch_time_min_h', 0.0)
        batch_time_max_h = totals.get('batch_time_max_h', 0.0)
        last_batch_final_volume = totals.get('last_batch_final_volume_L', 0.0)
        last_batch_final_volume_fraction = totals.get('last_batch_final_volume_fraction', 0.0)
        batch_final_volume_min_L = totals.get('batch_final_volume_min_L', 0.0)
        batch_final_volume_max_L = totals.get('batch_final_volume_max_L', 0.0)
        batch_final_volume_min_fraction = totals.get('batch_final_volume_min_fraction', 0.0)
        batch_final_volume_max_fraction = totals.get('batch_final_volume_max_fraction', 0.0)
        phase_active_runtime_h = dict(totals.get('phase_active_runtime_h', {}))
        batch_phase_active_runtime_h = list(totals.get('batch_phase_active_runtime_h', []))

        if batch_time_min_h == float('inf'):
            batch_time_min_h = 0.0
        if batch_final_volume_min_L == float('inf'):
            batch_final_volume_min_L = 0.0
        if batch_final_volume_min_fraction == float('inf'):
            batch_final_volume_min_fraction = 0.0

        batch_time_range_h = batch_time_max_h - batch_time_min_h

        # Total dry cell mass (kg)
        total_DCM_in_kg = total_cells_in * dry_cell_mass / 1000
        total_DCM_in_cmmol = total_cells_in / cells_per_mol * 1000

        # "out" is the full boundary biomass output (harvested + lost) and feeds the
        # elemental/mass closures; "harvested" is the productive fraction used for KPIs.
        total_DCM_out_kg = total_cells_out * dry_cell_mass / 1000
        total_DCM_out_cmmol = total_cells_out / cells_per_mol * 1000

        total_DCM_harvested_kg = total_cells_harvested * dry_cell_mass / 1000
        total_DCM_harvested_cmmol = total_cells_harvested / cells_per_mol * 1000
        total_WCM_harvested_kg = total_cells_harvested * wet_cell_mass / 1000

        total_DCM_lost_kg = total_cells_lost * dry_cell_mass / 1000
        total_DCM_lost_cmmol = total_cells_lost / cells_per_mol * 1000

        total_DCM_produced_kg = total_cells_net * dry_cell_mass / 1000
        total_DCM_produced_cmmol = total_cells_net / cells_per_mol * 1000

        # Space-Time Yield (kg m⁻³ day⁻¹), using harvested biomass and the
        # active modeled cultivation duration. Yearly uptime already excludes
        # offline/turnaround days.
        if total_cultivation_time > 0.0:
            STY = total_DCM_harvested_kg / V_max / (total_cultivation_time / 24)
            STY_WCM = total_WCM_harvested_kg / V_max / (total_cultivation_time / 24)
        else:
            # A pre-integration state rejection has no cultivation duration.
            # Undefined productivity is represented as NaN so optimization's
            # termination-reason guard, rather than a division error, rejects it.
            STY = np.nan
            STY_WCM = np.nan

        # Process-boundary medium use: initial charge plus feed medium, excluding
        # pH-control acid/base additions. This input-per-product ratio is a cost,
        # so lower values are preferable.
        total_medium_in_m3 = total_medium_in / 1000

        # Nutrient mass conversions. Convert mmol to mol.
        # Substrate nets are negative (consumed); use abs() for mass.
        total_glutamine_g = total_glutamine_in / 1000 * self.MW_glutamine
        total_glucose_g   = total_glucose_in   / 1000 * self.MW_glucose
        total_AA_g        = total_AA_in        / 1000 * self.MW_AAs_ex_glutamine
        total_glutamine_out_g = total_glutamine_out / 1000 * self.MW_glutamine
        total_glucose_out_g   = total_glucose_out   / 1000 * self.MW_glucose
        total_AA_out_g        = total_AA_out        / 1000 * self.MW_AAs_ex_glutamine
        total_glutamine_consumed_g = abs(total_glutamine_net) / 1000 * self.MW_glutamine
        total_glucose_consumed_g   = abs(total_glucose_net)   / 1000 * self.MW_glucose
        total_AA_consumed_g        = abs(total_AA_net)        / 1000 * self.MW_AAs_ex_glutamine
        total_lactate_produced_g   = total_lactate_net   / 1000 * self.MW_lactate
        total_ammonium_produced_g  = total_ammonium_net  / 1000 * self.MW_ammonium
        total_CT_produced_g       = total_CT_net       / 1000 * self.MW_CO2
        total_O2_consumed_g        = abs(total_O2_net)        / 1000 * self.MW_O2
        total_water_produced_g     = total_water_produced     / 1000 * self.MW_water

        # Process-boundary nutrient input masses (kg). These deliberately use
        # inlet quantities, not net metabolic consumption.
        proteinaceous_input_kg = np.sum([total_glutamine_g, total_AA_g]) / 1000
        total_feed_nutrient_input_kg = total_glucose_g / 1000 + proteinaceous_input_kg
        harvested_protein_kg = total_DCM_harvested_kg * protein_fraction

        def input_per_product(input_amount: float, product_amount: float) -> float:
            """Return an auditable input/product ratio without divide warnings.

            A finite positive input with no product is an infinite conversion
            cost. A zero-input/zero-product case is undefined and represented
            as ``nan`` so optimization can reject it explicitly.
            """
            if np.isclose(product_amount, 0.0):
                return np.inf if not np.isclose(input_amount, 0.0) else np.nan
            return input_amount / product_amount

        # Input-per-product KPIs. All three are lower-is-better metrics.
        medium_conversion_efficiency_m3_kg = input_per_product(
            total_medium_in_m3,
            total_DCM_harvested_kg,
        )
        medium_conversion_efficiency_m3_kg_WCM = input_per_product(
            total_medium_in_m3,
            total_WCM_harvested_kg,
        )
        protein_conversion_efficiency_kg_kg = input_per_product(
            proteinaceous_input_kg,
            harvested_protein_kg,
        )
        feed_conversion_efficiency_kg_kg = input_per_product(
            total_feed_nutrient_input_kg,
            total_DCM_harvested_kg,
        )
        feed_conversion_efficiency_kg_kg_WCM = input_per_product(
            total_feed_nutrient_input_kg,
            total_WCM_harvested_kg,
        )
        total_nutrient_out_kg = (total_glucose_out_g + total_glutamine_out_g + total_AA_out_g) / 1000
        total_nutrient_consumed_kg = (total_glucose_consumed_g + total_glutamine_consumed_g + total_AA_consumed_g) / 1000

        return {
            'total_DCM_in_kg':               total_DCM_in_kg,
            'total_DCM_in_cmmol':             total_DCM_in_cmmol,
            'total_DCM_out_kg':              total_DCM_out_kg,
            'total_DCM_out_cmmol':            total_DCM_out_cmmol,
            'total_DCM_harvested_kg':        total_DCM_harvested_kg,
            'total_DCM_harvested_cmmol':      total_DCM_harvested_cmmol,
            'total_WCM_harvested_kg':        total_WCM_harvested_kg,
            'total_DCM_lost_kg':             total_DCM_lost_kg,
            'total_DCM_lost_cmmol':           total_DCM_lost_cmmol,
            'total_DCM_produced_kg':          total_DCM_produced_kg,
            'total_DCM_produced_cmmol':        total_DCM_produced_cmmol,
            'STY_kg_m3_day':                 STY,
            'STY_WCM_kg_m3_day':             STY_WCM,
            'medium_conversion_efficiency_m3_kg': medium_conversion_efficiency_m3_kg,
            'medium_conversion_efficiency_m3_kg_WCM': medium_conversion_efficiency_m3_kg_WCM,
            'protein_conversion_efficiency_kg_kg': protein_conversion_efficiency_kg_kg,
            'feed_conversion_efficiency_kg_kg':    feed_conversion_efficiency_kg_kg,
            'feed_conversion_efficiency_kg_kg_WCM': feed_conversion_efficiency_kg_kg_WCM,
            'total_medium_in_L':            total_medium_in,
            'total_base_in_L':              total_base_in,
            'total_acid_in_L':              total_acid_in,
            'total_outlet_volume_L':         total_outlet_volume,
            'total_permeate_volume_L':       total_permeate_volume,
            'total_glucose_in_mmol':         total_glucose_in,
            'total_glucose_out_mmol':        total_glucose_out,
            'total_glucose_net_mmol':        total_glucose_net,
            'total_glutamine_in_mmol':       total_glutamine_in,
            'total_glutamine_out_mmol':      total_glutamine_out,
            'total_glutamine_net_mmol':      total_glutamine_net,
            'total_AA_in_mmol':              total_AA_in,
            'total_AA_out_mmol':             total_AA_out,
            'total_AA_net_mmol':             total_AA_net,
            'total_lactate_in_mmol':         total_lactate_in,
            'total_lactate_out_mmol':        total_lactate_out,
            'total_lactate_net_mmol':        total_lactate_net,
            'total_ammonium_in_mmol':        total_ammonium_in,
            'total_ammonium_out_mmol':       total_ammonium_out,
            'total_ammonium_net_mmol':       total_ammonium_net,
            'total_O2_in_mmol':              total_O2_in,
            'total_O2_out_mmol':             total_O2_out,
            'total_O2_net_mmol':             total_O2_net,
            'total_CT_in_mmol':             total_CT_in,
            'total_CT_out_mmol':            total_CT_out,
            'total_CT_net_mmol':            total_CT_net,
            'total_water_produced_mmol':     total_water_produced,
            'total_nutrient_in_kg':          total_feed_nutrient_input_kg,
            'total_nutrient_out_kg':         total_nutrient_out_kg,
            'total_nutrient_consumed_kg':    total_nutrient_consumed_kg,
            'total_lactate_net_kg':         total_lactate_produced_g / 1000,
            'total_ammonium_net_kg':        total_ammonium_produced_g / 1000,
            'total_CT_net_kg':             total_CT_produced_g / 1000,
            'total_O2_net_kg':              total_O2_consumed_g / 1000,
            'total_water_produced_kg':       total_water_produced_g / 1000,
            'total_heat_generated':          total_heat_generated,
            'total_simulation_time_h':       total_cultivation_time,
            'number_of_batches':             totals['number_of_batches'],
            'phase_active_runtime_h':        phase_active_runtime_h,
            'batch_phase_active_runtime_h':  batch_phase_active_runtime_h,
            'last_batch_termination_reason': last_batch_termination_reason,
            'unique_batch_termination_reasons': unique_batch_termination_reasons,
            'batch_time_min_h':              batch_time_min_h,
            'batch_time_max_h':              batch_time_max_h,
            'batch_time_range_h':            batch_time_range_h,
            'last_batch_final_volume_L':     last_batch_final_volume,
            'last_batch_final_volume_fraction': last_batch_final_volume_fraction,
            'batch_final_volume_min_L':      batch_final_volume_min_L,
            'batch_final_volume_max_L':      batch_final_volume_max_L,
            'batch_final_volume_min_fraction': batch_final_volume_min_fraction,
            'batch_final_volume_max_fraction': batch_final_volume_max_fraction,
        }

    @staticmethod
    def _extract_total_stats(stats: dict) -> dict:
        """Return process totals/diagnostics from a combined stats dictionary."""
        totals = {}
        for key, value in stats.items():
            if key.startswith('total_') or key in _TOTAL_DIAGNOSTIC_KEYS:
                totals[key] = value
        return totals

    def _compute_summary_statistics(self, stats: dict) -> dict:
        """Select KPI fields from a combined totals-and-statistics mapping.

        Parameters
        ----------
        stats : dict
            Combined process statistics.

        Returns
        -------
        dict
            Derived KPI fields used by ``SimulationData.summary_stats``.
        """
        return {k: stats[k] for k in _DERIVED_SUMMARY_KEYS if k in stats}

### Export and persistence methods ###

    def save(self,
             filename: str,
             add_timestamp: bool = True,
             file_format: str = 'excel',
             include_metadata: bool = True,
             include_summary_stats: bool = True,
             include_cultivation_summary: bool = True,
             include_batch_data: bool = True,
             include_parameters: bool = True,
             include_pickle: bool = True,
             include_opt_parameters: bool = False,
             overwrite: bool = False,
             export_dir: Optional[Path] = None) -> Path:
        """Export result data and optionally serialize the full simulation.

        Delegates to SimulationData.save() for all data export. When
        ``include_pickle=True`` (the default), the pickle written is of the
        *full* CellCultureSimulation object (models + data) rather than just
        the SimulationData, so the saved file can be directly reloaded for
        optimization or sensitivity analysis.  The pickle is placed in the
        same export sub-directory as the Excel/CSV output.

        Parameters
        ----------
        filename : str
            Output filename stem.
        add_timestamp : bool, default=True
            Whether to append a timestamp to output names.
        file_format : {"excel", "csv"}, default="excel"
            Data-export format passed to :meth:`SimulationData.save`.
        include_metadata, include_summary_stats, include_cultivation_summary,
        include_batch_data, include_parameters : bool
            Data sections to include in the delegated export.
        include_pickle : bool, default=True
            Whether to write this complete :class:`CellCultureSimulation`, not
            merely its :class:`SimulationData`.
        include_opt_parameters : bool, default=False
            Whether to export optimized parameter values. Requires a simulation
            produced by ``OptimizationManager``.
        overwrite : bool, default=False
            Whether to replace existing files.
        export_dir : pathlib.Path, optional
            Parent export directory.

        Returns
        -------
        pathlib.Path
            Workbook path for Excel or export directory for CSV.

        Raises
        ------
        ValueError
            If optimized-parameter export is requested for a non-optimized
            simulation.
        """
        artifact_stem = (
            f"opt_{self.data.timestamp.strftime('%Y%m%d_%H%M')}"
            if add_timestamp else "opt"
        )

        # Delegate data export, suppressing the data-only pickle
        filepath = self.data.save(
            filename,
            add_timestamp=add_timestamp,
            file_format=file_format,
            include_metadata=include_metadata,
            include_summary_stats=include_summary_stats,
            include_cultivation_summary=include_cultivation_summary,
            include_batch_data=include_batch_data,
            include_parameters=include_parameters,
            include_pickle=False,
            overwrite=overwrite,
            export_dir=export_dir,
            file_stem=artifact_stem,
        )

        # Derive the export sub-directory from the returned path.
        # Excel → filepath is a file inside export_dir; CSV → filepath IS export_dir.
        if filepath.is_dir():
            export_dir = filepath
        else:
            export_dir = filepath.parent

        if include_pickle:
            pickle_path = resolve_pickle_save_path(artifact_stem, export_dir)
            save_pickle_object(self, pickle_path, overwrite=overwrite)
            print(f"CellCultureSimulation pickle saved to: {pickle_path}")

        if include_opt_parameters:
            if not self.optimized or self.optimization_manager is None:
                raise ValueError(
                    "include_opt_parameters=True was requested but this simulation was not optimized "
                    "and produced by OptimizationManager (sim.optimized is False). "
                    "Only call this on a simulation returned by OptimizationManager."
                )
            # Ensure the timestamped export directory still exists before
            # writing the final companion file. Cloud-sync clients on Windows
            # can transiently replace this directory after the workbook and
            # pickle have been written above.
            export_dir.mkdir(parents=True, exist_ok=True)
            self.optimization_manager.save_opt_parameters(
                export_dir / f"{artifact_stem}_opt_params.yaml", overwrite=overwrite
            )

        return filepath

    def save_pickle(self, filename: str, overwrite: bool = False,
                    base_export_dir: Optional[Path] = None) -> Path:
        """Serialize the full simulation object for later reuse.

        Unlike SimulationData.save_pickle(), this serialises the entire simulation
        object — configured kinetic model, ODE systems, gas/pH controllers, feed and
        perfusion strategies, *and* the accumulated SimulationData — so the loaded
        object can be passed directly to OptimizationManager or SensitivityAnalyzer
        without any re-initialisation.

        Parameters
        ----------
        filename : str
            Output name or path, with optional ``.pkl`` suffix.
        overwrite : bool, default=False
            Whether to replace an existing pickle.
        base_export_dir : pathlib.Path, optional
            Parent directory for relative names. Defaults to :data:`EXPORT_DIR`.

        Returns
        -------
        pathlib.Path
            Written pickle path.
        """
        export_base = base_export_dir if base_export_dir is not None else EXPORT_DIR
        filepath = resolve_pickle_save_path(filename, export_base)
        filepath = save_pickle_object(self, filepath, overwrite=overwrite)

        print(f"CellCultureSimulation saved to: {filepath}")
        return filepath

    @classmethod
    def load_pickle(cls, filename: Union[str, Path],
                    base_export_dir: Optional[Path] = None) -> 'CellCultureSimulation':
        """Load a previously serialized :class:`CellCultureSimulation`.

        Searches for the file in *base_export_dir* (or the package default
        ``EXPORT_DIR``) and all of its sub-directories when only a bare filename
        is supplied.

        Parameters
        ----------
        filename : str or pathlib.Path
            Pickle name or path, with optional ``.pkl`` suffix.
        base_export_dir : pathlib.Path, optional
            Root for relative-name lookup. Defaults to :data:`EXPORT_DIR`.

        Returns
        -------
        CellCultureSimulation
            Fully restored simulation instance.

        Raises
        ------
        FileNotFoundError
            If no matching pickle can be located.
        TypeError
            If the pickle does not contain :class:`CellCultureSimulation`.
        """
        search_root = base_export_dir if base_export_dir is not None else EXPORT_DIR
        filepath = resolve_pickle_load_path(filename, search_root)
        sim = load_pickle_object(filepath, expected_type=cls)
        
        print(f"CellCultureSimulation loaded from: {filepath}")
        return sim
