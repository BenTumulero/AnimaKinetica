"""Operating-mode ODE systems for animal-cell-culture bioreactors.

The concrete systems ``BatchODESystem``, ``FedBatchODESystem``,
``ChemostatODESystem``, and ``PerfusionODESystem`` integrate viable cells,
medium components, volume, gas-related states, and pH-related states.
``ODESystem`` provides shared kinetic-rate calculations, gas and pH control,
optional non-carbonate-buffer handling, and time-resolved logging.

Each right-hand-side evaluation updates the coupled models before calculating
state derivatives. ``ODESystem.finalize_logging`` interpolates controller and
kinetic signals onto the accepted solver time grid for storage in ``BatchData``.

Use ``CellCultureSimulation`` for normal simulations. It calls
``create_ode_systems`` for the selected ``OperatingMode``.
"""

# odesystems.py
from abc import ABC, abstractmethod
from enum import Enum
from time import perf_counter
from typing import List, Optional
import numpy as np
from animakinetica.models.kinetics import KineticModel
from animakinetica.models.events import *
from animakinetica.core.parameters import SimulationParameters
from animakinetica.models.stoichiometry import StoichiometricModel
from animakinetica.models.feed_strategies import FeedStrategyModel, GlucoseSetpointFeedStrategy
from animakinetica.models.perfusion_strategies import PerfusionStrategyModel, GlucoseSetpointPerfusionStrategy
from animakinetica.models.gas_controllers import GasControllerModel
from animakinetica.models.pH_controller import pHControllerModel

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = [
    "OperatingMode",
    "ODESystem",
    "BatchODESystem",
    "FedBatchODESystem",
    "ChemostatODESystem",
    "PerfusionODESystem",
    "create_ode_systems",
]

# Enum for model selection
class OperatingMode(Enum):
    """Supported process configurations.

    ``BATCH_WITH_STARTUP_FEDBATCH`` explicitly runs an initial batch phase
    before fed-batch operation. For the other multi-phase modes, a batch phase
    is included only when ``fed_batch.C_Glc_batch_trigger`` is configured.
    """
    BATCH = "batch"
    BATCH_WITH_STARTUP_FEDBATCH = "batch-with-startup-fed-batch"
    FEDBATCH = "fed-batch"
    CHEMOSTAT = "chemostat"
    PERFUSION = "perfusion"
    CONCENTRATED_FEDBATCH = "concentrated-fed-batch"
    CONTINUOUS_PERFUSION = "continuous-perfusion"

    def __str__(self):
        """Return the human-readable operating-mode name."""
        mode_names = {
            self.BATCH: "Batch",
            self.BATCH_WITH_STARTUP_FEDBATCH: "Batch (startup Fedbatch)",
            self.FEDBATCH: "Fedbatch",
            self.CHEMOSTAT: "Chemostat",
            self.PERFUSION: "Perfusion",
            self.CONCENTRATED_FEDBATCH: "Concentrated Fedbatch", 
            self.CONTINUOUS_PERFUSION: "Continuous Perfusion"           
        }
        return mode_names.get(self, self.value) 
    
    @property
    def abbr(self):
        """Return the abbreviation used in configuration and result labels."""
        mode_abbr = {
            self.BATCH: "B",
            self.BATCH_WITH_STARTUP_FEDBATCH: "BSF",
            self.FEDBATCH: "F",
            self.CHEMOSTAT: "C",
            self.PERFUSION: "P",
            self.CONCENTRATED_FEDBATCH: "CF",
            self.CONTINUOUS_PERFUSION: "CP"           
        }
        return mode_abbr.get(self, self.value)

# Abstract base class for ODE systems
class ODESystem(ABC):
    """Abstract base class for operating-mode ODE systems.

    Subclasses cache their mode-specific medium and hydraulic parameters and
    implement ``rhs``. This class updates shared kinetic, gas-transfer,
    and pH-control models; manages optional buffer states; and records signals
    needed for ``BatchData``.
    """

    def __init__(self, params: SimulationParameters, stoichiometric_model: StoichiometricModel, kinetic_model: KineticModel, gas_control_model: GasControllerModel):
        """Initialize shared models, cached parameters, and logging storage.

        Parameters
        ----------
        params : SimulationParameters
            Validated simulation parameters.
        stoichiometric_model : StoichiometricModel
            Model supplying current yield coefficients.
        kinetic_model : KineticModel
            Model supplying growth and cell-specific rates.
        gas_control_model : GasControllerModel
            Model supplying dissolved-oxygen control and OTR.
        """
        self._cache_parameters(params)
        self.stoichiometric_model = stoichiometric_model
        self.kinetic_model = kinetic_model
        self.gas_controller = gas_control_model
        self.pH_controller = pHControllerModel(params, self.gas_controller)
        # Osmolality background state starts from initial conditions and is
        # transported dynamically in the ODE system.
        self.osm_base = float(params.initial_conditions.osm_bg_init)
        
        # When True, _log_intermediates records all 30+ auxiliary lists (kinetics,
        # gas control, pH, etc.) needed for detailed batch visualisation. When
        # False, only the minimal signals required for summary statistics are
        # recorded: time, process dilution rate, and base-addition dilution rate.
        # Set by run_simulation() based on the store_batches flag.
        self.log_details: bool = True

        # Wall-clock timeout for ODE integration (used during optimization to
        # abort parameter combinations that cause the solver to stall, e.g. due
        # to threshold chattering in TBS/TBP strategies).  0 = disabled.
        # Call set_wall_timeout() just before integration starts.
        self._wall_start: float = 0.0
        self._wall_timeout: float = 0.0
        
        # Dynamic lists to store ALL intermediate evaluations during ODE solving
        # These capture every rhs() call at solve_ivp's adaptive timepoints
        self._t_log: List[float] = []          # All evaluation times
        self._D_log: List[float] = []          # All D values
        self._N_log: List[float] = []          # All N values
        self._Fg_log: List[float] = []         # All Fg values
        self._P_over_V_log: List[float] = []   # All power-per-volume values
        self._v_s_log: List[float] = []        # All superficial gas velocity values
        self._y_O2_log: List[float] = []       # All y_O2 values
        self._kLa_log: List[float] = []        # All kLa values
        self._OTR_log: List[float] = []        # All OTR values
        self._CTR_log: List[float] = []        # All CTR values
        self._C_CO2_log: List[float] = []      # All C_CO2 values
        self._D_base_log: List[float] = []     # All D_base values
        self._D_acid_log: List[float] = []     # All D_acid values
        self._y_CO2_log: List[float] = []      # All y_CO2 values
        self._osm_log: List[float] = []        # All osmolality values
        self._osm_background_log: List[float] = []  # All osmolality background-state values

        # Kinetic model variable logs
        self._mu_log: List[float] = []         # All growth rate values
        self._q_Glc_log: List[float] = []      # All glucose uptake rate values
        self._q_Gln_log: List[float] = []      # All glutamine uptake rate values
        self._q_Lac_log: List[float] = []      # All lactate production rate values
        self._q_Amm_log: List[float] = []      # All ammonia production rate values
        self._q_O2_log: List[float] = []       # All O2 uptake rate values
        self._q_CO2_log: List[float] = []      # All CO2 production rate values
        self._q_AA_log: List[float] = []       # All amino acid uptake rate values
        self._q_H2O_log: List[float] = []      # All metabolic water production rate values
        self._heat_rate_log: List[float] = []  # All volumetric heat-generation rate values
        self._lag_term_log: List[float] = []   # All lag term values
        self._NF_glucose_log: List[float] = [] # All glucose nutrient factor values
        self._NF_glutamine_log: List[float] = [] # All glutamine nutrient factor values
        self._NF_AA_log: List[float] = []      # All amino-acid nutrient factor values
        self._NF_O2_log: List[float] = []      # All O2 nutrient factor values
        self._NF_log: List[float] = []         # All combined nutrient factor values
        self._IF_O2_log: List[float] = []      # All O2 inhibition factor values
        self._IF_CO2_log: List[float] = []     # All CO2 inhibition factor values
        self._IF_lactate_log: List[float] = [] # All lactate inhibition factor values
        self._IF_ammonia_log: List[float] = [] # All ammonia inhibition factor values
        self._IF_osmolality_log: List[float] = [] # All osmolality inhibition factor values
        self._IF_log: List[float] = []         # All combined inhibition factor values
        # Stoichiometric yield coefficient logs
        self._Y_Glc_X_log: List[float] = []
        self._Y_Lac_Glc_log: List[float] = []
        self._Y_Gln_Glc_log: List[float] = []
        self._Y_O2_Glc_log: List[float] = []
        self._Y_CO2_Glc_log: List[float] = []
        self._Y_AA_Glc_log: List[float] = []
        self._Y_Amm_Gln_log: List[float] = []
        
        # Final interpolated arrays (populated by finalize_logging after integration)
        self.D_list: np.ndarray = np.array([])        # Dilution rate (1/h)
        self.N_list: np.ndarray = np.array([])        # Agitation speed (rpm)
        self.Fg_list: np.ndarray = np.array([])       # Gas flow rate (vvh; L/L/h)
        self.P_over_V_list: np.ndarray = np.array([]) # Power input per volume (W/m^3)
        self.v_s_list: np.ndarray = np.array([])      # Superficial gas velocity (m/s)
        self.y_O2_list: np.ndarray = np.array([])     # O2 mole fraction in gas
        self.kLa_list: np.ndarray = np.array([])      # Volumetric mass transfer coefficient (1/h)
        self.OTR_list: np.ndarray = np.array([])      # Oxygen transfer rate (mmol/L/h)
        self.CTR_list: np.ndarray = np.array([])      # CO2 transfer rate (mmol/L/h)
        self.C_CO2_list: np.ndarray = np.array([])    # CO2 concentration (mmol/L)
        self.D_base_list: np.ndarray = np.array([])   # Base dilution rate (1/h)
        self.y_CO2_list: np.ndarray = np.array([])    # CO2 mole fraction in gas
        self.osm_list: np.ndarray = np.array([])      # Osmolality (mOsm)
        self.osm_background_list: np.ndarray = np.array([])  # Osmolality background state (mOsm)
        
        # Kinetic model final arrays
        self.mu_list: np.ndarray = np.array([])       # Growth rate (1/h)
        self.q_Glc_list: np.ndarray = np.array([])    # Glucose uptake rate (mmol/cell/h)
        self.q_Gln_list: np.ndarray = np.array([])    # Glutamine uptake rate (mmol/cell/h)
        self.q_Lac_list: np.ndarray = np.array([])    # Lactate production rate (mmol/cell/h)
        self.q_Amm_list: np.ndarray = np.array([])    # Ammonia production rate (mmol/cell/h)
        self.q_AA_list: np.ndarray = np.array([])     # Amino acid uptake rate (mmol/cell/h)
        self.q_O2_list: np.ndarray = np.array([])     # O2 uptake rate (mmol/cell/h)
        self.q_CO2_list: np.ndarray = np.array([])    # CO2 production rate (mmol/cell/h)
        self.q_H2O_list: np.ndarray = np.array([])    # Metabolic water production rate (mmol/cell/h)
        self.heat_rate_list: np.ndarray = np.array([]) # Volumetric heat-generation rate (kJ/L/h)
        self.lag_term_list: np.ndarray = np.array([]) # Lag term
        self.NF_glucose_list: np.ndarray = np.array([]) # Glucose nutrient factor
        self.NF_glutamine_list: np.ndarray = np.array([]) # Glutamine nutrient factor
        self.NF_AA_list: np.ndarray = np.array([])    # Amino-acid nutrient factor
        self.NF_O2_list: np.ndarray = np.array([])    # O2 nutrient factor
        self.NF_list: np.ndarray = np.array([])       # Combined nutrient factor
        self.IF_O2_list: np.ndarray = np.array([])    # O2 inhibition factor
        self.IF_CO2_list: np.ndarray = np.array([])   # CO2 inhibition factor
        self.IF_lactate_list: np.ndarray = np.array([]) # Lactate inhibition factor
        self.IF_ammonia_list: np.ndarray = np.array([]) # Ammonia inhibition factor
        self.IF_osmolality_list: np.ndarray = np.array([]) # Osmolality inhibition factor
        self.IF_list: np.ndarray = np.array([])       # Combined inhibition factor
        # Stoichiometric yield coefficient arrays
        self.Y_Glc_X_list: np.ndarray = np.array([])
        self.Y_Lac_Glc_list: np.ndarray = np.array([])
        self.Y_Gln_Glc_list: np.ndarray = np.array([])
        self.Y_O2_Glc_list: np.ndarray = np.array([])
        self.Y_CO2_Glc_list: np.ndarray = np.array([])
        self.Y_AA_Glc_list: np.ndarray = np.array([])
        self.Y_Amm_Gln_list: np.ndarray = np.array([])

        # --- Non-carbonate buffer logging ---
        # One log list per buffer (concentration), plus H+ flux terms.
        self._C_buffer_log: List[List[float]] = []  # list of per-buffer concentration logs
        self._dC_H_dt_from_buffer_log: List[float] = []
        self._dC_H_dt_from_Amm_log: List[float] = []
        self.C_buffer_list: np.ndarray = np.array([])  # shape: n_buffers × n_points
        self.dC_H_dt_from_buffer_list: np.ndarray = np.array([])
        self.dC_H_dt_from_Amm_list: np.ndarray = np.array([])
        self.n_buffers: int = self.pH_controller.n_buffers
        for _i in range(self.n_buffers):
            self._C_buffer_log.append([])
        # Cache of current buffer concentrations (set in _update_shared_models)
        self._current_C_buffer: List[float] = []

    @staticmethod
    def _clamp_y(y: list) -> list:
        """Ensure state variables do not go negative during integration.

        Volume (index 5) is floored at 1e-6 L to prevent division by zero when
        computing dilution rates.

        Parameters
        ----------
        y : list
            State vector to clamp.

        Returns
        -------
        list
            New state vector with non-negative values and a positive volume.
        """
        clamped = [max(0.0, yi) for yi in y]
        clamped[5] = max(1e-6, clamped[5])  # Volume floor
        return clamped

    def set_wall_timeout(self, timeout: float) -> None:
        """Set the maximum wall-clock duration for subsequent RHS evaluations.

        Call immediately before integration. ``_log_intermediates`` raises
        ``RuntimeError`` when the timeout elapses; ``0`` disables the guard.

        Parameters
        ----------
        timeout : float
            Maximum permitted wall-clock time in seconds.
        """
        self._wall_timeout = timeout
        self._wall_start = perf_counter()

    def _pop_last_log_entries(self, n: int = 1) -> None:
        """Remove the ``n`` most recently appended intermediate-log entries.

        Used by ``_log_intermediates`` to undo phantom log entries created
        by a solve_ivp trial step that was later rejected by the adaptive
        step-size controller (see docstring there). All log lists are always
        kept in lock-step (one entry per call to ``_log_intermediates`` that
        passed the forward-progress check), so popping the same count from
        every list preserves index alignment.

        Parameters
        ----------
        n : int, default=1
            Number of latest entries to remove.
        """
        for _ in range(n):
            if not self._t_log:
                return
            self._t_log.pop()
            self._D_log.pop()
            self._D_base_log.pop()
            self._D_acid_log.pop()
            self._q_H2O_log.pop()
            self._heat_rate_log.pop()
            self._OTR_log.pop()
            self._CTR_log.pop()

            if not self.log_details:
                continue

            self._C_CO2_log.pop()
            self._y_CO2_log.pop()
            self._osm_log.pop()
            self._osm_background_log.pop()
            self._N_log.pop()
            self._Fg_log.pop()
            self._P_over_V_log.pop()
            self._v_s_log.pop()
            self._y_O2_log.pop()
            self._kLa_log.pop()
            self._mu_log.pop()
            self._q_Glc_log.pop()
            self._q_Gln_log.pop()
            self._q_Lac_log.pop()
            self._q_Amm_log.pop()
            self._q_AA_log.pop()
            self._q_O2_log.pop()
            self._q_CO2_log.pop()
            self._lag_term_log.pop()
            self._NF_glucose_log.pop()
            self._NF_glutamine_log.pop()
            self._NF_AA_log.pop()
            self._NF_O2_log.pop()
            self._NF_log.pop()
            self._IF_O2_log.pop()
            self._IF_CO2_log.pop()
            self._IF_lactate_log.pop()
            self._IF_ammonia_log.pop()
            self._IF_osmolality_log.pop()
            self._IF_log.pop()
            self._Y_Glc_X_log.pop()
            self._Y_Lac_Glc_log.pop()
            self._Y_Gln_Glc_log.pop()
            self._Y_O2_Glc_log.pop()
            self._Y_CO2_Glc_log.pop()
            self._Y_AA_Glc_log.pop()
            self._Y_Amm_Gln_log.pop()
            for i in range(self.n_buffers):
                if self._C_buffer_log[i]:
                    self._C_buffer_log[i].pop()
            self._dC_H_dt_from_buffer_log.pop()
            self._dC_H_dt_from_Amm_log.pop()

    def _log_intermediates(self, t: float, D: float):
        """Log intermediate values at ODE evaluation points.
        
        Always logs t, D, D_base, D_acid, q_H2O, heat_rate, OTR, and CTR (required
        for summary-statistics and mass-balance/inlet-flow calculations). The
        remaining auxiliary lists are only populated when ``self.log_details`` is
        True (i.e. when the caller intends to store detailed batch data for
        visualisation / export).

        Call ``finalize_logging`` after ``scipy.integrate.solve_ivp`` completes
        to interpolate these values onto its accepted solution time points.

        Parameters
        ----------
        t : float
            Current simulation time (h).
        D : float
            Medium dilution rate excluding base and acid addition (1/h).
        """
        # Wall-clock timeout guard: fires on every rhs call, not just forward steps.
        # LSODA calls rhs many times for error estimation / Jacobian approximation at
        # the same or earlier t, so the check must sit outside the t-advance guard.
        if self._wall_timeout > 0 and (perf_counter() - self._wall_start) > self._wall_timeout:
            raise RuntimeError(
                f"ODE integration aborted: wall-clock timeout of {self._wall_timeout:.1f}s exceeded at t={t:.2f}h"
            )

        # Adaptive-step solvers (BDF/LSODA) call rhs() at a trial t that can
        # later be REJECTED by local error control and retried at a smaller t.
        # The old strict "t > last logged t" guard logged the (phantom) trial
        # point, then silently dropped the retry -- and every subsequent
        # genuinely-accepted step -- until real time advanced back past that
        # phantom timestamp. In stiff regions with many rejections this
        # starved the log relative to the accepted solution grid (e.g. 29
        # logged points vs. 151 accepted sol.t points for one fed-batch
        # phase), which finalize_logging() then interpolates across, smearing
        # q_Glc_list/q_Lac_list/etc. and producing 10-28% mass-balance errors
        # in the downstream trapezoidal rate integrals despite the raw ODE
        # state itself being perfectly self-consistent.
        #
        # Fix: prune any previously logged samples that are no longer strictly
        # behind the current call's t before appending. This self-corrects
        # after a rejected-step retry (the phantom trial entry gets popped
        # once time backtracks below it) and also keeps only the most recent
        # evaluation for repeated same-t calls (e.g. Newton/Jacobian
        # evaluations during an implicit step).
        while self._t_log and self._t_log[-1] >= t:
            self._pop_last_log_entries()

        self._t_log.append(t)
        self._D_log.append(D)
        self._D_base_log.append(self.pH_controller.D_base)
        self._D_acid_log.append(self.pH_controller.D_acid)
        self._q_H2O_log.append(self.kinetic_model.q_H2O_current)
        self._heat_rate_log.append(self.kinetic_model.heat_rate_current)
        # OTR/CTR are required (like D/D_base/D_acid above) for mass-balance and
        # summary-statistics correctness, regardless of detailed logging. If these
        # were gated behind log_details, run_temporary_simulation() (used by the
        # optimizer and sensitivity analysis, which always sets store_batches=False)
        # would silently zero out gas-transfer O2/CO2 boundary terms, causing
        # spurious mass-balance closure failures that don't reproduce when the same
        # parameters are run with store_batches=True.
        self._OTR_log.append(self.gas_controller.OTR_current)
        self._CTR_log.append(self.pH_controller.CTR_current)

        # Skip detailed logging when not needed (e.g. during optimization)
        if not self.log_details:
            return

        self._C_CO2_log.append(self.pH_controller.C_H2CO3)
        self._y_CO2_log.append(self.pH_controller.y_CO2_current)
        self._osm_log.append(self.kinetic_model.osm_current)
        self._osm_background_log.append(self.C_Osm_bg_current)

        # Extract gas controller state (updated during calculate_gas_transfer_rates)
        self._N_log.append(self.gas_controller.N_current)
        self._Fg_log.append(self.gas_controller.Fg_current)
        self._P_over_V_log.append(self.gas_controller.P_over_V_current)
        self._v_s_log.append(self.gas_controller.v_s_current)
        self._y_O2_log.append(self.gas_controller.y_O2_current)
        self._kLa_log.append(self.gas_controller.klO2a_current)

        # Extract kinetic model state variables
        self._mu_log.append(self.kinetic_model.mu_current)
        self._q_Glc_log.append(self.kinetic_model.q_Glc_current)
        self._q_Gln_log.append(self.kinetic_model.q_Gln_current)
        self._q_Lac_log.append(self.kinetic_model.q_Lac_current)
        self._q_Amm_log.append(self.kinetic_model.q_Amm_current)
        self._q_AA_log.append(self.kinetic_model.q_AA_current)
        self._q_O2_log.append(self.kinetic_model.q_O2_current)
        self._q_CO2_log.append(self.kinetic_model.q_CO2_current)
        self._lag_term_log.append(self.kinetic_model.lag_term_current)
        self._NF_glucose_log.append(self.kinetic_model.NF_glucose_current)
        self._NF_glutamine_log.append(self.kinetic_model.NF_glutamine_current)
        self._NF_AA_log.append(self.kinetic_model.NF_AA_current)
        self._NF_O2_log.append(self.kinetic_model.NF_O2_current)
        self._NF_log.append(self.kinetic_model.NF_current)
        self._IF_O2_log.append(self.kinetic_model.IF_O2_current)
        self._IF_CO2_log.append(self.kinetic_model.IF_CO2_current)
        self._IF_lactate_log.append(self.kinetic_model.IF_lactate_current)
        self._IF_ammonia_log.append(self.kinetic_model.IF_ammonia_current)
        self._IF_osmolality_log.append(self.kinetic_model.IF_osmolality_current)
        self._IF_log.append(self.kinetic_model.IF_current)
        self._Y_Glc_X_log.append(self.stoichiometric_model.Y_Glc_X_current)
        self._Y_Lac_Glc_log.append(self.stoichiometric_model.Y_Lac_Glc_current)
        self._Y_Gln_Glc_log.append(self.stoichiometric_model.Y_Gln_Glc_current)
        self._Y_O2_Glc_log.append(self.stoichiometric_model.Y_O2_Glc_current)
        self._Y_CO2_Glc_log.append(self.stoichiometric_model.Y_CO2_Glc_current)
        self._Y_AA_Glc_log.append(self.stoichiometric_model.Y_AA_Glc_current)
        self._Y_Amm_Gln_log.append(self.stoichiometric_model.Y_Amm_Gln_current)

        # Non-carbonate buffer concentrations and H+ flux terms
        for i in range(self.n_buffers):
            self._C_buffer_log[i].append(
                self._current_C_buffer[i] if i < len(self._current_C_buffer) else 0.0
            )
        self._dC_H_dt_from_buffer_log.append(self.pH_controller.dC_H_dt_from_buffer)
        self._dC_H_dt_from_Amm_log.append(self.pH_controller.dC_H_dt_from_Amm)
    
    def finalize_logging(self, t_solution: np.ndarray):
        """Interpolate logged values to match solution timepoints.
        
        Call this after ``scipy.integrate.solve_ivp`` completes to populate
        the final arrays.
        When ``self.log_details`` is False, D_list, D_base_list, D_acid_list,
        q_H2O_list, heat_rate_list, OTR_list, and CTR_list are still interpolated
        because they are needed for summary-statistics and mass-balance/total-
        inlet calculations; all other ``*_list`` arrays stay as their reset
        default (empty np.array).
        
        Parameters
        ----------
        t_solution : numpy.ndarray
            Accepted time points from the solver solution (h).
        """
        if len(self._t_log) == 0:
            return  # No data logged
        
        # Convert lists to arrays for interpolation
        t_log_arr = np.array(self._t_log)
        
        # D_list, D_base_list, D_acid_list, OTR_list, and CTR_list are always needed
        # by summary/mass-balance calculations (see the matching note in
        # _log_intermediates() above), so they are interpolated unconditionally.
        self.D_list = np.interp(t_solution, t_log_arr, self._D_log)
        self.D_base_list = np.interp(t_solution, t_log_arr, self._D_base_log)
        self.D_acid_list = np.interp(t_solution, t_log_arr, self._D_acid_log)
        self.q_H2O_list = np.interp(t_solution, t_log_arr, self._q_H2O_log)
        self.heat_rate_list = np.interp(t_solution, t_log_arr, self._heat_rate_log)
        self.OTR_list = np.interp(t_solution, t_log_arr, self._OTR_log)
        self.CTR_list = np.interp(t_solution, t_log_arr, self._CTR_log)

        if not self.log_details:
            return  # Skip remaining interpolation when detailed logging is off

        # Interpolate each logged quantity to solution timepoints
        self.N_list = np.interp(t_solution, t_log_arr, self._N_log)
        self.Fg_list = np.interp(t_solution, t_log_arr, self._Fg_log)
        self.P_over_V_list = np.interp(t_solution, t_log_arr, self._P_over_V_log)
        self.v_s_list = np.interp(t_solution, t_log_arr, self._v_s_log)
        self.y_O2_list = np.interp(t_solution, t_log_arr, self._y_O2_log)
        self.kLa_list = np.interp(t_solution, t_log_arr, self._kLa_log)
        self.C_CO2_list = np.interp(t_solution, t_log_arr, self._C_CO2_log)
        self.y_CO2_list = np.interp(t_solution, t_log_arr, self._y_CO2_log)
        self.osm_list = np.interp(t_solution, t_log_arr, self._osm_log)
        self.osm_background_list = np.interp(t_solution, t_log_arr, self._osm_background_log)
        
        # Interpolate kinetic model variables
        self.mu_list = np.interp(t_solution, t_log_arr, self._mu_log)
        self.q_Glc_list = np.interp(t_solution, t_log_arr, self._q_Glc_log)
        self.q_Gln_list = np.interp(t_solution, t_log_arr, self._q_Gln_log)
        self.q_Lac_list = np.interp(t_solution, t_log_arr, self._q_Lac_log)
        self.q_Amm_list = np.interp(t_solution, t_log_arr, self._q_Amm_log)
        self.q_AA_list = np.interp(t_solution, t_log_arr, self._q_AA_log)
        self.q_O2_list = np.interp(t_solution, t_log_arr, self._q_O2_log)
        self.q_CO2_list = np.interp(t_solution, t_log_arr, self._q_CO2_log)
        self.lag_term_list = np.interp(t_solution, t_log_arr, self._lag_term_log)
        self.NF_glucose_list = np.interp(t_solution, t_log_arr, self._NF_glucose_log)
        self.NF_glutamine_list = np.interp(t_solution, t_log_arr, self._NF_glutamine_log)
        self.NF_AA_list = np.interp(t_solution, t_log_arr, self._NF_AA_log)
        self.NF_O2_list = np.interp(t_solution, t_log_arr, self._NF_O2_log)
        self.NF_list = np.interp(t_solution, t_log_arr, self._NF_log)
        self.IF_O2_list = np.interp(t_solution, t_log_arr, self._IF_O2_log)
        self.IF_CO2_list = np.interp(t_solution, t_log_arr, self._IF_CO2_log)
        self.IF_lactate_list = np.interp(t_solution, t_log_arr, self._IF_lactate_log)
        self.IF_ammonia_list = np.interp(t_solution, t_log_arr, self._IF_ammonia_log)
        self.IF_osmolality_list = np.interp(t_solution, t_log_arr, self._IF_osmolality_log)
        self.IF_list = np.interp(t_solution, t_log_arr, self._IF_log)
        self.Y_Glc_X_list = np.interp(t_solution, t_log_arr, self._Y_Glc_X_log)
        self.Y_Lac_Glc_list = np.interp(t_solution, t_log_arr, self._Y_Lac_Glc_log)
        self.Y_Gln_Glc_list = np.interp(t_solution, t_log_arr, self._Y_Gln_Glc_log)
        self.Y_O2_Glc_list = np.interp(t_solution, t_log_arr, self._Y_O2_Glc_log)
        self.Y_CO2_Glc_list = np.interp(t_solution, t_log_arr, self._Y_CO2_Glc_log)
        self.Y_AA_Glc_list = np.interp(t_solution, t_log_arr, self._Y_AA_Glc_log)
        self.Y_Amm_Gln_list = np.interp(t_solution, t_log_arr, self._Y_Amm_Gln_log)

        # Interpolate non-carbonate buffer arrays
        if self.n_buffers > 0:
            self.C_buffer_list = np.array([
                np.interp(t_solution, t_log_arr, self._C_buffer_log[i])
                for i in range(self.n_buffers)
            ])  # shape: n_buffers × n_points
        self.dC_H_dt_from_buffer_list = np.interp(t_solution, t_log_arr, self._dC_H_dt_from_buffer_log)
        self.dC_H_dt_from_Amm_list = np.interp(t_solution, t_log_arr, self._dC_H_dt_from_Amm_log)

    def reset_logging(self):
        """Reset logs and interpolated arrays before a new batch phase.
        
        Clears all logged intermediate values and final arrays.
        Only touches the detailed lists when ``self.log_details`` is True.
        """
        # Always clear the essential logs needed by summary/mass-balance
        # calculations, including OTR/CTR (see the matching note in
        # _log_intermediates() — these are appended unconditionally, so they
        # must also be cleared unconditionally to stay in lock-step with
        # _t_log across phases/batches).
        self._t_log.clear()
        self._D_log.clear()
        self._D_base_log.clear()
        self._D_acid_log.clear()
        self._q_H2O_log.clear()
        self._heat_rate_log.clear()
        self._OTR_log.clear()
        self._CTR_log.clear()
        self.D_list = np.array([])
        self.D_base_list = np.array([])
        self.D_acid_list = np.array([])
        self.q_H2O_list = np.array([])
        self.heat_rate_list = np.array([])
        self.OTR_list = np.array([])
        self.CTR_list = np.array([])

        if not self.log_details:
            return  # Skip clearing detailed lists — they were never populated

        # Clear detailed dynamic logging lists
        self._N_log.clear()
        self._Fg_log.clear()
        self._P_over_V_log.clear()
        self._v_s_log.clear()
        self._y_O2_log.clear()
        self._kLa_log.clear()
        self._C_CO2_log.clear()
        self._y_CO2_log.clear()
        self._osm_log.clear()
        self._osm_background_log.clear()
        
        # Clear kinetic model logging lists
        self._mu_log.clear()
        self._q_Glc_log.clear()
        self._q_Gln_log.clear()
        self._q_Lac_log.clear()
        self._q_AA_log.clear()
        self._q_Amm_log.clear()
        self._q_O2_log.clear()
        self._q_CO2_log.clear()
        self._lag_term_log.clear()
        self._NF_glucose_log.clear()
        self._NF_glutamine_log.clear()
        self._NF_AA_log.clear()
        self._NF_O2_log.clear()
        self._NF_log.clear()
        self._IF_O2_log.clear()
        self._IF_CO2_log.clear()
        self._IF_lactate_log.clear()
        self._IF_ammonia_log.clear()
        self._IF_osmolality_log.clear()
        self._IF_log.clear()
        self._Y_Glc_X_log.clear()
        self._Y_Lac_Glc_log.clear()
        self._Y_Gln_Glc_log.clear()
        self._Y_O2_Glc_log.clear()
        self._Y_CO2_Glc_log.clear()
        self._Y_AA_Glc_log.clear()
        self._Y_Amm_Gln_log.clear()

        # Clear non-carbonate buffer logging lists
        for i in range(self.n_buffers):
            self._C_buffer_log[i].clear()
        self._dC_H_dt_from_buffer_log.clear()
        self._dC_H_dt_from_Amm_log.clear()

        # Reset final arrays to empty (OTR_list/CTR_list are reset unconditionally above)
        self.N_list = np.array([])
        self.Fg_list = np.array([])
        self.P_over_V_list = np.array([])
        self.v_s_list = np.array([])
        self.y_O2_list = np.array([])
        self.kLa_list = np.array([])
        self.C_CO2_list = np.array([])
        self.y_CO2_list = np.array([])
        self.osm_list = np.array([])
        self.osm_background_list = np.array([])
        
        # Reset kinetic model final arrays
        self.mu_list = np.array([])
        self.q_Glc_list = np.array([])
        self.q_Gln_list = np.array([])
        self.q_Lac_list = np.array([])
        self.q_AA_list = np.array([])
        self.q_Amm_list = np.array([])
        self.q_O2_list = np.array([])
        self.q_CO2_list = np.array([])
        self.lag_term_list = np.array([])
        self.NF_glucose_list = np.array([])
        self.NF_glutamine_list = np.array([])
        self.NF_AA_list = np.array([])
        self.NF_O2_list = np.array([])
        self.NF_list = np.array([])
        self.IF_O2_list = np.array([])
        self.IF_CO2_list = np.array([])
        self.IF_lactate_list = np.array([])
        self.IF_ammonia_list = np.array([])
        self.IF_osmolality_list = np.array([])
        self.IF_list = np.array([])
        self.Y_Glc_X_list = np.array([])
        self.Y_Lac_Glc_list = np.array([])
        self.Y_Gln_Glc_list = np.array([])
        self.Y_O2_Glc_list = np.array([])
        self.Y_CO2_Glc_list = np.array([])
        self.Y_AA_Glc_list = np.array([])
        self.Y_Amm_Gln_list = np.array([])
        # Reset non-carbonate buffer final arrays
        self.C_buffer_list = np.array([])
        self.dC_H_dt_from_buffer_list = np.array([])
        self.dC_H_dt_from_Amm_list = np.array([])
        self._current_C_buffer = []
    
    def _update_shared_models(self, t: float, y: List[float], D: float, C_O2_in: float,
                             C_H_in: float, C_CO2_in: float, C_HCO3_in: float, C_CO3_in: float,
                             log_values: bool = True, D_max: Optional[float] = None):
        """Compute all shared state by updating kinetic model, gas controller, and pH controller.
        
        This method centralizes pH equilibrium, osmolality, kinetic-rate,
        gas-transfer, and pH-control calculations. It performs a two-pass
        controller update so the gas controller and pH controller use a
        consistent total inlet rate, including base and acid additions. After
        it returns, current model attributes (for example,
        ``kinetic_model.mu_current`` and ``gas_controller.OTR_current``) are
        available to the mode-specific balance equations.
        
        Parameters
        ----------
        t : float
            Current simulation time (h).
        y : list[float]
            State vector ``[Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT,
            C_H, C_Na, C_AA, C_Osm_bg, C_buffer_0, ...]``. Buffer states are
            appended when configured.
        D : float
            Medium dilution rate (1/h).
        C_O2_in : float
            Feed oxygen concentration (mmol/L).
        C_H_in : float
            Feed hydrogen-ion concentration (mol/L).
        C_CO2_in : float
            Feed dissolved-carbon-dioxide concentration (mmol/L).
        C_HCO3_in : float
            Feed bicarbonate concentration (mmol/L).
        C_CO3_in : float
            Feed carbonate concentration (mmol/L).
        log_values : bool, default=True
            Whether to append evaluated signals to intermediate logs.
        D_max : float, optional
            Upper bound on total inlet dilution rate (1/h).
        """
        # Unpack state variables (core 12 states + optional buffer states)
        Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA, C_Osm_bg = y[:12]
        C_buffer_list = list(y[12:]) if len(y) > 12 else []
        self._current_C_buffer = C_buffer_list
        self.C_Osm_bg_current = C_Osm_bg
        
        # Calculate pH equilibrium and osmolality. Based on the new C_CT and C_H values, it will compute C_CO2. Needs to be before kinetic model update as C_CO2 is needed there.
        self.pH_controller.calculate_equilibrium(C_CT, C_H)
        C_CO2 = self.pH_controller.C_H2CO3
        HCO3 = self.pH_controller.C_HCO3
        CO3 = self.pH_controller.C_CO3
        # Store osmolality in kinetic model for logging
        self.kinetic_model.osm_current = C_Osm_bg + C_Glc + C_Gln + C_Lac + C_Amm + C_Na + C_AA + HCO3 + CO3 + self.pH_controller.get_buffer_osm_contribution(C_buffer_list)
        
        # Update kinetic model with current state and stoichiometric model.
        # StoichiometricKinetics computes mu first, then calls stoichiometric_model.calculate_yields(mu)
        # internally, ensuring yields are always consistent with the current growth rate.
        # In CONSTANT mode this is a free cached lookup; in DYNAMIC mode yields are recalculated.
        self.kinetic_model.calculate_rates(t, y, C_CO2)
        
        def _effective_inlet_concentrations(d_base: float, d_acid: float) -> tuple[float, float, float, float, float, float]:
            d_total = D + d_base + d_acid
            if d_total <= 0:
                return d_total, 0.0, 0.0, 0.0, 0.0, 0.0
            return (
                d_total,
                D * C_O2_in / d_total,
                D * C_H_in / d_total,
                D * C_CO2_in / d_total,
                D * C_HCO3_in / d_total,
                D * C_CO3_in / d_total,
            )

        # Base and acid addition are computed by the pH controller, but the gas controller
        # also needs the current total hydraulic turnover once chemical additions become a
        # real inlet.  Use a short two-pass update so both controllers see a consistent
        # flow state.  First pass uses the D_base/D_acid from the previous timepoint.
        D_base = self.pH_controller.D_base
        D_acid = self.pH_controller.D_acid
        D_total, C_O2_in_eff, C_H_in_eff, C_CO2_in_eff, C_HCO3_in_eff, C_CO3_in_eff = _effective_inlet_concentrations(D_base, D_acid)

        # Apply combined flow rate cap (feed + base + acid ≤ D_max) after first pass.
        if D_max is not None and D_total > D_max:
            D_chemical = D_base + D_acid
            if D_chemical > 0:
                allowed = max(D_max - D, 0.0)
                scale = allowed / D_chemical
                D_base = D_base * scale
                D_acid = D_acid * scale
                self.pH_controller.D_base = D_base
                self.pH_controller.D_acid = D_acid
                D_total, C_O2_in_eff, C_H_in_eff, C_CO2_in_eff, C_HCO3_in_eff, C_CO3_in_eff = _effective_inlet_concentrations(D_base, D_acid)

        self.gas_controller.control_DO(V=V, D=D_total, C_O2_in=C_O2_in_eff, C_O2=C_O2,
                                       q_O2=self.kinetic_model.q_O2_current, Xv=Xv)
        self.pH_controller.control_pH(V=V, D=D_total, C_H_in=C_H_in_eff, C_CO2_in=C_CO2_in_eff,
                                      C_HCO3_in=C_HCO3_in_eff, C_CO3_in=C_CO3_in_eff,
                                      q_Lac=self.kinetic_model.q_Lac_current,
                                      q_CO2=self.kinetic_model.q_CO2_current,
                                      q_Amm=self.kinetic_model.q_Amm_current, Xv=Xv,
                                      C_buffer_list=C_buffer_list)

        # Second pass: re-read D_base/D_acid from pH controller (may have changed), apply cap again.
        D_base = self.pH_controller.D_base
        D_acid = self.pH_controller.D_acid
        D_total, C_O2_in_eff, C_H_in_eff, C_CO2_in_eff, C_HCO3_in_eff, C_CO3_in_eff = _effective_inlet_concentrations(D_base, D_acid)

        if D_max is not None and D_total > D_max:
            D_chemical = D_base + D_acid
            if D_chemical > 0:
                allowed = max(D_max - D, 0.0)
                scale = allowed / D_chemical
                self.pH_controller.D_base = D_base * scale
                self.pH_controller.D_acid = D_acid * scale
                D_total, C_O2_in_eff, C_H_in_eff, C_CO2_in_eff, C_HCO3_in_eff, C_CO3_in_eff = _effective_inlet_concentrations(self.pH_controller.D_base, self.pH_controller.D_acid)

        self.gas_controller.control_DO(V=V, D=D_total, C_O2_in=C_O2_in_eff, C_O2=C_O2,
                                       q_O2=self.kinetic_model.q_O2_current, Xv=Xv)
        self.pH_controller.control_pH(V=V, D=D_total, C_H_in=C_H_in_eff, C_CO2_in=C_CO2_in_eff,
                                      C_HCO3_in=C_HCO3_in_eff, C_CO3_in=C_CO3_in_eff,
                                      q_Lac=self.kinetic_model.q_Lac_current,
                                      q_CO2=self.kinetic_model.q_CO2_current,
                                      q_Amm=self.kinetic_model.q_Amm_current, Xv=Xv,
                                      C_buffer_list=C_buffer_list)
                
        # Log intermediate values
        if log_values:
            self._log_intermediates(t, D)

    def _compute_buffer_derivatives(self, C_buffer_list: List[float], D_medium: float, D_base: float) -> List[float]:
        """Compute dC_buffer_dt for each non-carbonate buffer.

        Buffer total concentration changes only via feed input and dilution/washout
        (buffer "consumption" is a speciation shift handled algebraically, not a
        mass sink).  The general form for each buffer *i* is:

            dC_buffer_dt[i] = (C_feed[i] - C_buffer[i]) * D_medium
                              - C_buffer[i] * D_base

        In constant-volume modes (chemostat, perfusion) D_medium > 0 with
        matching harvest, so the buffer reaches a steady state where feed-in =
        washout.  In batch mode D_medium = 0 and only base dilution applies.

        Parameters
        ----------
        C_buffer_list : list[float]
            Current total concentrations of each buffer (mmol/L).
        D_medium : float
            Medium (feed/perfusion) dilution rate excluding base (1/h).
        D_base : float
            Base-addition dilution rate (1/h).

        Returns
        -------
        list[float]
            dC_buffer_dt for each buffer (mmol/L/h).  Empty list when no buffers.
        """
        if self.n_buffers == 0:
            return []
        derivs = []
        for i in range(self.n_buffers):
            C_feed_i = self.pH_controller._buffer_C_feed[i]
            C_buf_i = C_buffer_list[i] if i < len(C_buffer_list) else 0.0
            dC_dt = (C_feed_i - C_buf_i) * D_medium - C_buf_i * D_base
            derivs.append(dC_dt)
        return derivs

    @abstractmethod
    def _cache_parameters(self, params: SimulationParameters):
        """Cache parameters needed by this operating-mode implementation."""
        pass

    @abstractmethod
    def rhs(self, t: float, y: List[float], params: SimulationParameters) -> List[float]: # Params not required in method but required for solve_ivp event functions
        """Compute the state derivatives for the active operating mode.

        Implementations must call ``_update_shared_models`` before using
        kinetic, gas-transfer, or pH-control attributes in their balances.

        Parameters
        ----------
        t : float
            Current simulation time (h).
        y : list[float]
            State vector ``[Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT,
            C_H, C_Na, C_AA, C_Osm_bg, C_buffer_0, ...]``. ``Xv`` is in
            cells/L; medium concentrations and buffer totals are in mmol/L;
            ``V`` is in L; ``C_H`` is in mol/L; and ``C_Osm_bg`` is in mOsm.
        params : SimulationParameters
            Simulation parameters. This argument is retained for solver event
            functions.

        Returns
        -------
        list[float]
            Derivatives in the same order as ``y``, including one derivative
            for each configured buffer state.
        """


class BatchODESystem(ODESystem):
    """Closed-culture balances with no medium feed or harvest.

    Culture volume can still increase through pH-control base or acid addition.
    Nutrients and buffer totals are therefore diluted by those chemical flows.
    """

    def __init__(self, params: SimulationParameters, stoichiometric_model: StoichiometricModel, kinetic_model: KineticModel, gas_control_model: GasControllerModel):
        """Create a batch ODE system using the shared process models."""
        super().__init__(params, stoichiometric_model, kinetic_model, gas_control_model)

    def _cache_parameters(self, params: SimulationParameters):
        """Cache frequently used parameters"""
        pass # Placeholder

    def rhs(self, t: float, y: List[float], params: SimulationParameters) -> List[float]:
        """Return batch-phase derivatives.

        See ``ODESystem.rhs`` for the state-vector order.
        """
        y = self._clamp_y(y)
        # Unpack state variables (core 12 + optional buffer states)
        Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA, C_Osm_bg = y[:12]
        C_buffer_list = list(y[12:]) if len(y) > 12 else []
        
        # Compute all shared state (updates kinetic model, gas controller, pH controller)
        self._update_shared_models(t, y, D=0, C_O2_in=0, C_H_in=0, 
                                   C_CO2_in=0, C_HCO3_in=0, C_CO3_in=0)

        D_base = self.pH_controller.D_base
        D_acid = self.pH_controller.D_acid
        D_chem = D_base + D_acid  # Total chemical addition rate (base + acid)

        # ODEs. Cell count [cells / L] * specific production rates [mmol/cell/h]
        dXv_dt   = Xv * (self.kinetic_model.mu_current - D_chem)
        dC_Glc_dt = - C_Glc * D_chem - Xv * self.kinetic_model.q_Glc_current
        dC_Gln_dt = - C_Gln * D_chem - Xv * self.kinetic_model.q_Gln_current
        dC_Lac_dt = - C_Lac * D_chem + Xv * self.kinetic_model.q_Lac_current
        dC_Amm_dt = - C_Amm * D_chem + Xv * self.kinetic_model.q_Amm_current
        dV_dt    = V * D_chem
        dC_O2_dt   = - C_O2 * D_chem - Xv * self.kinetic_model.q_O2_current + self.gas_controller.OTR_current
        dC_CT_dt  = - C_CT * D_chem + Xv * self.kinetic_model.q_CO2_current + self.pH_controller.CTR_current
        dC_H_dt_raw   = - C_H * D_chem + self.pH_controller.dC_H_dt_from_Lac / 1000 + self.pH_controller.dC_H_dt_from_Amm / 1000 + self.pH_controller.dC_H_dt_from_carbonate / 1000 + self.pH_controller.CTR_current * self.pH_controller.H_per_CO2 / 1000 - D_base * self.pH_controller.C_OH_base + D_acid * self.pH_controller.C_H_acid + self.pH_controller.dC_H_dt_from_buffer / 1000
        # Apply non-carbonate buffer capacity: buffers resist pH drift by factor (1 + β_total)
        _beta_total = self.pH_controller.compute_buffer_capacity(C_buffer_list, C_H)
        dC_H_dt_raw = dC_H_dt_raw / (1.0 + _beta_total)
        dC_H_dt = 0.0 if abs(dC_H_dt_raw) < 1e-12 else dC_H_dt_raw
        dC_Na_dt_raw = - C_Na * D_chem + D_base * self.pH_controller.C_Na_base * 1000 + D_acid * self.pH_controller.C_Na_acid * 1000
        dC_Na_dt = 0.0 if abs(dC_Na_dt_raw) < 1e-12 else dC_Na_dt_raw
        dC_AA_dt = - C_AA * D_chem - Xv * self.kinetic_model.q_AA_current
        dC_Osm_bg_dt = - C_Osm_bg * D_chem
        # Buffer derivatives: batch has no feed (D_medium=0), only base/acid dilution
        dC_buffer_dt = self._compute_buffer_derivatives(C_buffer_list, D_medium=0.0, D_base=D_chem)
        return [dXv_dt, dC_Glc_dt, dC_Gln_dt, dC_Lac_dt, dC_Amm_dt, dV_dt, dC_O2_dt, dC_CT_dt, dC_H_dt, dC_Na_dt, dC_AA_dt, dC_Osm_bg_dt] + dC_buffer_dt

    def get_required_params(self):
        return []


# Concrete ODE system for fed-batch
class FedBatchODESystem(ODESystem):
    """Fed-batch balances with a feed-rate strategy and no harvest.

    Feed volume and pH-control chemical additions increase culture volume. The
    total inlet rate is capped by ``reactor.D_max`` when necessary.
    """

    def __init__(self, params: SimulationParameters, stoichiometric_model: StoichiometricModel, kinetic_model: KineticModel, feed_strategy_model: FeedStrategyModel, gas_control_model: GasControllerModel):
        """Create a fed-batch system.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters containing ``fed_batch`` settings.
        stoichiometric_model : StoichiometricModel
            Model supplying current yield coefficients.
        kinetic_model : KineticModel
            Model supplying growth and cell-specific rates.
        feed_strategy_model : FeedStrategyModel
            Model that calculates the liquid feed rate.
        gas_control_model : GasControllerModel
            Model that calculates oxygen transfer and dissolved-oxygen control.
        """
        self.feed_strategy_model = feed_strategy_model
        super().__init__(params, stoichiometric_model, kinetic_model, gas_control_model)

        # Use pH controller to equilibrate carbonate species in feed medium
        self.pH_controller.calculate_equilibrium(self.C_CT_feed, self.C_H_feed)  # Equilibrate carbonate species in feed medium at pH of feed medium
        self.C_CO2_feed, self.C_HCO3_feed, self.C_CO3_feed = self.pH_controller.C_H2CO3, self.pH_controller.C_HCO3, self.pH_controller.C_CO3
        self.pH_controller.calculate_equilibrium(self.pH_controller.C_CT_init, self.pH_controller.C_H_init)  # Reset the carbonate species to initial culture conditions


    def _cache_parameters(self, params: SimulationParameters):
        """Cache fed-batch medium concentrations and the hydraulic rate limit."""
        # Access parameters through SimulationParameters
        self.C_Glc_feed = float(params.fed_batch.C_Glc_feed)
        self.C_Gln_feed = float(params.fed_batch.C_Gln_feed)
        self.C_AA_feed = float(params.fed_batch.C_AA_feed)
        self.C_Osm_bg_feed = float(params.fed_batch.osm_bg_feed)
        self.C_O2_feed = float(params.shared_medium_properties.C_O2)
        self.C_CT_feed = float(params.shared_medium_properties.C_CT)
        self.C_H_feed = float(params.shared_medium_properties.C_H)
        # Maximum total dilution rate (feed + base + acid) to prevent
        # unrealistic volume growth.  Equal to reactor.D_max which is
        # numerically F_L_max_specific.
        self.D_max = float(params.reactor.D_max)        

    def rhs(self, t: float, y: List[float], params: SimulationParameters) -> List[float]:
        """Return fed-batch derivatives.

        See ``ODESystem.rhs`` for the state-vector order.
        """
        y = self._clamp_y(y)
        # Unpack state variables (core 12 + optional buffer states) and calculate feed rate
        Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA, C_Osm_bg = y[:12]
        C_buffer_list = list(y[12:]) if len(y) > 12 else []

        F_L = self.feed_strategy_model.calculate_feed_rate(
            t,
            y,
            q_Glc=self.kinetic_model.q_Glc_current,
            D_base=self.pH_controller.D_base,
            C_Glc_feed=self.C_Glc_feed,
        )
        D = F_L / V
        
        if isinstance(self.feed_strategy_model, GlucoseSetpointFeedStrategy):
            # First pass refreshes q_Glc and pH-controller D_base before
            # algebraic feed backcalculation.
            self._update_shared_models(t, y, D=D, C_O2_in=self.C_O2_feed, C_H_in=self.C_H_feed,
                                       C_CO2_in=self.C_CO2_feed, C_HCO3_in=self.C_HCO3_feed,
                                       C_CO3_in=self.C_CO3_feed, log_values=False, D_max=self.D_max)

            F_L = self.feed_strategy_model.calculate_feed_rate(
                t,
                y,
                q_Glc=self.kinetic_model.q_Glc_current,
                D_base=self.pH_controller.D_base,
                C_Glc_feed=self.C_Glc_feed,
            )
            D = F_L / V

    
        # Compute all shared state (updates kinetic model, gas controller, pH controller)
        self._update_shared_models(t, y, D=D, C_O2_in=self.C_O2_feed, C_H_in=self.C_H_feed,
                                   C_CO2_in=self.C_CO2_feed, C_HCO3_in=self.C_HCO3_feed, 
                                   C_CO3_in=self.C_CO3_feed, D_max=self.D_max)
        
        
        D_base = self.pH_controller.D_base
        D_acid = self.pH_controller.D_acid
        D_total = D + D_base + D_acid

        # ODEs
        dXv_dt   = Xv * (self.kinetic_model.mu_current - D_total)
        dC_Glc_dt = (self.C_Glc_feed - C_Glc) * D - C_Glc * (D_base + D_acid) - Xv * self.kinetic_model.q_Glc_current
        dC_Gln_dt = (self.C_Gln_feed - C_Gln) * D - C_Gln * (D_base + D_acid) - Xv * self.kinetic_model.q_Gln_current
        dC_Lac_dt = -C_Lac * D_total + Xv * self.kinetic_model.q_Lac_current
        dC_Amm_dt = -C_Amm * D_total + Xv * self.kinetic_model.q_Amm_current
        dV_dt    = V * D_total
        dC_O2_dt   = (self.C_O2_feed - C_O2) * D - C_O2 * (D_base + D_acid) - Xv * self.kinetic_model.q_O2_current + self.gas_controller.OTR_current
        dC_CT_dt  = (self.C_CO2_feed + self.C_HCO3_feed + self.C_CO3_feed - C_CT) * D - C_CT * (D_base + D_acid) + Xv * self.kinetic_model.q_CO2_current + self.pH_controller.CTR_current
        dC_H_dt_raw   = (self.C_H_feed - C_H) * D - C_H * (D_base + D_acid) + self.pH_controller.dC_H_dt_from_Lac / 1000 + self.pH_controller.dC_H_dt_from_Amm / 1000 + self.pH_controller.dC_H_dt_from_carbonate / 1000 + self.pH_controller.CTR_current * self.pH_controller.H_per_CO2 / 1000 - D_base * self.pH_controller.C_OH_base + D_acid * self.pH_controller.C_H_acid + self.pH_controller.dC_H_dt_from_buffer / 1000
        # Apply non-carbonate buffer capacity: buffers resist pH drift by factor (1 + β_total)
        _beta_total = self.pH_controller.compute_buffer_capacity(C_buffer_list, C_H)
        dC_H_dt_raw = dC_H_dt_raw / (1.0 + _beta_total)
        dC_H_dt = 0.0 if abs(dC_H_dt_raw) < 1e-12 else dC_H_dt_raw
        dC_Na_dt_raw = - C_Na * D_total + D_base * self.pH_controller.C_Na_base * 1000 + D_acid * self.pH_controller.C_Na_acid * 1000
        dC_Na_dt = 0.0 if abs(dC_Na_dt_raw) < 1e-12 else dC_Na_dt_raw
        dC_AA_dt = (self.C_AA_feed - C_AA) * D - C_AA * (D_base + D_acid) - Xv * self.kinetic_model.q_AA_current
        dC_Osm_bg_dt = (self.C_Osm_bg_feed - C_Osm_bg) * D - C_Osm_bg * (D_base + D_acid)
        # Buffer derivatives: feed dilution D, base/acid dilution
        dC_buffer_dt = self._compute_buffer_derivatives(C_buffer_list, D_medium=D, D_base=D_base + D_acid)
        return [dXv_dt, dC_Glc_dt, dC_Gln_dt, dC_Lac_dt, dC_Amm_dt, dV_dt, dC_O2_dt, dC_CT_dt, dC_H_dt, dC_Na_dt, dC_AA_dt, dC_Osm_bg_dt] + dC_buffer_dt
    
    def get_required_params(self):
        param_list = ['fed_batch.F_L', 'fed_batch.C_Glc_feed', 'fed_batch.C_Gln_feed', 'fed_batch.C_AA_feed']
        param_list += self.feed_strategy_model.get_required_params()
        return param_list

class ChemostatODESystem(ODESystem):
    """Constant-volume chemostat balances with a configured medium dilution rate.

    The configured rate is limited to ``reactor.D_max``. Base and acid addition
    are included as inlet and matching outlet flows, preserving vessel volume.
    """

    def __init__(self, params: SimulationParameters, stoichiometric_model: StoichiometricModel, kinetic_model: KineticModel, gas_control_model: GasControllerModel):
        """Create a chemostat system using the shared process models."""
        super().__init__(params, stoichiometric_model, kinetic_model, gas_control_model)

        # Use pH controller to equilibrate carbonate species in feed medium
        self.pH_controller.calculate_equilibrium(self.C_CT_feed, self.C_H_feed)  # Equilibrate carbonate species in feed medium at pH of feed medium
        self.C_CO2_feed, self.C_HCO3_feed, self.C_CO3_feed = self.pH_controller.C_H2CO3, self.pH_controller.C_HCO3, self.pH_controller.C_CO3
        self.pH_controller.calculate_equilibrium(self.pH_controller.C_CT_init, self.pH_controller.C_H_init)  # Reset the carbonate species to initial culture conditions


    def _cache_parameters(self, params: SimulationParameters):
        """Cache chemostat feed composition and the configured dilution rate."""
        assert params.chemostat is not None, "Chemostat parameters required for ChemostatODESystem"

        D_configured = float(params.chemostat.D)  # Dilution rate (1/h)
        self.D_max = float(params.reactor.D_max)
        self.D = min(D_configured, self.D_max)  # Clamp to reactor maximum
        # Access parameters through SimulationParameters
        self.C_Glc_feed = float(params.chemostat.C_Glc_feed)
        self.C_Gln_feed = float(params.chemostat.C_Gln_feed)
        self.C_AA_feed = float(params.chemostat.C_AA_feed)
        self.C_Osm_bg_feed = float(params.chemostat.osm_bg_feed)
        self.C_O2_feed = float(params.shared_medium_properties.C_O2)
        self.C_CT_feed = float(params.shared_medium_properties.C_CT)
        self.C_H_feed = float(params.shared_medium_properties.C_H)        


    def rhs(self, t: float, y: List[float], params: SimulationParameters) -> List[float]:
        """Return chemostat derivatives.

        See ``ODESystem.rhs`` for the state-vector order.
        """
        y = self._clamp_y(y)
        # Unpack state variables (core 12 + optional buffer states)
        Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA, C_Osm_bg = y[:12]
        C_buffer_list = list(y[12:]) if len(y) > 12 else []
        
        # Compute all shared state (updates kinetic model, gas controller, pH controller)
        self._update_shared_models(t, y, D=self.D, C_O2_in=self.C_O2_feed, C_H_in=self.C_H_feed,
                                   C_CO2_in=self.C_CO2_feed, C_HCO3_in=self.C_HCO3_feed, 
                                   C_CO3_in=self.C_CO3_feed, D_max=self.D_max)
        
        D_base = self.pH_controller.D_base
        D_acid = self.pH_controller.D_acid
        D_total = self.D + D_base + D_acid

        # ODEs
        dXv_dt   = Xv * (self.kinetic_model.mu_current - D_total)
        dC_Glc_dt = (self.C_Glc_feed - C_Glc) * self.D - C_Glc * (D_base + D_acid) - Xv * self.kinetic_model.q_Glc_current
        dC_Gln_dt = (self.C_Gln_feed - C_Gln) * self.D - C_Gln * (D_base + D_acid) - Xv * self.kinetic_model.q_Gln_current
        dC_Lac_dt = -C_Lac * D_total + Xv * self.kinetic_model.q_Lac_current
        dC_Amm_dt = -C_Amm * D_total + Xv * self.kinetic_model.q_Amm_current
        dV_dt    = 0
        dC_O2_dt   = (self.C_O2_feed - C_O2) * self.D - C_O2 * (D_base + D_acid) - Xv * self.kinetic_model.q_O2_current + self.gas_controller.OTR_current
        dC_CT_dt  = (self.C_CO2_feed + self.C_HCO3_feed + self.C_CO3_feed - C_CT) * self.D - C_CT * (D_base + D_acid) + Xv * self.kinetic_model.q_CO2_current + self.pH_controller.CTR_current
        dC_H_dt_raw   = (self.C_H_feed - C_H) * self.D - C_H * (D_base + D_acid) + self.pH_controller.dC_H_dt_from_Lac / 1000 + self.pH_controller.dC_H_dt_from_Amm / 1000 + self.pH_controller.dC_H_dt_from_carbonate / 1000 + self.pH_controller.CTR_current * self.pH_controller.H_per_CO2 / 1000 - D_base * self.pH_controller.C_OH_base + D_acid * self.pH_controller.C_H_acid + self.pH_controller.dC_H_dt_from_buffer / 1000
        # Apply non-carbonate buffer capacity: buffers resist pH drift by factor (1 + β_total)
        _beta_total = self.pH_controller.compute_buffer_capacity(C_buffer_list, C_H)
        dC_H_dt_raw = dC_H_dt_raw / (1.0 + _beta_total)
        dC_H_dt = 0.0 if abs(dC_H_dt_raw) < 1e-12 else dC_H_dt_raw
        dC_Na_dt_raw = - C_Na * D_total + D_base * self.pH_controller.C_Na_base * 1000 + D_acid * self.pH_controller.C_Na_acid * 1000
        dC_Na_dt = 0.0 if abs(dC_Na_dt_raw) < 1e-12 else dC_Na_dt_raw
        dC_AA_dt = (self.C_AA_feed - C_AA) * self.D - C_AA * (D_base + D_acid) - Xv * self.kinetic_model.q_AA_current
        dC_Osm_bg_dt = (self.C_Osm_bg_feed - C_Osm_bg) * self.D - C_Osm_bg * (D_base + D_acid)
        # Buffer derivatives: feed dilution self.D, base/acid dilution
        dC_buffer_dt = self._compute_buffer_derivatives(C_buffer_list, D_medium=self.D, D_base=D_base + D_acid)
        return [dXv_dt, dC_Glc_dt, dC_Gln_dt, dC_Lac_dt, dC_Amm_dt, dV_dt, dC_O2_dt, dC_CT_dt, dC_H_dt, dC_Na_dt, dC_AA_dt, dC_Osm_bg_dt] + dC_buffer_dt

    def get_required_params(self):
            return ['chemostat.C_Glc_feed', 'chemostat.C_Gln_feed', 'chemostat.C_AA_feed', 'chemostat.D', 'reactor.D_max']

# Concrete ODE system for perfusion
class PerfusionODESystem(ODESystem):
    """Constant-volume perfusion balances with cell retention.

    ``perfusion_strategy_model`` determines medium dilution. The cell-loss
    term accounts for separator efficiency; with ``continuous=True``, the
    configured bleed rate is included as well.
    """

    def __init__(self, params: SimulationParameters, stoichiometric_model: StoichiometricModel, kinetic_model: KineticModel, perfusion_strategy_model: PerfusionStrategyModel, gas_control_model: GasControllerModel, continuous: bool = False):
        """Create a perfusion ODE system.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters containing ``perfusion`` settings.
        stoichiometric_model : StoichiometricModel
            Model supplying current yield coefficients.
        kinetic_model : KineticModel
            Model supplying growth and cell-specific rates.
        perfusion_strategy_model : PerfusionStrategyModel
            Model that calculates medium dilution.
        gas_control_model : GasControllerModel
            Model that calculates oxygen transfer and dissolved-oxygen control.
        continuous : bool, default=False
            Whether to include ``perfusion.B`` as the bleed rate.
        """
        self.perfusion_strategy_model = perfusion_strategy_model
        self.continuous = continuous
        super().__init__(params, stoichiometric_model, kinetic_model, gas_control_model)

        # Use pH controller to equilibrate carbonate species in feed medium
        self.pH_controller.calculate_equilibrium(self.C_CT_feed, self.C_H_feed)  # Equilibrate carbonate species in feed medium at pH of feed medium
        self.C_CO2_feed, self.C_HCO3_feed, self.C_CO3_feed = self.pH_controller.C_H2CO3, self.pH_controller.C_HCO3, self.pH_controller.C_CO3
        self.pH_controller.calculate_equilibrium(self.pH_controller.C_CT_init, self.pH_controller.C_H_init)  # Reset the carbonate species to initial culture conditions


    def _cache_parameters(self, params: SimulationParameters):
        """Cache perfusion feed composition, separator settings, and rate limit."""
        assert params.perfusion is not None, "Perfusion parameters required for PerfusionODESystem"

        # Access parameters through SimulationParameters
        self.C_Glc_Pfeed = float(params.perfusion.C_Glc_feed)
        self.C_Gln_Pfeed = float(params.perfusion.C_Gln_feed)
        self.C_AA_feed = float(params.perfusion.C_AA_feed)
        self.C_Osm_bg_feed = float(params.perfusion.osm_bg_feed)
        self.eff_separator = float(params.perfusion.eff_separator)
        self.B = float(params.perfusion.B) if self.continuous else 0.0
        self.C_O2_feed = float(params.shared_medium_properties.C_O2)
        self.C_CT_feed = float(params.shared_medium_properties.C_CT)
        self.C_H_feed = float(params.shared_medium_properties.C_H)
        # Maximum total dilution rate (feed + base + acid) to prevent
        # unrealistic volume growth.
        self.D_max = float(params.reactor.D_max)        


    def rhs(self, t: float, y: List[float], params: SimulationParameters) -> List[float]:
        """Return perfusion derivatives.

        See ``ODESystem.rhs`` for the state-vector order.
        """
        y = self._clamp_y(y)
        # Unpack state variables (core 12 + optional buffer states) and calculate dilution rate
        Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA, C_Osm_bg = y[:12]
        C_buffer_list = list(y[12:]) if len(y) > 12 else []
        D = self.perfusion_strategy_model.calculate_dilution_rate(
            t,
            y,
            q_Glc=self.kinetic_model.q_Glc_current,
            D_base=self.pH_controller.D_base + self.pH_controller.D_acid,
            C_Glc_feed=self.C_Glc_Pfeed,
        )

        if isinstance(self.perfusion_strategy_model, GlucoseSetpointPerfusionStrategy):
            # First pass updates q_Glc and D_base for algebraic dilution calculation.
            # Keep only the final corrected pass in logs.
            self._update_shared_models(t, y, D=D, C_O2_in=self.C_O2_feed, C_H_in=self.C_H_feed,
                                       C_CO2_in=self.C_CO2_feed, C_HCO3_in=self.C_HCO3_feed,
                                       C_CO3_in=self.C_CO3_feed, log_values=False, D_max=self.D_max)

            D = self.perfusion_strategy_model.calculate_dilution_rate(
                t,
                y,
                q_Glc=self.kinetic_model.q_Glc_current,
                D_base=self.pH_controller.D_base + self.pH_controller.D_acid,
                C_Glc_feed=self.C_Glc_Pfeed,
            )
        
        # Compute all shared state (updates kinetic model, gas controller, pH controller)
        self._update_shared_models(t, y, D=D, C_O2_in=self.C_O2_feed, C_H_in=self.C_H_feed,
                                   C_CO2_in=self.C_CO2_feed, C_HCO3_in=self.C_HCO3_feed, 
                                   C_CO3_in=self.C_CO3_feed, D_max=self.D_max)
        

        D_base = self.pH_controller.D_base
        D_acid = self.pH_controller.D_acid
        D_total_chem = D + D_base + D_acid

        # Pre-compute perfusion-specific terms
        mu = self.kinetic_model.mu_current
        H = D_total_chem - self.B
        eff_loss = H * (1 - self.eff_separator)

        # ODEs
        dXv_dt   = Xv * (mu - self.B - eff_loss)
        dC_Glc_dt = (self.C_Glc_Pfeed - C_Glc) * D - C_Glc * (D_base + D_acid) - Xv * self.kinetic_model.q_Glc_current
        dC_Gln_dt = (self.C_Gln_Pfeed - C_Gln) * D - C_Gln * (D_base + D_acid) - Xv * self.kinetic_model.q_Gln_current
        dC_Lac_dt = -C_Lac * D_total_chem + Xv * self.kinetic_model.q_Lac_current
        dC_Amm_dt = -C_Amm * D_total_chem + Xv * self.kinetic_model.q_Amm_current
        dV_dt    = 0
        dC_O2_dt   = (self.C_O2_feed - C_O2) * D - C_O2 * (D_base + D_acid) - Xv * self.kinetic_model.q_O2_current + self.gas_controller.OTR_current
        dC_CT_dt  = (self.C_CO2_feed + self.C_HCO3_feed + self.C_CO3_feed - C_CT) * D - C_CT * (D_base + D_acid) + Xv * self.kinetic_model.q_CO2_current + self.pH_controller.CTR_current
        dC_H_dt_raw   = (self.C_H_feed - C_H) * D - C_H * (D_base + D_acid) + self.pH_controller.dC_H_dt_from_Lac / 1000 + self.pH_controller.dC_H_dt_from_Amm / 1000 + self.pH_controller.dC_H_dt_from_carbonate / 1000 + self.pH_controller.CTR_current * self.pH_controller.H_per_CO2 / 1000 - D_base * self.pH_controller.C_OH_base + D_acid * self.pH_controller.C_H_acid + self.pH_controller.dC_H_dt_from_buffer / 1000
        # Apply non-carbonate buffer capacity: buffers resist pH drift by factor (1 + β_total)
        _beta_total = self.pH_controller.compute_buffer_capacity(C_buffer_list, C_H)
        dC_H_dt_raw = dC_H_dt_raw / (1.0 + _beta_total)
        dC_H_dt = 0.0 if abs(dC_H_dt_raw) < 1e-12 else dC_H_dt_raw
        dC_Na_dt_raw = - C_Na * D_total_chem + D_base * self.pH_controller.C_Na_base * 1000 + D_acid * self.pH_controller.C_Na_acid * 1000
        dC_Na_dt = 0.0 if abs(dC_Na_dt_raw) < 1e-12 else dC_Na_dt_raw
        dC_AA_dt = (self.C_AA_feed - C_AA) * D - C_AA * (D_base + D_acid) - Xv * self.kinetic_model.q_AA_current
        dC_Osm_bg_dt = (self.C_Osm_bg_feed - C_Osm_bg) * D - C_Osm_bg * (D_base + D_acid)
        # Buffer derivatives: perfusion dilution D, base/acid dilution
        dC_buffer_dt = self._compute_buffer_derivatives(C_buffer_list, D_medium=D, D_base=D_base + D_acid)
        return [dXv_dt, dC_Glc_dt, dC_Gln_dt, dC_Lac_dt, dC_Amm_dt, dV_dt, dC_O2_dt, dC_CT_dt, dC_H_dt, dC_Na_dt, dC_AA_dt, dC_Osm_bg_dt] + dC_buffer_dt
    
    def get_required_params(self):
        param_list = ['perfusion.C_Glc_Pfeed', 'perfusion.C_Gln_Pfeed', 'perfusion.C_AA_feed', 'perfusion.mu_shear_death', 'perfusion.eff_separator']
        if self.continuous:
            param_list += ['perfusion.B']
        param_list += self.perfusion_strategy_model.get_required_params()
        return param_list


# Factory for ODE systems
def create_ode_systems(params: SimulationParameters,operating_mode: OperatingMode, stoichiometric_model: StoichiometricModel, kinetic_model: KineticModel, gas_control_model: GasControllerModel, feed_strategy_model: Optional[FeedStrategyModel] = None, perfusion_strategy_model: Optional[PerfusionStrategyModel] = None) -> dict:
    """Create the ODE systems required by an operating mode.

    A batch system is prepended to fed-batch-derived modes when
    ``fed_batch.C_Glc_batch_trigger`` is configured. Chemostat and continuous
    perfusion always include a fed-batch startup system; concentrated fed-batch
    includes separate fed-batch and perfusion systems for event-driven
    switching.

    Parameters
    ----------
    params : SimulationParameters
        Validated simulation parameters.
    operating_mode : OperatingMode
        Requested operating mode.
    stoichiometric_model : StoichiometricModel
        Model shared by all returned ODE systems.
    kinetic_model : KineticModel
        Kinetic model shared by all returned ODE systems.
    gas_control_model : GasControllerModel
        Gas-control model shared by all returned systems.
    feed_strategy_model : FeedStrategyModel, optional
        Required for fed-batch-derived modes and chemostat startup; unused for
        batch and standalone perfusion.
    perfusion_strategy_model : PerfusionStrategyModel, optional
        Required for perfusion-derived modes.

    Returns
    -------
    dict[str, ODESystem]
        Mapping from phase names, such as ``"FEDBATCH"`` or ``"PERFUSION"``,
        to ODE-system instances.

    Raises
    ------
    ValueError
        If the supplied strategy models are incompatible with the selected
        operating mode.
    """
    # Helper: check whether a batch phase should be prepended.
    # A batch phase is prepended when C_Glc_batch_trigger is set in fed_batch params.
    _has_batch_trigger = (
        params.fed_batch is not None
        and params.fed_batch.C_Glc_batch_trigger is not None
    )
    _batch_ode = BatchODESystem(params, stoichiometric_model, kinetic_model, gas_control_model) if _has_batch_trigger else None

    if operating_mode == OperatingMode.BATCH:
        if feed_strategy_model is not None:
            raise ValueError("Feed strategy is not required for batch operation")
        if perfusion_strategy_model is not None:
            raise ValueError("Perfusion strategy is not required for batch operation")
        return {'BATCH': BatchODESystem(params, stoichiometric_model, kinetic_model, gas_control_model)}

    elif operating_mode == OperatingMode.BATCH_WITH_STARTUP_FEDBATCH:
        if feed_strategy_model is None:
            raise ValueError("FedBatch feed strategy is required for batch-with-startup-fed-batch operation")
        if perfusion_strategy_model is not None:
            raise ValueError("Perfusion strategy is not required for batch-with-startup-fed-batch operation")
        return {
            'FEDBATCH': FedBatchODESystem(params, stoichiometric_model, kinetic_model, feed_strategy_model, gas_control_model),
            'BATCH': BatchODESystem(params, stoichiometric_model, kinetic_model, gas_control_model),
        }

    elif operating_mode == OperatingMode.FEDBATCH:
        if feed_strategy_model is None:
            raise ValueError("Feed strategy is required for fed-batch operation")
        if perfusion_strategy_model is not None:
            raise ValueError("Perfusion strategy is not required for fed-batch operation")
        result = {}
        if _batch_ode is not None:
            result['BATCH'] = _batch_ode
        result['FEDBATCH'] = FedBatchODESystem(params, stoichiometric_model, kinetic_model, feed_strategy_model, gas_control_model)
        return result
        
    elif operating_mode == OperatingMode.CHEMOSTAT:
        if feed_strategy_model is None:
            raise ValueError("FedBatch feed strategy is required for chemostat operation as the operation starts with a fed-batch phase")
        if perfusion_strategy_model is not None:
            raise ValueError("Perfusion strategy is not required for chemostat operation")
        result = {}
        if _batch_ode is not None:
            result['BATCH'] = _batch_ode
        result['FEDBATCH'] = FedBatchODESystem(params, stoichiometric_model, kinetic_model, feed_strategy_model, gas_control_model)
        result['CHEMOSTAT'] = ChemostatODESystem(params, stoichiometric_model, kinetic_model, gas_control_model)
        return result
    
    elif operating_mode == OperatingMode.PERFUSION:
        if feed_strategy_model is not None:
            raise ValueError("FedBatch feed strategy is not required for perfusion operation")
        if perfusion_strategy_model is None:
            raise ValueError("Perfusion strategy is required for perfusion operation")
        return {'PERFUSION': PerfusionODESystem(params, stoichiometric_model, kinetic_model, perfusion_strategy_model, gas_control_model)}

    elif operating_mode == OperatingMode.CONCENTRATED_FEDBATCH:
        if feed_strategy_model is None:
            raise ValueError("FedBatch feed strategy is required for fed-batch-perfusion operation")
        if perfusion_strategy_model is None:
            raise ValueError("Perfusion strategy is required for fed-batch-perfusion operation")
        result = {}
        if _batch_ode is not None:
            result['BATCH'] = _batch_ode
        result['FEDBATCH'] = FedBatchODESystem(params, stoichiometric_model, kinetic_model, feed_strategy_model, gas_control_model)
        result['PERFUSION'] = PerfusionODESystem(params, stoichiometric_model, kinetic_model, perfusion_strategy_model, gas_control_model, continuous=False)
        return result
    
    elif operating_mode == OperatingMode.CONTINUOUS_PERFUSION:
        if feed_strategy_model is None:
            raise ValueError("FedBatch feed strategy is required for continuous perfusion operation as the operation starts with a fed-batch phase")
        if perfusion_strategy_model is None:
            raise ValueError("Perfusion strategy is required for continuous perfusion operation")
        result = {}
        if _batch_ode is not None:
            result['BATCH'] = _batch_ode
        result['FEDBATCH'] = FedBatchODESystem(params, stoichiometric_model, kinetic_model, feed_strategy_model, gas_control_model)
        result['CONTINUOUS_PERFUSION'] = PerfusionODESystem(params, stoichiometric_model, kinetic_model, perfusion_strategy_model, gas_control_model, continuous=True)
        return result

    else:
        raise ValueError(f"Unsupported operating mode: {operating_mode}")