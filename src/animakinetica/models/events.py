"""Terminal and phase-transition events used by ODE integrations.

An event returns a signed distance to its threshold. For all exported event
functions, a zero crossing terminates the current ``scipy.integrate.solve_ivp``
phase (``terminal = True``). The ``direction`` attribute restricts detection to
the physically relevant crossing: ``-1`` for a falling value and ``1`` for a
rising value. A terminal event can therefore represent either the end of a
cultivation or a controlled switch to the next ODE phase; the simulation engine
uses the phase schedule to determine what follows.

State-vector contract
---------------------
All events accept ``(t, y, params)``. ``t`` is time in hours, and the first
twelve entries of ``y`` are:

``[Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA, C_Osm_bg]``

where cell density is in cells/L; concentrations are in mmol/L except ``C_H``
in mol/L; ``V`` is in L; and ``C_Osm_bg`` is the background osmolality in mOsm.
Optional non-carbonate buffer states may follow these entries and are not used
by this module.

Event groups
------------
* **General stop conditions:** nutrient or oxygen depletion, pH excursion,
    minimum viable-cell density, viscosity limit, maximum volume, and ammonia or
    lactate IC100 inhibition ceilings.
* **Phase transitions:** glucose falling to the configured fed-batch trigger,
    cell density rising to begin chemostat, and cell density/lactate/ammonia
    rising to begin perfusion.
* **Concentrated fed-batch harvest:** a pickle-safe callable stops a perfusion
    phase at ``perfusion.Xv_harvest_stop`` after an optional minimum phase runtime.

``CellCultureSimulation`` selects the applicable events for each ODE phase.
Direct ``solve_ivp`` callers must pass the parameter object to both the right
hand side and event functions, for example ``args=(params,)``.
"""
from animakinetica.core.parameters import SimulationParameters
import numpy as np

__version__ = "1.0.0"
__author__ = "Ben Tumulero"

__all__ = [
    "glucose_depletion_event",
    "glutamine_depletion_event",
    "AA_depletion_event",
    "oxygen_depletion_event",
    "pH_deviation_event",
    "viscosity_limit_event",
    "max_volume_event",
    "celldensity_chemostat_trigger_event",
    "celldensity_pf_trigger_event",
    "lactate_pf_trigger_event",
    "ammonia_pf_trigger_event",
    "PerfusionHarvestStopEvent",
    "make_perfusion_harvest_stop_event",
    "cell_minimum_event",
    "glucose_batch_trigger_event",
    "ammonia_IC100_event",
    "lactate_IC100_event",
]

# --- General stop conditions ---
def glucose_depletion_event(t, y, params: SimulationParameters):
    """Stop when glucose falls to 0.001 mmol/L.

    The event value is ``C_Glc - 0.001`` and is detected on a decreasing
    crossing. ``params`` is accepted to satisfy the common ``solve_ivp`` event
    signature but is not used.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector.
    params : SimulationParameters
        Simulation parameters; accepted but not used.
    """
    threshold = 0.001  # mmol/L
    return y[1] - threshold  # y[1] is C_Glc

glucose_depletion_event.terminal = True
glucose_depletion_event.direction = -1

def glutamine_depletion_event(t, y, params: SimulationParameters):
    """Stop when glutamine falls to 0.001 mmol/L.

    The event value is ``C_Gln - 0.001`` and is detected on a decreasing
    crossing. ``params`` is accepted but not used.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector.
    params : SimulationParameters
        Simulation parameters; accepted but not used.
    """
    threshold = 0.001  # mmol/L
    return y[2] - threshold  # y[2] is C_Gln

glutamine_depletion_event.terminal = True
glutamine_depletion_event.direction = -1


def AA_depletion_event(t, y, params: SimulationParameters):
    """Stop when total amino acids excluding glutamine fall to 0.001 mmol/L.

    The event value is ``C_AA - 0.001`` and is detected on a decreasing
    crossing. ``params`` is accepted but not used.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector.
    params : SimulationParameters
        Simulation parameters; accepted but not used.
    """
    threshold = 0.001  # mmol/L
    return y[10] - threshold  # y[10] is C_AA

AA_depletion_event.terminal = True
AA_depletion_event.direction = -1


def oxygen_depletion_event(t, y, params: SimulationParameters):
    """Stop when dissolved oxygen falls to ``1e-5`` mmol/L.

    The event value is ``C_O2 - 1e-5`` and is detected on a decreasing
    crossing. ``params`` is accepted but not used.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector.
    params : SimulationParameters
        Simulation parameters; accepted but not used.
    """
    threshold = 1e-5  # mmol/L
    return y[6] - threshold  # y[6] is C_O2


oxygen_depletion_event.terminal = True
oxygen_depletion_event.direction = -1


def pH_deviation_event(t, y, params: SimulationParameters):
    """Stop when hydrogen-ion concentration leaves the configured pH band.

    ``ReactorParameters`` derives ``C_H_lower`` and ``C_H_upper`` from
    ``pH_setpoint ± pH_tolerance``. The returned minimum distance is positive
    within the allowed concentration interval and crosses zero as pH becomes
    either too high or too low. The event requires ``params.reactor``.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``C_H`` at index 8 [mol/L].
    params : SimulationParameters
        Simulation parameters containing reactor pH limits.
    """
    C_H = y[8]  # y[8] is C_H
    C_H_lower = float(params.reactor.C_H_lower)
    C_H_upper = float(params.reactor.C_H_upper)

    # Positive within the allowed interval; negative outside it.
    distance_to_lower = C_H - C_H_lower
    distance_to_upper = C_H_upper - C_H
    
    return min(distance_to_lower, distance_to_upper)



pH_deviation_event.terminal = True
pH_deviation_event.direction = -1

def cell_minimum_event(t, y, params: SimulationParameters):
    """Stop when viable-cell density falls to ``cell_properties.Xv_min``.

    The event value is ``Xv - Xv_min`` and is detected on a decreasing
    crossing. The event requires ``params.cell_properties``.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``Xv`` at index 0 [cells/L].
    params : SimulationParameters
        Simulation parameters containing the minimum viable-cell density.
    """
    return y[0] - params.cell_properties.Xv_min  # y[0] is Xv

cell_minimum_event.terminal = True
cell_minimum_event.direction = -1

def viscosity_limit_event(t, y, params: SimulationParameters):
    """Stop when viable-cell density reaches the viscosity limit.

    The event value is ``Xv - X_viscosity_limit`` and is detected on a rising
    crossing. ``params.cell_properties.X_viscosity_limit`` is in cells/L. A
    reactor section must also be present because all simulations using this
    event require reactor parameters.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``Xv`` at index 0 [cells/L].
    params : SimulationParameters
        Simulation parameters containing the viscosity-limit cell density.
    """
    if params.reactor is None:
        raise AttributeError("SimulationParameters object has no reactor parameters")
    if params.cell_properties is None:
        raise AttributeError("SimulationParameters object has no cell properties")

    return y[0] - params.cell_properties.X_viscosity_limit  # y[0] is Xv

viscosity_limit_event.terminal = True
viscosity_limit_event.direction = 1


def max_volume_event(t, y, params: SimulationParameters):
    """Stop when culture volume reaches the maximum reactor volume.

    The event value is ``V - V_max - 1e-4 L`` and is detected on a rising
    crossing. Consequently, the phase stops at a volume 0.0001 L above the
    configured ``params.reactor.V_max``. This preserves the existing numerical
    convention; it is not an early safety margin.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``V`` at index 5 [L].
    params : SimulationParameters
        Simulation parameters containing the maximum culture volume.
    """
    if params.reactor is None:
        raise AttributeError("SimulationParameters object has no reactor parameters")
    
    return y[5] - params.reactor.V_max - 0.0001  # y[5] is V

max_volume_event.terminal = True
max_volume_event.direction = 1

# --- Chemostat transition condition ---
def celldensity_chemostat_trigger_event(t, y, params: SimulationParameters):
    """End fed-batch at ``chemostat.Xv_chemostat_trig`` to begin chemostat.

    The event value is ``Xv - Xv_chemostat_trig`` and is detected on a rising
    crossing. The simulation engine attaches it only to the preceding fed-batch
    phase when the optional trigger is configured.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``Xv`` at index 0 [cells/L].
    params : SimulationParameters
        Simulation parameters containing the chemostat trigger.
    """
    if params.chemostat is None:
        raise AttributeError("SimulationParameters object has no chemostat parameters")
    
    return y[0] - params.chemostat.Xv_chemostat_trig  # y[0] is Xv

celldensity_chemostat_trigger_event.terminal = True
celldensity_chemostat_trigger_event.direction = 1


# --- Perfusion transition conditions ---
def celldensity_pf_trigger_event(t, y, params: SimulationParameters):
    """End fed-batch at ``perfusion.Xv_pf_trig`` to begin perfusion.

    The event value is ``Xv - Xv_pf_trig`` and is detected on a rising
    crossing. The simulation engine enables this optional trigger only for
    fed-batch phases that precede a perfusion phase.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``Xv`` at index 0 [cells/L].
    params : SimulationParameters
        Simulation parameters containing the perfusion cell-density trigger.
    """
    if params.perfusion is None:
        raise AttributeError("SimulationParameters object has no perfusion parameters")

    return y[0] - params.perfusion.Xv_pf_trig  # y[0] is Xv

celldensity_pf_trigger_event.terminal = True
celldensity_pf_trigger_event.direction = 1

def lactate_pf_trigger_event(t, y, params: SimulationParameters):
    """End fed-batch at ``perfusion.C_Lac_pf_trig`` to begin perfusion.

    The event value is ``C_Lac - C_Lac_pf_trig`` and is detected on a rising
    crossing. This is a phase-transition threshold, distinct from the kinetic
    lactate IC100 stop condition.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``C_Lac`` at index 3 [mmol/L].
    params : SimulationParameters
        Simulation parameters containing the perfusion lactate trigger.
    """
    if params.perfusion is None:
        raise AttributeError("SimulationParameters object has no perfusion parameters")

    return y[3] - params.perfusion.C_Lac_pf_trig  # y[3] is C_Lac

lactate_pf_trigger_event.terminal = True
lactate_pf_trigger_event.direction = 1


def ammonia_pf_trigger_event(t, y, params: SimulationParameters):
    """End fed-batch at ``perfusion.C_Amm_pf_trig`` to begin perfusion.

    The event value is ``C_Amm - C_Amm_pf_trig`` and is detected on a rising
    crossing. This is a phase-transition threshold, distinct from the kinetic
    ammonia IC100 stop condition.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``C_Amm`` at index 4 [mmol/L].
    params : SimulationParameters
        Simulation parameters containing the perfusion ammonia trigger.
    """
    if params.perfusion is None:
        raise AttributeError("SimulationParameters object has no perfusion parameters")
    
    return y[4] - params.perfusion.C_Amm_pf_trig  # y[4] is C_Amm

ammonia_pf_trigger_event.terminal = True
ammonia_pf_trigger_event.direction = 1


class PerfusionHarvestStopEvent:
    """Pickle-safe perfusion harvest event for concentrated fed-batch runs.

    The event remains negative until phase-local ``t`` reaches
    ``min_perfusion_runtime_h``. It then returns
    ``Xv - perfusion.Xv_harvest_stop`` and detects its rising zero crossing.
    Defining the callable at module scope, instead of in a closure, permits
    serialization by Windows multiprocessing workers using the spawn method.

    Parameters
    ----------
    min_perfusion_runtime_h : float, default=0.0
        Minimum duration of the current perfusion phase in hours before the
        harvest threshold can terminate it.
    """

    def __init__(self, min_perfusion_runtime_h: float = 0.0):
        self.min_perfusion_runtime_h = float(min_perfusion_runtime_h)
        # solve_ivp reads these attributes from the callable.
        self.terminal = True
        self.direction = 1

    def __call__(self, t, y, params: SimulationParameters):
        """Return the phase-local harvest-stop event value.

        Requires ``params.perfusion.Xv_harvest_stop``. Returning ``-1.0``
        before the minimum runtime keeps the terminal event inactive even when
        the carried-over cell density already exceeds the harvest threshold.

        Parameters
        ----------
        t : float
            Phase-local perfusion time [h].
        y : array-like
            Current state vector containing ``Xv`` at index 0 [cells/L].
        params : SimulationParameters
            Simulation parameters containing the perfusion harvest threshold.

        Returns
        -------
        float
            Negative sentinel before the minimum runtime, otherwise
            ``Xv - Xv_harvest_stop``.

        Raises
        ------
        AttributeError
            If the perfusion section or harvest-stop threshold is unavailable.
        """
        if params.perfusion is None:
            raise AttributeError("SimulationParameters object has no perfusion parameters")
        if params.perfusion.Xv_harvest_stop is None:
            raise AttributeError("SimulationParameters.perfusion has no Xv_harvest_stop parameter")

        if t < self.min_perfusion_runtime_h:
            return -1.0

        return y[0] - params.perfusion.Xv_harvest_stop  # y[0] is Xv


def make_perfusion_harvest_stop_event(min_perfusion_runtime_h: float = 0.0):
    """Create a serializable terminal event for the perfusion harvest threshold.

    Parameters
    ----------
    min_perfusion_runtime_h : float, default=0.0
        Minimum phase-local perfusion duration before the returned event can
        cross zero.

    Returns
    -------
    PerfusionHarvestStopEvent
        A callable configured with ``terminal=True`` and ``direction=1``.

    Notes
    -----
    This avoids immediate termination after a fed-batch-to-perfusion switch
    when the inherited cell density already satisfies ``Xv_harvest_stop``.
    """
    return PerfusionHarvestStopEvent(min_perfusion_runtime_h=min_perfusion_runtime_h)


# --- Batch-to-fed-batch transition condition ---
def glucose_batch_trigger_event(t, y, params: SimulationParameters):
    """End batch at ``fed_batch.C_Glc_batch_trigger`` to begin fed-batch.

    The event value is ``C_Glc - C_Glc_batch_trigger`` and is detected on a
    decreasing crossing. If the optional fed-batch section or trigger is absent,
    it returns a positive constant and cannot fire. The simulation engine only
    appends this event when a batch phase is immediately followed by fed-batch.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``C_Glc`` at index 1 [mmol/L].
    params : SimulationParameters
        Simulation parameters that may contain the fed-batch trigger.
    """
    if params.fed_batch is None or params.fed_batch.C_Glc_batch_trigger is None:
        return 1.0  # Never triggers if parameter is not set

    return y[1] - float(params.fed_batch.C_Glc_batch_trigger)  # y[1] is C_Glc

glucose_batch_trigger_event.terminal = True
glucose_batch_trigger_event.direction = -1


# --- Inhibition-ceiling stop conditions ---


def ammonia_IC100_event(t, y, params: SimulationParameters):
    """Stop when ammonia reaches ``kinetics.IC100_Amm``.

    The returned value, ``IC100_Amm - C_Amm``, is positive below the ceiling
    and crosses zero on a decreasing crossing. When no IC100 is configured,
    the event returns ``1e9`` and never fires.

    This prevents BDF and LSODA from stalling at the non-smooth power-decay
    inhibition ceiling, where ``max(1 - C / IC100, 0) ** phi`` has an infinite
    derivative for ``phi < 1``. At IC100, this model represents complete growth
    inhibition.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``C_Amm`` at index 4 [mmol/L].
    params : SimulationParameters
        Simulation parameters that may contain ``kinetics.IC100_Amm``.
    """
    if params.kinetics is None or params.kinetics.IC100_Amm is None:
        return 1e9  # never triggers when IC100 is not defined
    return float(params.kinetics.IC100_Amm) - y[4]  # y[4] is C_Amm

ammonia_IC100_event.terminal = True
ammonia_IC100_event.direction = -1


def lactate_IC100_event(t, y, params: SimulationParameters):
    """Stop when lactate reaches ``kinetics.IC100_Lac``.

    The returned value, ``IC100_Lac - C_Lac``, is positive below the ceiling
    and crosses zero on a decreasing crossing. When no IC100 is configured,
    the event returns ``1e9`` and never fires.

    This prevents BDF and LSODA from stalling at the non-smooth power-decay
    inhibition ceiling, where ``max(1 - C / IC100, 0) ** phi`` has an infinite
    derivative for ``phi < 1``. At IC100, this model represents complete growth
    inhibition.

    Parameters
    ----------
    t : float
        Phase-local time [h]; accepted but not used.
    y : array-like
        Current state vector containing ``C_Lac`` at index 3 [mmol/L].
    params : SimulationParameters
        Simulation parameters that may contain ``kinetics.IC100_Lac``.
    """
    if params.kinetics is None or params.kinetics.IC100_Lac is None:
        return 1e9  # never triggers when IC100 is not defined
    return float(params.kinetics.IC100_Lac) - y[3]  # y[3] is C_Lac

lactate_IC100_event.terminal = True
lactate_IC100_event.direction = -1