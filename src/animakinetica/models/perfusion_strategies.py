"""Perfusion dilution-rate control strategies.

Each :class:`PerfusionStrategyModel` calculates the medium dilution rate
``D`` in 1/h for :class:`~animakinetica.models.odesystems.PerfusionODESystem`.
The available strategies provide constant dilution, toxicity-based control,
cell-specific perfusion-rate (CSPR) control, and glucose-setpoint control.
Use :func:`create_perfusion_strategy` to construct the implementation selected
by :class:`PerfusionStrategy`.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
import numpy as np
from animakinetica.core.parameters import SimulationParameters

__version__ = "1.0.0"

__all__ = [
    "PerfusionStrategy",
    "PerfusionStrategyModel",
    "ConstantDilutionStrategy",
    "ToxicityBasedStepStrategy",
    "ToxicityBasedProportionalStrategy",
    "CSPRStrategy",
    "GlucoseSetpointPerfusionStrategy",
    "create_perfusion_strategy",
]


class PerfusionStrategy(Enum):
    """Supported perfusion dilution-rate control strategies."""
    CONSTANT = "constant"
    TOXICITY_BASED_STEP = "toxicity_based_step"
    TOXICITY_BASED_PROPORTIONAL = "toxicity_based_proportional"
    CSPR = "cspr"
    GLUCOSE_SETPOINT = "glucose_setpoint"
    
    def __str__(self):
        """Return the human-readable strategy name.

        Returns
        -------
        str
            The enum value with underscores replaced by spaces and title cased.
        """
        return self.value.replace('_', ' ').title()

    @property
    def abbr(self):
        """Return the abbreviation used in configuration names.

        Returns
        -------
        str
            ``"C"``, ``"TBS"``, ``"TBP"``, ``"CSPR"``, or ``"GSP"``.
        """
        mode_abbr = {
            self.CONSTANT: "C",
            self.TOXICITY_BASED_STEP: "TBS",
            self.TOXICITY_BASED_PROPORTIONAL: "TBP",
            self.CSPR: "CSPR",
            self.GLUCOSE_SETPOINT: "GSP",
        }
        return mode_abbr.get(self, self.value)


class PerfusionStrategyModel(ABC):
    """Abstract interface for perfusion dilution-rate controllers.

    Subclasses cache their required values from ``params`` at construction and
    implement :meth:`calculate_dilution_rate`. The perfusion ODE system passes
    the complete state vector; all strategies in this module use its first five
    entries as ``Xv``, ``C_Glc``, ``C_Gln``, ``C_Lac``, and ``C_Amm``.
    """
    def __init__(self, params: SimulationParameters):
        """Initialize the controller from simulation parameters.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters containing the controller-specific perfusion
            settings and the reactor dilution-rate limit.
        """
        self._cache_parameters(params)
    
    @abstractmethod
    def _cache_parameters(self, params: SimulationParameters):
        """Cache controller parameters needed during ODE integration.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters from which the implementation obtains its
            required values.
        """
        pass

    @abstractmethod
    def calculate_dilution_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """
        Calculate the commanded perfusion dilution rate.

        Parameters
        ----------
        t : float
            Current simulation time in h.
        y : list[float]
            Current ODE state vector. Its leading entries are ``Xv`` (cells/L),
            ``C_Glc``, ``C_Gln``, ``C_Lac``, and ``C_Amm`` (mmol/L).
        q_Glc : float, optional
            Cell-specific glucose uptake rate in mmol/(cell h). Required only
            by :class:`GlucoseSetpointPerfusionStrategy`.
        D_base : float, optional
            Base and acid addition dilution rate in 1/h. Required only by
            :class:`GlucoseSetpointPerfusionStrategy`.
        C_Glc_feed : float, optional
            Glucose concentration in the perfusion feed in mmol/L. Required
            only by :class:`GlucoseSetpointPerfusionStrategy`.

        Returns
        -------
        float
            Commanded medium dilution rate in 1/h.
        """
        pass
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """Return the parameter paths required by this controller.

        Returns
        -------
        list[str]
            Dot-separated paths in :class:`SimulationParameters`.
        """
        pass


class ConstantDilutionStrategy(PerfusionStrategyModel):
    """Apply a constant baseline dilution rate, limited by ``reactor.D_max``."""
    def _cache_parameters(self, params: SimulationParameters):
        """Cache the baseline and maximum dilution rates.

        Parameters
        ----------
        params : SimulationParameters
            Parameters containing ``perfusion.D_baseline`` and
            ``reactor.D_max``.
        """
        assert params.perfusion is not None, "Perfusion parameters required for PerfusionODESystem"
        self.D_baseline = params.perfusion.D_baseline
        self.D_max = float(params.reactor.D_max)

    
    def calculate_dilution_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Return the baseline dilution rate.

        Parameters
        ----------
        t, y, q_Glc, D_base, C_Glc_feed
            Accepted to implement :meth:`PerfusionStrategyModel.calculate_dilution_rate`;
            unused by this strategy.

        Returns
        -------
        float
            ``perfusion.D_baseline`` in 1/h, capped at ``reactor.D_max``.
        """
        D = float(self.D_baseline)
        return min(D, self.D_max)
    
    def get_required_params(self) -> List[str]:
        """Return the parameter paths used by the constant controller.

        Returns
        -------
        list[str]
            ``perfusion.D_baseline`` and ``reactor.D_max``.
        """
        return ['perfusion.D_baseline', 'reactor.D_max']


class ToxicityBasedStepStrategy(PerfusionStrategyModel):
    """
    Dilution rate based on lactate and ammonia concentrations — ratchet variant.

    The dilution level is a one-way latch: it can step up when a threshold is
    crossed but never steps back down, even if metabolite concentrations fall
    below the trigger again.  This eliminates chattering (rapid switching near
    a threshold) which causes LSODA to take extremely small steps.

    Levels (0 → 1 → 2, never reversed) are selected from ``D_baseline``,
    ``D_1``, and ``D_2``. Level 1 activates when lactate or ammonia reaches
    25% of its respective ``IC100``; level 2 activates at 50%.
    """
    def _cache_parameters(self, params: SimulationParameters):
        """Cache step levels, toxicity thresholds, and the dilution limit.

        Parameters
        ----------
        params : SimulationParameters
            Parameters containing perfusion step rates, reactor limit, and
            lactate and ammonia ``IC100`` values.
        """
        assert params.perfusion is not None, "Perfusion parameters required for PerfusionODESystem"
        self.D_baseline = params.perfusion.D_baseline
        self.D_1 = params.perfusion.D_1
        self.D_2 = params.perfusion.D_2
        self.D_max = float(params.reactor.D_max)
        self.C_Lac_threshold_1 = params.kinetics.IC100_Lac * 0.25
        self.C_Amm_threshold_1 = params.kinetics.IC100_Amm * 0.25
        self.C_Lac_threshold_2 = params.kinetics.IC100_Lac * 0.5
        self.C_Amm_threshold_2 = params.kinetics.IC100_Amm * 0.5
        # Ratchet state: 0 = D_baseline, 1 = D_1, 2 = D_2. Never decremented.
        self._current_level: int = 0

    def calculate_dilution_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """
        Calculate the ratcheted dilution rate from lactate and ammonia.

        Parameters
        ----------
        t, q_Glc, D_base, C_Glc_feed
            Accepted to implement :meth:`PerfusionStrategyModel.calculate_dilution_rate`;
            unused by this strategy.
        y : list[float]
            ODE state vector, with lactate at index 3 and ammonia at index 4
            in mmol/L.

        Returns
        -------
        float
            The current latched dilution level in 1/h, capped at
            ``reactor.D_max``.
        """
        C_Lac = y[3]
        C_Amm = y[4]

        # Advance the ratchet if a higher threshold is now exceeded.
        # Checked from highest to lowest so a single call can jump straight to level 2.
        if self._current_level < 2:
            if C_Lac >= self.C_Lac_threshold_2 or C_Amm >= self.C_Amm_threshold_2:
                self._current_level = 2
            elif C_Lac >= self.C_Lac_threshold_1 or C_Amm >= self.C_Amm_threshold_1:
                self._current_level = max(self._current_level, 1)

        if self._current_level == 2:
            D = float(self.D_2)
        elif self._current_level == 1:
            D = float(self.D_1)
        else:
            D = float(self.D_baseline)

        return min(D, self.D_max)

    def get_required_params(self) -> List[str]:
        """Return the parameter paths used by the toxicity step controller.

        Returns
        -------
        list[str]
            Baseline and step-rate multiplier paths, the reactor limit, and
            lactate and ammonia ``IC100`` paths.
        """
        return [
            'perfusion.D_baseline',
            'perfusion.D_1_multiplier',
            'perfusion.D_2_multiplier',
            'reactor.D_max',
            'kinetics.IC100_Lac',
            'kinetics.IC100_Amm',
        ]


class ToxicityBasedProportionalStrategy(PerfusionStrategyModel):
    """
    Dilution rate with proportional response to waste metabolite concentrations.
    
    The controller calculates
    ``D = D_baseline + k_Lac * C_Lac / IC100_Lac + k_Amm * C_Amm / IC100_Amm``
    and caps the result at ``reactor.D_max``.
    """

    def _cache_parameters(self, params: SimulationParameters):
        """Cache toxicity gains, inhibition concentrations, and rate limits.

        Parameters
        ----------
        params : SimulationParameters
            Parameters containing the proportional gains, baseline rate,
            reactor limit, and metabolite ``IC100`` values.
        """
        assert params.perfusion is not None, "Perfusion parameters required for PerfusionODESystem"

        self.D_baseline = params.perfusion.D_baseline
        self.D_max = float(params.reactor.D_max)
        self.k_Lac = params.perfusion.k_Lac  # Proportional gain for lactate
        self.k_Amm = params.perfusion.k_Amm  # Proportional gain for ammonia
        self.IC100_Lac = params.kinetics.IC100_Lac
        self.IC100_Amm = params.kinetics.IC100_Amm

    def calculate_dilution_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """
        Calculate the toxicity-proportional dilution rate.

        Parameters
        ----------
        t, q_Glc, D_base, C_Glc_feed
            Accepted to implement :meth:`PerfusionStrategyModel.calculate_dilution_rate`;
            unused by this strategy.
        y : list[float]
            ODE state vector, with lactate at index 3 and ammonia at index 4
            in mmol/L.

        Returns
        -------
        float
            Dilution rate in 1/h, capped at ``reactor.D_max``.
        """
        # Unpack lactate and ammonia concentrations
        C_Lac = y[3]
        C_Amm = y[4]
        
        # Calculate normalized toxicity levels
        Lac_norm = C_Lac / self.IC100_Lac
        Amm_norm = C_Amm / self.IC100_Amm
        
        # Calculate dilution rate with proportional control
        D = self.D_baseline + self.k_Lac * Lac_norm + self.k_Amm * Amm_norm
        return min(float(D), self.D_max)
    
    def get_required_params(self) -> List[str]:
        """Return the parameter paths used by the proportional controller.

        Returns
        -------
        list[str]
            Baseline rate, toxicity gains, reactor limit, and metabolite
            ``IC100`` paths.
        """
        return [
            'perfusion.D_baseline',
            'perfusion.k_Lac',
            'perfusion.k_Amm',
            'reactor.D_max',
            'kinetics.IC100_Lac',
            'kinetics.IC100_Amm',
        ]


class CSPRStrategy(PerfusionStrategyModel):
    """
    Dilution rate based on cell-specific-perfusion rate (CSPR).
    
    Adjusts dilution rate proportionally to viable cell density:
    D = CSPR * Xv
    
    Here ``CSPR`` is the configured cell-specific perfusion rate and ``Xv`` is
    viable cell density. The result is capped at ``reactor.D_max``.
    """
    def _cache_parameters(self, params: SimulationParameters):
        """Cache the cell-specific perfusion rate and dilution limit.

        Parameters
        ----------
        params : SimulationParameters
            Parameters containing ``perfusion.CSPR`` and ``reactor.D_max``.
        """
        assert params.perfusion is not None, "Perfusion parameters required for PerfusionODESystem"

        self.CSPR = params.perfusion.CSPR
        self.D_max = float(params.reactor.D_max)


    def calculate_dilution_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """
        Calculate the dilution rate from viable cell density.

        Parameters
        ----------
        t, q_Glc, D_base, C_Glc_feed
            Accepted to implement :meth:`PerfusionStrategyModel.calculate_dilution_rate`;
            unused by this strategy.
        y : list[float]
            ODE state vector with viable cell density ``Xv`` at index 0 in
            cells/L.

        Returns
        -------
        float
            ``perfusion.CSPR * Xv`` in 1/h, capped at ``reactor.D_max``.
        """
        
        D = self.CSPR * y[0]  # y[0] is Xv
        return min(float(D), self.D_max)
    
    def get_required_params(self) -> List[str]:
        """Return the parameter paths used by the CSPR controller.

        Returns
        -------
        list[str]
            ``perfusion.CSPR`` and ``reactor.D_max``.
        """
        return ['perfusion.CSPR', 'reactor.D_max']


class GlucoseSetpointPerfusionStrategy(PerfusionStrategyModel):
    """
    Glucose setpoint perfusion strategy.

    Uses algebraic dilution backcalculation:
        D = (Xv * q_Glc + C_Glc_setpoint * D_base) / (C_Glc_feed - C_Glc_setpoint)
    with optional proportional correction, deadband, denominator guarding,
    and optional slew-rate limiting. This strategy requires ``q_Glc``,
    ``D_base``, and ``C_Glc_feed`` from
    :class:`~animakinetica.models.odesystems.PerfusionODESystem`.
    """

    def _cache_parameters(self, params: SimulationParameters):
        """Cache glucose-control settings and initialize limiter state.

        Parameters
        ----------
        params : SimulationParameters
            Parameters containing the glucose setpoint and optional controller
            tuning values under ``perfusion`` plus ``reactor.D_max``.
        """
        assert params.perfusion is not None, "Perfusion parameters required for GlucoseSetpointPerfusionStrategy"
        assert params.perfusion.C_Glc_setpoint is not None, "perfusion.C_Glc_setpoint required for GlucoseSetpointPerfusionStrategy"

        self.C_Glc_setpoint = float(params.perfusion.C_Glc_setpoint)
        self.D_max = float(params.reactor.D_max)
        self.Kp = float(params.perfusion.C_Glc_setpoint_Kp) if params.perfusion.C_Glc_setpoint_Kp is not None else 0.0
        self.deadband = float(params.perfusion.C_Glc_setpoint_deadband) if params.perfusion.C_Glc_setpoint_deadband is not None else 0.0
        self.slew_rate = float(params.perfusion.C_Glc_setpoint_slew_rate) if params.perfusion.C_Glc_setpoint_slew_rate is not None else 0.0
        self.den_epsilon = float(params.perfusion.C_Glc_setpoint_den_epsilon) if params.perfusion.C_Glc_setpoint_den_epsilon is not None else 1e-12

        # Runtime state for optional slew-rate limiting on D.
        self._last_t: Optional[float] = None
        self._last_D: Optional[float] = None

    def calculate_dilution_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Calculate an algebraic dilution rate for glucose-setpoint control.

        Parameters
        ----------
        t : float
            Current simulation time in h; used by the optional slew-rate
            limiter.
        y : list[float]
            ODE state vector with ``Xv`` at index 0 in cells/L and glucose at
            index 1 in mmol/L.
        q_Glc : float
            Cell-specific glucose uptake rate in mmol/(cell h).
        D_base : float
            Combined base and acid addition dilution rate in 1/h.
        C_Glc_feed : float
            Perfusion-feed glucose concentration in mmol/L.

        Returns
        -------
        float
            Non-negative commanded dilution rate in 1/h, capped at
            ``reactor.D_max``.

        Raises
        ------
        ValueError
            If ``q_Glc``, ``D_base``, or ``C_Glc_feed`` is not supplied.
        """
        Xv = float(y[0])
        C_Glc = float(y[1])
        if q_Glc is None or D_base is None or C_Glc_feed is None:
            raise ValueError(
                "GlucoseSetpointPerfusionStrategy requires q_Glc, D_base, and C_Glc_feed "
                "for algebraic control."
            )

        denominator = float(C_Glc_feed) - self.C_Glc_setpoint
        if abs(denominator) < self.den_epsilon:
            D_cmd = 0.0
            self._last_t = float(t)
            self._last_D = D_cmd
            return D_cmd

        D_alg = (Xv * float(q_Glc) + self.C_Glc_setpoint * float(D_base)) / denominator
        if not np.isfinite(D_alg):
            return 0.0

        # Optional proportional correction for faster convergence from offsets.
        error = self.C_Glc_setpoint - C_Glc
        if abs(error) <= self.deadband:
            error = 0.0

        e_norm = error / denominator
        D_cmd = float(D_alg + self.Kp * e_norm)
        D_cmd = max(0.0, min(D_cmd, self.D_max))

        # Optional slew-rate limiter in 1/h^2 (change in D per hour).
        if self.slew_rate > 0.0 and self._last_t is not None and self._last_D is not None:
            dt = max(float(t) - self._last_t, 0.0)
            max_delta = self.slew_rate * dt
            if D_cmd > self._last_D + max_delta:
                D_cmd = self._last_D + max_delta
            elif D_cmd < self._last_D - max_delta:
                D_cmd = self._last_D - max_delta
            D_cmd = max(0.0, min(D_cmd, self.D_max))

        self._last_t = float(t)
        self._last_D = D_cmd
        return D_cmd

    def get_required_params(self) -> List[str]:
        """Return the mandatory parameter paths for glucose-setpoint control.

        Returns
        -------
        list[str]
            ``perfusion.C_Glc_setpoint`` and ``reactor.D_max``. Optional
            tuning parameters use internal defaults when omitted.
        """
        return [
            'perfusion.C_Glc_setpoint',
            'reactor.D_max'
        ]


def create_perfusion_strategy(strategy_type: PerfusionStrategy, params: SimulationParameters) -> PerfusionStrategyModel:
    """Create the controller implementation selected by a perfusion strategy.

    Parameters
    ----------
    strategy_type : PerfusionStrategy
        Strategy enum selecting the controller class.
    params : SimulationParameters
        Simulation parameters used to initialize the selected controller.

    Returns
    -------
    PerfusionStrategyModel
        Initialized controller matching ``strategy_type``.

    Raises
    ------
    ValueError
        If ``strategy_type`` is not a supported :class:`PerfusionStrategy`.
    """
    strategies = {
        PerfusionStrategy.CONSTANT: ConstantDilutionStrategy,
        PerfusionStrategy.TOXICITY_BASED_STEP: ToxicityBasedStepStrategy,
        PerfusionStrategy.TOXICITY_BASED_PROPORTIONAL: ToxicityBasedProportionalStrategy,
        PerfusionStrategy.CSPR: CSPRStrategy,
        PerfusionStrategy.GLUCOSE_SETPOINT: GlucoseSetpointPerfusionStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown perfusion strategy: {strategy_type}")
    
    return strategies[strategy_type](params)