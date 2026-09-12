"""Liquid-feed strategies for fed-batch ODE systems.

Each strategy calculates the medium feed flow rate ``F_L`` in L/h and caps its
upper value at ``reactor.F_L_max``. The fed-batch ODE system supplies the current phase
time and state vector and converts the returned flow to medium dilution as
``D = F_L / V``. Base and acid additions are controlled independently and are
not included in ``F_L``.

The state vector uses the shared simulation ordering. Feed strategies use only
the following entries: ``y[0]`` (``Xv``, viable-cell density in cells/L),
``y[1]`` (``C_Glc``, glucose concentration in mmol/L), and ``y[5]`` (``V``,
culture volume in L).

The constant, exponential, linear, and cell-specific feed-rate (CSFR)
strategies ignore the optional glucose-control inputs accepted by the common
interface. ``GlucoseSetpointFeedStrategy`` requires them to calculate an
algebraic flow rate that offsets glucose uptake and base-addition dilution.
Use :func:`create_feed_model` to instantiate a strategy selected through
:class:`FeedStrategy`.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
import numpy as np
from animakinetica.core.parameters import SimulationParameters

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = [
    "FeedStrategy",
    "FeedStrategyModel",
    "ConstantFeedStrategy",
    "ExponentialFeedStrategy",
    "LinearFeedStrategy",
    "CSFRStrategy",
    "GlucoseSetpointFeedStrategy",
    "create_feed_model",
]

class FeedStrategy(Enum):
    """Supported fed-batch liquid-feed control laws.

    ``CONSTANT``, ``EXPONENTIAL``, and ``LINEAR`` use time-based feed profiles.
    ``CSFR`` scales feed with viable-cell density and culture volume.
    ``GLUCOSE_SETPOINT`` calculates a feed rate from a glucose mass balance.
    """
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CSFR = "CSFR"
    GLUCOSE_SETPOINT = "glucose_setpoint"
    
    def __str__(self):
        """Return a title-cased label derived from the enum value.

        Returns
        -------
        str
            Human-readable strategy label.
        """
        return self.value.replace('_', ' ').title()
    
    @property
    def abbr(self):
        """Return the stable abbreviation used in configuration filenames.

        Returns
        -------
        str
            ``C``, ``E``, ``L``, ``CSFR``, or ``GSP`` for the corresponding
            strategy.
        """
        strategy_abbr = {
            self.CONSTANT: "C",
            self.EXPONENTIAL: "E",
            self.LINEAR: "L",
            self.CSFR: "CSFR",
            self.GLUCOSE_SETPOINT: "GSP"
        }
        return strategy_abbr.get(self, self.value)

class FeedStrategyModel(ABC):
    """Abstract interface for fed-batch feed-rate strategies.

    Implementations cache their required values during construction because
    simulation parameters are frozen after simulation initialization. All
    subclasses must return a feed flow in L/h from :meth:`calculate_feed_rate`
    and identify their parameter dependencies with :meth:`get_required_params`.
    """
    
    def __init__(self, params: SimulationParameters):
        """Initialize the model and cache values from simulation parameters.

        Parameters
        ----------
        params : SimulationParameters
            Validated simulation parameters containing the fed-batch and
            reactor values required by the concrete strategy.
        """
        self._cache_parameters(params)

    @abstractmethod
    def _cache_parameters(self, params: SimulationParameters):
        """Cache concrete-strategy parameters from ``params``.

        Parameters
        ----------
        params : SimulationParameters
            Parameter object supplied during strategy construction.
        """
        pass

    @abstractmethod
    def calculate_feed_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Calculate the commanded liquid-feed flow rate.

        Parameters
        ----------
        t : float
            Current ODE-phase time in h.
        y : list[float]
            Current state vector. All strategies may receive the full vector;
            concrete strategies use only the entries documented at module level.
        q_Glc : float, optional
            Specific glucose uptake rate in mmol/(cell·h). Required only by
            :class:`GlucoseSetpointFeedStrategy`.
        D_base : float, optional
            Current pH-controller base-addition dilution rate in 1/h. Required
            only by :class:`GlucoseSetpointFeedStrategy`.
        C_Glc_feed : float, optional
            Glucose concentration in the feed medium in mmol/L. Required only
            by :class:`GlucoseSetpointFeedStrategy`.

        Returns
        -------
        float
            Commanded feed flow ``F_L`` in L/h. Concrete strategies cap its
            upper value at ``reactor.F_L_max``.
        """
        pass
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """Return parameter paths required by this strategy.

        Returns
        -------
        list[str]
            Parameter paths used by ODE-system requirement reporting.
        """
        pass

class ConstantFeedStrategy(FeedStrategyModel):
    """Feed at the configured initial flow rate throughout the phase.

    The commanded flow is ``min(F_L0, reactor.F_L_max)``, where ``F_L0`` is
    derived from ``fed_batch.F_L0_specific`` and the configured initial volume.
    """
    
    def _cache_parameters(self, params: SimulationParameters):
        """Cache ``fed_batch.F_L0`` and the reactor feed-flow limit."""
        self.fed_batch_F_L0 = float(params.fed_batch.F_L0)
        self.F_L_max = float(params.reactor.F_L_max)

    def calculate_feed_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Return the capped constant feed flow.

        Parameters
        ----------
        t : float
            Current phase time in h; accepted but not used.
        y : list[float]
            Current state vector; accepted but not used.
        q_Glc, D_base, C_Glc_feed : float, optional
            Glucose-control inputs; accepted but not used.

        Returns
        -------
        float
            ``min(F_L0, F_L_max)`` in L/h.
        """
        F_L = self.fed_batch_F_L0
        return min(F_L, self.F_L_max)
    
    def get_required_params(self) -> List[str]:
        """Return parameter paths used to derive the constant feed flow."""
        return ['fed_batch.F_L0_specific', 'reactor.F_L_max_specific', 'reactor.V_init']

class ExponentialFeedStrategy(FeedStrategyModel):
    """Apply a capped exponential feed profile.

    The unbounded profile is ``F_L(t) = F_L0 exp(r t)``, where ``t`` is the
    current phase time in h and ``r`` is ``fed_batch.r`` in 1/h. The returned
    flow cannot exceed ``reactor.F_L_max``.
    """
    
    def _cache_parameters(self, params: SimulationParameters):
        """Cache ``F_L0``, exponential coefficient ``r``, and flow limit.

        Raises
        ------
        AssertionError
            If ``fed_batch.r`` is not configured.
        """
        assert params.fed_batch.r is not None, "fed_batch.r required for ExponentialFeedStrategy"
        self.F_L0 = params.fed_batch.F_L0
        self.r = params.fed_batch.r
        self.F_L_max = float(params.reactor.F_L_max)

    def calculate_feed_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Return the capped exponential feed flow for the current phase time.

        Parameters
        ----------
        t : float
            Current phase time in h.
        y : list[float]
            Current state vector; accepted but not used.
        q_Glc, D_base, C_Glc_feed : float, optional
            Glucose-control inputs; accepted but not used.

        Returns
        -------
        float
            ``min(F_L0 * exp(r * t), F_L_max)`` in L/h.
        """

        F_L = float(self.F_L0 * np.exp(self.r * t))
        return min(F_L, self.F_L_max)
    
    def get_required_params(self) -> List[str]:
        """Return parameter paths used by the exponential profile."""
        return ['fed_batch.F_L0_specific', 'reactor.F_L_max_specific', 'reactor.V_init', 'fed_batch.r']

class LinearFeedStrategy(FeedStrategyModel):
    """Apply a capped linear feed ramp.

    The unbounded profile is ``F_L(t) = F_L0 + feed_slope * t``, where ``t``
    is the current phase time in h. The returned flow cannot exceed
    ``reactor.F_L_max``.
    """
    
    def _cache_parameters(self, params: SimulationParameters):
        """Cache ``F_L0``, ``feed_slope``, and the reactor flow limit.

        Raises
        ------
        AssertionError
            If ``fed_batch.feed_slope`` is not configured.
        """
        assert params.fed_batch.feed_slope is not None, "fed_batch.feed_slope required for LinearFeedStrategy"
        self.F_L0 = float(params.fed_batch.F_L0)
        self.slope = float(params.fed_batch.feed_slope)
        self.F_L_max = float(params.reactor.F_L_max)

    def calculate_feed_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Return the capped linear-ramp feed flow for the current phase time.

        Parameters
        ----------
        t : float
            Current phase time in h.
        y : list[float]
            Current state vector; accepted but not used.
        q_Glc, D_base, C_Glc_feed : float, optional
            Glucose-control inputs; accepted but not used.

        Returns
        -------
        float
            ``min(F_L0 + feed_slope * t, F_L_max)`` in L/h.
        """
        F_L = self.F_L0 + self.slope * t
        return min(F_L, self.F_L_max)
    
    def get_required_params(self) -> List[str]:
        """Return parameter paths used by the linear profile."""
        return ['fed_batch.F_L0_specific', 'reactor.F_L_max_specific', 'reactor.V_init', 'fed_batch.feed_slope']

class CSFRStrategy(FeedStrategyModel):
    """Scale feed with viable-cell density using a cell-specific feed rate.

    The strategy calculates ``D = CSFR * Xv`` and ``F_L = D * V``. Therefore,
    ``F_L = CSFR * Xv * V``. It does not use glucose feedback or proportional
    control; the final flow is limited by ``reactor.F_L_max``.
    """
    
    def _cache_parameters(self, params: SimulationParameters):
        """Cache ``fed_batch.CSFR`` and the reactor flow limit.

        Raises
        ------
        AssertionError
            If ``fed_batch.CSFR`` is not configured.
        """
        assert params.fed_batch.CSFR is not None, "fed_batch.CSFR required for CSFRStrategy"
        self.CSFR = params.fed_batch.CSFR
        self.F_L_max = float(params.reactor.F_L_max)

    def calculate_feed_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Calculate the capped cell-specific feed flow.

        Parameters
        ----------
        t : float
            Current phase time in h; accepted but not used.
        y : list[float]
            State vector containing ``Xv`` at index 0 and ``V`` at index 5.
        q_Glc, D_base, C_Glc_feed : float, optional
            Accepted for interface compatibility and not used.

        Returns
        -------
        float
            ``min(CSFR * Xv * V, F_L_max)`` in L/h.
        """
        F_L = float(self.CSFR * y[0] * y[5])  # y[0] is Xv, y[5] is V; L cell-1 h-1 * cells L-1 * L = L h-1
        if self.F_L_max is not None:
            F_L = min(F_L, self.F_L_max)
        return F_L

    def get_required_params(self) -> List[str]:
        """Return parameter paths used by the CSFR calculation."""
        return ['fed_batch.CSFR', 'reactor.F_L_max_specific']


class GlucoseSetpointFeedStrategy(FeedStrategyModel):
    """Control feed using an algebraic glucose setpoint calculation.

    The baseline medium dilution rate follows the glucose balance at the target
    concentration:

    ``D_alg = (Xv * q_Glc + C_Glc_setpoint * D_base) / (C_Glc_feed - C_Glc_setpoint)``.

    An optional proportional correction is added after applying the configured
    deadband. The commanded flow is ``F_L = D_cmd * V``, bounded to
    ``[0, reactor.F_L_max]``. An optional slew-rate limit bounds the change in
    flow between consecutive evaluations. If the feed-to-setpoint concentration
    difference is smaller than ``C_Glc_setpoint_den_epsilon``, the method
    returns zero flow to avoid an ill-conditioned calculation.
    """

    def _cache_parameters(self, params: SimulationParameters):
        """Cache glucose-control parameters and initialize slew-rate state.

        Raises
        ------
        AssertionError
            If the fed-batch section or ``C_Glc_setpoint`` is not configured.
        """
        assert params.fed_batch is not None, "fed_batch parameters required for GlucoseSetpointFeedStrategy"
        assert params.fed_batch.C_Glc_setpoint is not None, "fed_batch.C_Glc_setpoint required for GlucoseSetpointFeedStrategy"

        self.C_Glc_setpoint = float(params.fed_batch.C_Glc_setpoint)
        self.F_L_max = float(params.reactor.F_L_max)
        self.Kp = float(params.fed_batch.C_Glc_setpoint_Kp) if params.fed_batch.C_Glc_setpoint_Kp is not None else 0.0
        self.deadband = float(params.fed_batch.C_Glc_setpoint_deadband) if params.fed_batch.C_Glc_setpoint_deadband is not None else 0.0
        self.slew_rate = float(params.fed_batch.C_Glc_setpoint_slew_rate) if params.fed_batch.C_Glc_setpoint_slew_rate is not None else 0.0
        self.den_epsilon = float(params.fed_batch.C_Glc_setpoint_den_epsilon) if params.fed_batch.C_Glc_setpoint_den_epsilon is not None else 1e-12

        # Runtime state for optional slew-rate limiting.
        self._last_t: Optional[float] = None
        self._last_F_L: Optional[float] = None

    def calculate_feed_rate(
        self,
        t: float,
        y: List[float],
        q_Glc: Optional[float] = None,
        D_base: Optional[float] = None,
        C_Glc_feed: Optional[float] = None,
    ) -> float:
        """Calculate the bounded feed flow required to hold glucose near setpoint.

        Parameters
        ----------
        t : float
            Current ODE-phase time in h. Used to determine the elapsed time for
            optional slew-rate limiting.
        y : list[float]
            State vector containing ``Xv`` at index 0, ``C_Glc`` at index 1,
            and culture volume ``V`` at index 5.
        q_Glc : float
            Current specific glucose uptake rate in mmol/(cell·h).
        D_base : float
            Current pH-controller base-addition dilution rate in 1/h.
        C_Glc_feed : float
            Feed-medium glucose concentration in mmol/L.

        Returns
        -------
        float
            Commanded feed flow in L/h, bounded to ``[0, F_L_max]``. Returns
            ``0.0`` when the denominator is too small or the algebraic dilution
            rate is non-finite.

        Raises
        ------
        ValueError
            If ``q_Glc``, ``D_base``, or ``C_Glc_feed`` is omitted.
        """
        Xv = float(y[0])
        C_Glc = float(y[1])
        V = float(y[5])

        if q_Glc is None or D_base is None or C_Glc_feed is None:
            raise ValueError(
                "GlucoseSetpointFeedStrategy requires q_Glc, D_base, and C_Glc_feed "
                "for algebraic control."
            )

        
        denominator = float(C_Glc_feed) - self.C_Glc_setpoint
        if abs(denominator) < self.den_epsilon:
            F_L = 0.0
            self._last_t = float(t)
            self._last_F_L = F_L
            return F_L

        D_alg = (Xv * float(q_Glc) + self.C_Glc_setpoint * float(D_base)) / denominator
        if not np.isfinite(D_alg):
            return 0.0

        # Optional proportional correction for faster convergence from setpoint offsets.
        error = self.C_Glc_setpoint - C_Glc
        if abs(error) <= self.deadband:
            error = 0.0

        e_norm = error / denominator
        D_cmd = D_alg + self.Kp * e_norm

        F_L = float(D_cmd) * V
        F_L = max(0.0, min(F_L, self.F_L_max))

        # Optional slew-rate limiter in L/h^2 (change in F_L per hour).
        if self.slew_rate > 0.0 and self._last_t is not None and self._last_F_L is not None:
            dt = max(float(t) - self._last_t, 0.0)
            max_delta = self.slew_rate * dt
            if F_L > self._last_F_L + max_delta:
                F_L = self._last_F_L + max_delta
            elif F_L < self._last_F_L - max_delta:
                F_L = self._last_F_L - max_delta
            F_L = max(0.0, min(F_L, self.F_L_max))

        self._last_t = float(t)
        self._last_F_L = F_L
        return F_L

    def get_required_params(self) -> List[str]:
        """Return parameter paths required for glucose setpoint control."""
        return [
            'fed_batch.C_Glc_setpoint',
            'reactor.F_L_max_specific'
        ]

def create_feed_model(strategy_type: FeedStrategy, params: SimulationParameters) -> FeedStrategyModel:
    """Create the concrete feed strategy selected by an enum value.

    Parameters
    ----------
    strategy_type : FeedStrategy
        Feed-strategy enum member to instantiate.
    params : SimulationParameters
        Simulation parameters used to cache the selected strategy's inputs.

    Returns
    -------
    FeedStrategyModel
        Initialized ``ConstantFeedStrategy``, ``ExponentialFeedStrategy``,
        ``LinearFeedStrategy``, ``CSFRStrategy``, or
        ``GlucoseSetpointFeedStrategy``.

    Raises
    ------
    ValueError
        If ``strategy_type`` is not a supported :class:`FeedStrategy` member.
    AssertionError
        If the selected strategy's required optional parameter is absent.
    """
    strategies = {
        FeedStrategy.CONSTANT: ConstantFeedStrategy,
        FeedStrategy.EXPONENTIAL: ExponentialFeedStrategy,
        FeedStrategy.LINEAR: LinearFeedStrategy,
        FeedStrategy.CSFR: CSFRStrategy,
        FeedStrategy.GLUCOSE_SETPOINT: GlucoseSetpointFeedStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown feed strategy: {strategy_type}")
    
    return strategies[strategy_type](params)