"""Kinetic growth, limitation, inhibition, and rate models.

The module provides a common kinetic interface and two implementations:
``StoichiometricKinetics`` uses yields from ``StoichiometricModel`` and
``ExcessKineticModel`` derives excess substrate fluxes before stoichiometric
back-coupling. Both calculate Monod nutrient factors, inhibition factors,
growth rate, specific uptake/production rates, metabolic water, and heat.


Classes
-------
- Kinetics: Enum for selecting the kinetic-model formulation.
- KineticModel: Abstract base class for kinetic models.
- StoichiometricKinetics: Implements apparent stoichiometry yield relations to calculation kinetics.
- ExcessKineticModel: Implements advanced kinetics with excess substrate effects.

Functions
---------
- create_kinetic_model: Create a configured kinetic-model instance.

Use :func:`create_kinetic_model` with a configured
``StoichiometricModel``. ``calculate_rates`` updates the model's ``*_current``
attributes; ODE systems copy those values into detailed trace arrays.
"""
# Imports
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from animakinetica.core.parameters import SimulationParameters

if TYPE_CHECKING:
    from animakinetica.models.stoichiometry import StoichiometricModel


__version__ = "1.0.0"

__all__ = [
    "Kinetics",
    "KineticModel",
    "StoichiometricKinetics",
    "ExcessKineticModel",
    "create_kinetic_model",
]

## Kinetic Models ##
# Enum for model selection
class Kinetics(Enum):
    """Supported kinetic-model formulations.

    ``STOICHIOMETRIC`` calculates all rates from the yield vector supplied by
    :class:`~animakinetica.models.stoichiometry.StoichiometricModel`.
    ``EXCESS`` derives apparent catabolic glucose, glutamine, and lactate
    yields from empirical excess-rate terms before atom-balanced closure.
    """
    STOICHIOMETRIC = "stoichiometric"
    EXCESS = "excess"

    def __str__(self):
        """Return the display name for this formulation.

        Returns
        -------
        str
            ``"Stoichiometric"`` or ``"Excess"``.
        """
        model_names = {
            self.STOICHIOMETRIC: "Stoichiometric",
            self.EXCESS: "Excess"
        }
        return model_names.get(self, self.value.title())
    
    @property
    def abbr(self):
        """Return the compact identifier used in configuration filenames.

        Returns
        -------
        str
            ``"S"`` for stoichiometric kinetics or ``"E"`` for excess
            kinetics.
        """
        model_abbr = {
            self.STOICHIOMETRIC: "S",
            self.EXCESS: "E"
        }
        return model_abbr.get(self, self.value)

# Abstract base class for kinetics models
class KineticModel(ABC):
    """Abstract base class for kinetic models used by ODE systems.

    Subclasses calculate state-dependent specific rates and store them in
    ``*_current`` attributes. This stateful interface avoids allocating a
    result object at every ODE evaluation and lets ODE systems record detailed
    traces when performance mode is disabled.

    Parameters
    ----------
    params : SimulationParameters
        Immutable simulation parameters containing cell, kinetic, and
        inhibition constants.
    stoichiometric_model : StoichiometricModel
        Model that supplies full yield vectors and metabolic power for the
        configured stoichiometry mode.
    **kwargs
        Optional inhibition-model selectors. ``inhibition_model_Lac`` and
        ``inhibition_model_Amm`` default to ``"power_decay"``;
        ``inhibition_model_Osm`` defaults to ``"linear"``.
    """
    def __init__(
        self,
        params: SimulationParameters,
        stoichiometric_model: "StoichiometricModel",
        **kwargs,
    ):
        """Initialize cached parameters, inhibition choices, and rate state."""
        self.inhibition_model_Lac = kwargs.get("inhibition_model_Lac", "power_decay")
        self.inhibition_model_Amm = kwargs.get("inhibition_model_Amm", "power_decay")
        self.inhibition_model_Osm = kwargs.get("inhibition_model_Osm", "linear")
        self.stoichiometric_model = stoichiometric_model
        self._cache_parameters(params)
        self._validate_inhibition_parameters()
        
        # Initialize state variables for tracking during ODE integration
        self.lag_term_current = 0.0         # Current lag term value
        self.NF_glucose_current = 0.0       # Current glucose nutrient factor
        self.NF_glutamine_current = 0.0     # Current glutamine nutrient factor
        self.NF_AA_current = 0.0            # Current amino-acid nutrient factor
        self.NF_O2_current = 0.0            # Current O2 nutrient factor
        self.NF_current = 0.0               # Current combined nutrient factor
        self.IF_O2_current = 0.0            # Current O2 inhibition factor
        self.IF_CO2_current = 0.0           # Current CO2 inhibition factor
        self.IF_lactate_current = 0.0       # Current lactate inhibition factor
        self.IF_ammonia_current = 0.0       # Current ammonia inhibition factor
        self.IF_osmolality_current = 1.0    # Current osmolality inhibition factor (placeholder)
        self.IF_current = 0.0               # Current combined inhibition factor
        self.osm_current = 0.0              # Current osmolality (mOsm)
        
        # Initialize rate state variables
        self.mu_current = 0.0               # Current specific growth rate (1/h)
        self.q_Glc_current = 0.0            # Current glucose uptake rate (mmol/cell/h)
        self.q_Gln_current = 0.0            # Current glutamine uptake rate (mmol/cell/h)
        self.q_Lac_current = 0.0            # Current lactate production rate (mmol/cell/h)
        self.q_Amm_current = 0.0            # Current ammonia production rate (mmol/cell/h)
        self.q_O2_current = 0.0             # Current oxygen uptake rate (mmol/cell/h)
        self.q_CO2_current = 0.0            # Current CO2 production rate (mmol/cell/h)
        self.q_AA_current = 0.0            # Current amino acid uptake rate (mmol/cell/h)
        self.q_H2O_current = 0.0            # Current metabolic water production rate (mmol/cell/h)
        self.heat_rate_current = 0.0        # Current volumetric heat-generation rate (kJ/L/h)

    @abstractmethod
    def _cache_parameters(self, params: SimulationParameters):
        """Cache shared scalar parameters used during ODE evaluations.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters from which kinetic and inhibition constants
            are extracted.

        Notes
        -----
        This abstract method provides the common cache population and must be
        called by subclass overrides before their model-specific values are
        cached.
        """
        self.cells_per_mol = float(params.cell_properties.cells_per_mol)
        self.mu_max = float(params.cell_properties.mu_max)
        self.K_Glc = float(params.kinetics.K_Glc)
        self.K_Gln = float(params.kinetics.K_Gln)
        self.K_O2 = float(params.kinetics.K_O2)
        # Keep very low so AA only limits growth near depletion.
        self.K_AA = 1e-6
        self.t_lag = float(params.kinetics.t_lag)
        
        # Optional inhibition parameters - cache only if present
        self.phi_power_decay_Lac = float(params.kinetics.phi_power_decay_Lac) if params.kinetics.phi_power_decay_Lac is not None else None
        self.IC50_Lac = float(params.kinetics.IC50_Lac) if params.kinetics.IC50_Lac is not None else None
        self.IC100_Lac = float(params.kinetics.IC100_Lac) if params.kinetics.IC100_Lac is not None else None
        self.phi_power_decay_Amm = float(params.kinetics.phi_power_decay_Amm) if params.kinetics.phi_power_decay_Amm is not None else None
        self.IC50_Amm = float(params.kinetics.IC50_Amm) if params.kinetics.IC50_Amm is not None else None
        self.IC100_Amm = float(params.kinetics.IC100_Amm) if params.kinetics.IC100_Amm is not None else None
        self.phi_power_decay_Osm = float(params.kinetics.phi_power_decay_Osm) if params.kinetics.phi_power_decay_Osm is not None else None
        self.IC50_Osm = float(params.kinetics.IC50_Osm) if params.kinetics.IC50_Osm is not None else None
        # O2 and CO2 inhibition parameters, only one model implemented, these are not optional
        self.IC50_O2 = float(params.kinetics.IC50_O2)
        self.C_O2_inh_min = (
            float(params.kinetics.C_O2_inh_min)
            if params.kinetics.C_O2_inh_min is not None
            else 0.002
        )
        self.n_O2 = float(params.kinetics.n_O2)
        self.IC100_CO2 = float(params.kinetics.IC100_CO2)
        self.n_CO2 = float(params.kinetics.n_CO2)
        # self.Y_O2_Glc = float(params.kinetics.Y_O2_Glc) # Moved to stoichiometric model
        # self.Y_CO2_Glc = float(params.kinetics.Y_CO2_Glc) # Moved to stoichiometric model
        self.IC100_Osm = float(params.kinetics.IC100_Osm) # Not optional since linear model is used
        self.Osm_baseline_inhibition = float(params.kinetics.Osm_baseline_inhibition)
        # Derived parameters for performance
        self.IC50_O2_pow_n = np.power(self.IC50_O2, self.n_O2)
        self.IC100_CO2_pow_n = np.power(self.IC100_CO2, self.n_CO2)
        self.IC50_Osm_effective = self.IC50_Osm - self.Osm_baseline_inhibition if params.kinetics.IC50_Osm is not None else None
        self.IC100_Osm_effective = self.IC100_Osm - self.Osm_baseline_inhibition

    @abstractmethod
    def calculate_rates(self, t: float, y: List[float], C_CO2: float) -> None:
        """
        Calculate all specific rates for the given state and store them in _current attributes.
        
        Parameters
        ----------
        t : float
            Current time [h].
        y : list[float]
            State vector ``[Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, ...]``.
        C_CO2 : float
            Dissolved CO2 concentration [mmol/L].

        Notes
        -----
        Subclasses update ``mu_current``, the ``q_*_current`` specific-rate
        attributes, ``q_H2O_current``, and ``heat_rate_current`` in place.
        """
        pass

    def validate_inhibition_model(self, model: str):
        """Validate an inhibition-model selector.

        Parameters
        ----------
        model : str
            One of ``"linear"``, ``"exponential"``, ``"hyperbolic"``, or
            ``"power_decay"``.

        Raises
        ------
        ValueError
            If ``model`` is not supported.
        """
        valid_models = ["linear", "exponential", "hyperbolic", "power_decay"]
        if model not in valid_models:
            raise ValueError(f"Invalid inhibition model: {model}. Valid options are: {valid_models}")
    
    def _validate_inhibition_parameters(self):
        """Ensure each selected inhibition model has its required parameters.

        Raises
        ------
        ValueError
            If a selector is invalid or its required IC50, IC100, or
            power-decay exponent is absent from the kinetic parameters.
        """
        # Define required parameters for each inhibition model
        model_requirements = {
            "linear": ["IC100"],
            "exponential": ["IC50"],
            "hyperbolic": ["IC50"],
            "power_decay": ["IC100", "phi_power_decay"]
        }
        
        # Validate lactate inhibition model
        self.validate_inhibition_model(self.inhibition_model_Lac)
        required_params_lac = model_requirements[self.inhibition_model_Lac]
        
        for param in required_params_lac:
            if param == "IC50" and self.IC50_Lac is None:
                raise ValueError(f"Lactate inhibition model '{self.inhibition_model_Lac}' requires IC50_Lac parameter, but it is not provided.")
            elif param == "IC100" and self.IC100_Lac is None:
                raise ValueError(f"Lactate inhibition model '{self.inhibition_model_Lac}' requires IC100_Lac parameter, but it is not provided.")
            elif param == "phi_power_decay" and self.phi_power_decay_Lac is None:
                raise ValueError(f"Lactate inhibition model '{self.inhibition_model_Lac}' requires phi_power_decay parameter, but it is not provided.")
        
        # Validate ammonia inhibition model
        self.validate_inhibition_model(self.inhibition_model_Amm)
        required_params_amm = model_requirements[self.inhibition_model_Amm]
        
        for param in required_params_amm:
            if param == "IC50" and self.IC50_Amm is None:
                raise ValueError(f"Ammonia inhibition model '{self.inhibition_model_Amm}' requires IC50_Amm parameter, but it is not provided.")
            elif param == "IC100" and self.IC100_Amm is None:
                raise ValueError(f"Ammonia inhibition model '{self.inhibition_model_Amm}' requires IC100_Amm parameter, but it is not provided.")
            elif param == "phi_power_decay" and self.phi_power_decay_Amm is None:
                raise ValueError(f"Ammonia inhibition model '{self.inhibition_model_Amm}' requires phi_power_decay parameter, but it is not provided.")

        # Validate osmolality inhibition model
        self.validate_inhibition_model(self.inhibition_model_Osm)
        required_params_osm = model_requirements[self.inhibition_model_Osm]

        for param in required_params_osm:
            if param == "IC50" and self.IC50_Osm is None:
                raise ValueError(f"Osmolality inhibition model '{self.inhibition_model_Osm}' requires IC50_Osm parameter, but it is not provided.")
            elif param == "IC100" and self.IC100_Osm is None:
                raise ValueError(f"Osmolality inhibition model '{self.inhibition_model_Osm}' requires IC100_Osm parameter, but it is not provided.")
            elif param == "phi_power_decay" and not hasattr(self, 'phi_power_decay_Osm'):
                raise ValueError(f"Osmolality inhibition model '{self.inhibition_model_Osm}' requires phi_power_decay parameter, but it is not provided.")

    ### Growth Rate Kinetics ###
    # Base growth rate calculation using helper functions
    def monod_equation(self, Vmax: float, S: float, Ks: float) -> float:
        """
        Evaluate a non-negative generalized Monod relation.

        This helper is retained for reference; production nutrient-factor
        calculations are inlined to reduce ODE-evaluation overhead.

        Parameters
        ----------
        Vmax : float
            Maximum process rate.
        S : float
            Substrate or effector concentration.
        Ks : float
            Half-saturation concentration, in the same units as ``S``.

        Returns
        -------
        float
            $\max(V_{max}S/(K_s+S), 0)$.
        """

        return np.maximum(Vmax * S / (Ks + S), 0)

    def calculate_lag(self, t: float) -> float:
        """Calculate the transient lag-rate deduction for time ``t``.

        Parameters
        ----------
        t : float
            Elapsed culture time [h]. The method assumes ``t_lag > 0`` and is
            called by :meth:`calculate_mu` only while ``t < t_lag``.

        Returns
        -------
        float
            Non-steady lag deduction [1/h], which is subtracted from the
            Monod/inhibition growth rate.
        """
        q0 = (np.exp(self.mu_max * self.t_lag) - 1) ** (-1)
        a_lag = q0 / (q0 + np.exp(-self.mu_max * t))
        lag_term = (1 - t / self.t_lag) * a_lag * self.mu_max

        return lag_term

    # General inhibition function
    def inhibition_factor(self, conc: float, inhibition_model: str, **kwargs) -> float:
        """Evaluate an inhibitor-specific multiplicative growth factor.

        Parameters
        ----------
        conc : float
            Inhibitor concentration in the units associated with the supplied
            inhibition constants.
        inhibition_model : str
            ``"linear"``, ``"exponential"``, ``"hyperbolic"``, or
            ``"power_decay"``.
        **kwargs
            Model constants: ``IC100`` for linear, ``IC50`` for exponential
            and hyperbolic, and ``IC100`` plus ``phi_power_decay`` for power
            decay.

        Returns
        -------
        float
            Inhibition factor. The power-decay form is bounded to $[0, 1]$
            for non-negative concentrations; other forms follow their
            analytical expression.

        Raises
        ------
        ValueError
            If ``inhibition_model`` is unknown.
        """
        IC50 = kwargs.get("IC50", -1)
        IC100 = kwargs.get("IC100", -1)
        phi_power_decay = kwargs.get("phi_power_decay", -1)

        if inhibition_model == "linear":
            return 1 - conc / IC100 # Use IC100
        elif inhibition_model == "exponential":
            return np.exp(-conc / IC50) # Does not have an end point at 0.
        elif inhibition_model == "hyperbolic":
            return 1 / (1 + conc / IC50) # Use IC50. Does not have an end point at 0.
        elif inhibition_model == "power_decay":
            return np.power(np.maximum(1 - conc / IC100, 0.0), phi_power_decay)
        else: 
            raise ValueError(f"Unknown inhibition model: {inhibition_model}")

    def calculate_nutrient_factors(self, C_Glc: float, C_Gln: float, C_AA: float, C_O2: float) -> dict:
        """Calculate individual and combined Monod nutrient factors.

        Negative concentrations are treated as zero to prevent temporary ODE
        overshoot from creating negative limitation factors.

        Parameters
        ----------
        C_Glc, C_Gln, C_AA, C_O2 : float
            Glucose, glutamine, total amino-acid excluding glutamine, and
            dissolved-oxygen concentrations [mmol/L].

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            ``NF_glucose``, ``NF_glutamine``, ``NF_AA``, ``NF_O2``, and
            ``NF_combined``. The combined factor is the product of the four
            individual factors.
        """
        # Clamp to zero to avoid negative Monod values from transient numerical undershoot
        C_Glc_safe = np.maximum(C_Glc, 0.0)
        C_Gln_safe = np.maximum(C_Gln, 0.0)
        C_AA_safe  = np.maximum(C_AA,  0.0)
        C_O2_safe  = np.maximum(C_O2,  0.0)
        
        NF_glucose = C_Glc_safe / (self.K_Glc + C_Glc_safe)
        NF_glutamine = C_Gln_safe / (self.K_Gln + C_Gln_safe)
        NF_AA = C_AA_safe / (self.K_AA + C_AA_safe)
        NF_O2 = C_O2_safe / (self.K_O2 + C_O2_safe)
        NF_combined = NF_glucose * NF_glutamine * NF_AA * NF_O2
        
        return {
            'NF_glucose': NF_glucose,
            'NF_glutamine': NF_glutamine,
            'NF_AA': NF_AA,
            'NF_O2': NF_O2,
            'NF_combined': NF_combined
        }
    
    def calculate_inhibition_factors(self, C_Lac: float, C_Amm: float, 
                                    C_O2: float, C_CO2: float, Osm: float) -> dict:
        """Calculate individual and combined inhibition factors.

        Oxygen inhibition is disabled below ``C_O2_inh_min``. Negative CO2 is
        treated as zero because it can occur during transient ODE overshoot.

        Parameters
        ----------
        C_Lac, C_Amm, C_O2, C_CO2, Osm : float
            Lactate, ammonia, dissolved oxygen, dissolved CO2, and
            osmolality. Concentrations are [mmol/L]; osmolality is [mOsm].

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            ``IF_lactate``, ``IF_ammonia``, ``IF_O2``, ``IF_CO2``,
            ``IF_osmolality``, and ``IF_combined``. The combined factor is
            the product of all individual factors.
        """
        # Correct osmolality for baseline
        Osm_effective = np.abs(Osm - self.Osm_baseline_inhibition)
        
        # Calculate individual inhibition factors
        # Apply O2 inhibition only above the minimum inhibitory concentration.
        # Below this threshold, inhibition is disabled (factor = 1).
        C_O2_safe = np.maximum(C_O2, 0.0)
        C_O2_inhibiting = np.maximum(C_O2_safe - self.C_O2_inh_min, 0.0)
        IF_O2 = self.IC50_O2_pow_n / (self.IC50_O2_pow_n + np.power(C_O2_inhibiting, self.n_O2))
        # Clamp IF_O2 to [0, 1] range
        IF_O2 = np.clip(IF_O2, 0.0, 1.0)
        
        # Handle negative C_CO2 values to avoid power calculation warnings
        # Negative CO2 is unphysical but can occur transiently in numerical integration
        C_CO2_safe = np.maximum(C_CO2, 0.0)  # Clamp to zero if negative
        IF_CO2 = 1 - (np.power(C_CO2_safe, self.n_CO2) / self.IC100_CO2_pow_n)
        # Clamp IF_CO2 to [0, 1] range
        IF_CO2 = np.clip(IF_CO2, 0.0, 1.0)

        
        IF_lactate = self.inhibition_factor(
            C_Lac, self.inhibition_model_Lac,
            IC50=self.IC50_Lac, IC100=self.IC100_Lac,
            phi_power_decay=self.phi_power_decay_Lac
        )
        IF_ammonia = self.inhibition_factor(
            C_Amm, self.inhibition_model_Amm,
            IC50=self.IC50_Amm, IC100=self.IC100_Amm,
            phi_power_decay=self.phi_power_decay_Amm
        )
        IF_osmolality = self.inhibition_factor(
            Osm_effective, self.inhibition_model_Osm,
            IC50=self.IC50_Osm_effective, IC100=self.IC100_Osm_effective,
            phi_power_decay=self.phi_power_decay_Osm
        )
        
        IF_combined = IF_lactate * IF_ammonia * IF_O2 * IF_CO2 * IF_osmolality
        
        return {
            'IF_lactate': IF_lactate,
            'IF_ammonia': IF_ammonia,
            'IF_O2': IF_O2,
            'IF_CO2': IF_CO2,
            'IF_osmolality': IF_osmolality,
            'IF_combined': IF_combined
        }


    # Wrapper growth rate function: 
    def calculate_mu(self,
                    t: float,  
                    C_Glc: float,
                    C_Gln: float, 
                    C_AA: float,
                    C_Lac: float,
                    C_Amm: float,
                    C_O2: float,
                    C_CO2: float,
                    Osm: float
                    ) -> float:
        """Calculate growth rate from nutrient limitation, inhibition, and lag.

        The returned rate is
        $\max(\mu_{max} NF IF - \mathrm{lag\_term}, 0)$, where ``NF`` is the
        product of glucose, glutamine, amino-acid, and oxygen Monod factors.
        ``IF`` combines lactate, ammonia, oxygen, CO2, and osmolality effects.
        The individual and combined factors are retained in ``*_current``
        attributes for trace logging.

        Parameters
        ----------
        t : float
            Elapsed culture time [h].
        C_Glc, C_Gln, C_AA, C_Lac, C_Amm, C_O2, C_CO2, Osm : float
            Current glucose, glutamine, total amino-acid, lactate, ammonia,
            dissolved oxygen, dissolved CO2, and osmolality state. Concentrations
            are [mmol/L]; osmolality is [mOsm].

        Returns
        -------
        float
            Non-negative specific growth rate [1/h].
        """
        # Lag phase
        lag_term = self.calculate_lag(t) if t < self.t_lag else 0
        self.lag_term_current = lag_term

        # Calculate nutrient factors using centralized method
        nf_dict = self.calculate_nutrient_factors(C_Glc, C_Gln, C_AA, C_O2)
        
        # Store nutrient factors in state variables
        self.NF_glucose_current = nf_dict['NF_glucose']
        self.NF_glutamine_current = nf_dict['NF_glutamine']
        self.NF_AA_current = nf_dict['NF_AA']
        self.NF_O2_current = nf_dict['NF_O2']
        self.NF_current = nf_dict['NF_combined']

        # Calculate inhibition factors using centralized method
        if_dict = self.calculate_inhibition_factors(C_Lac, C_Amm, C_O2, C_CO2, Osm)
        
        # Store inhibition factors in state variables
        self.IF_O2_current = if_dict['IF_O2']
        self.IF_CO2_current = if_dict['IF_CO2']
        self.IF_lactate_current = if_dict['IF_lactate']
        self.IF_ammonia_current = if_dict['IF_ammonia']
        self.IF_osmolality_current = if_dict['IF_osmolality']
        self.IF_current = if_dict['IF_combined']

        return np.maximum(self.mu_max * self.NF_current * self.IF_current - lag_term, 0) 
    
    ### Stoichiometric kinetics functions ###
    def stochiometric_rate(self, q_S: float, Y_S: float) -> float:
        """Scale a specific substrate rate by its stoichiometric yield.

        Parameters
        ----------
        q_S : float
            Reference specific substrate uptake rate.
        Y_S : float
            Yield relative to the reference substrate.

        Returns
        -------
        float
            Specific consumption or production rate, ``q_S * Y_S``.
        """
        return q_S * Y_S

    # Generalized Pirt equation
    def q_pirt(self, mu: float, Y_max: float, m_subs: float) -> float:
        """Evaluate the Pirt uptake relation $q=m+\mu/Y$.

        Parameters
        ----------
        mu : float
            Specific growth rate [1/h].
        Y_max : float
            Biomass yield on substrate, typically [cells/mmol].
        m_subs : float
            Specific maintenance uptake rate, typically [mmol/cell/h].

        Returns
        -------
        float
            Specific substrate uptake rate, typically [mmol/cell/h].
            Returns ``0`` when ``Y_max`` is zero to avoid division by zero.
        """
        if Y_max == 0:
            return 0

        return m_subs + (1 / Y_max) * mu
   
    def _calculate_heat_rate(self, Xv: float) -> float:
        """Convert stoichiometric metabolic power to volumetric heat generation.

        Parameters
        ----------
        Xv : float
            Viable cell density [cells/L].

        Returns
        -------
        float
            Volumetric heat-generation rate [kJ/L/h].
        """
        return Xv * self.stoichiometric_model._metabolic_power / self.cells_per_mol / 1000.0


class StoichiometricKinetics(KineticModel):
    """Calculate Monod/inhibition growth and stoichiometrically consistent rates.

    The model calculates growth independently of yields, then requests the
    yield vector applicable to the current growth rate from the configured
    :class:`~animakinetica.models.stoichiometry.StoichiometricModel`. It is
    therefore compatible with constant and dynamic stoichiometry modes.
    """

    def _cache_parameters(self, params: SimulationParameters):
        """Cache common kinetic and inhibition parameters.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters used by the shared kinetic model.
        """
        super()._cache_parameters(params)

    def calculate_rates(self, t: float, y: List[float], C_CO2: float) -> None:
        """
        Calculate all specific rates using stoichiometric yield coefficients.
        
        Growth rate (mu) is computed first from Monod kinetics — it has no dependency on yields.
        Yields are then retrieved from the stoichiometric model at the current mu, ensuring
        full consistency between mu and all substrate/product rates without any lag.
        
        Parameters
        ----------
        t : float
            Current time [h].
        y : list[float]
            State vector ``[Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, ...]``.
        C_CO2 : float
            Dissolved CO2 concentration [mmol/L].

        Notes
        -----
        Updates cached growth, specific-rate, water, and heat values in place.
        """
        # Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA = y 
        
        # Step 1: Calculate growth rate — no yields needed (Monod + inhibition only)
        mu = self.calculate_mu(t, y[1], y[2], y[10], y[3], y[4], y[6], C_CO2, self.osm_current)
        cells_per_mmol = self.cells_per_mol / 1000  # convert cells / mol to cells/mmol

        # Step 2: Get yields consistent with this mu (free in CONSTANT mode, recalculated in DYNAMIC)
        yield_coefs = self.stoichiometric_model.calculate_yields(mu)

        # Step 3: Calculate substrate/product rates from current yields
        q_Glc = (mu * yield_coefs[0]) / cells_per_mmol  # convert mmol/mmol/h to mmol/cell/h
        
        # Extract yield coefficients
        q_Lac, q_Gln, q_O2, q_CO2, q_AA = float(q_Glc) * yield_coefs[1:6] # Units: mmol/cell/h
        q_Amm = q_Gln * yield_coefs[6] # Units: mmol/cell/h
        q_H2O = (mu * yield_coefs[7]) / cells_per_mmol # Units: mmol/cell/h
        heat_rate = self._calculate_heat_rate(y[0])
        
        # Store calculated rates in state variables
        self.mu_current = mu
        self.q_Glc_current = q_Glc
        self.q_Gln_current = q_Gln
        self.q_Lac_current = q_Lac
        self.q_Amm_current = q_Amm
        self.q_O2_current = q_O2
        self.q_CO2_current = q_CO2
        self.q_AA_current = q_AA
        self.q_H2O_current = q_H2O
        self.heat_rate_current = heat_rate
    
class ExcessKineticModel(KineticModel):
    """Use empirical excess fluxes with atom-balanced stoichiometric closure.

    Glucose and glutamine excess terms define apparent catabolic uptake, while
    a fitted lactate profile defines apparent catabolic lactate formation.
    These values are converted to catabolic yields, feasibility-clamped, and
    passed to ``StoichiometricModel.calculate_yields``. The returned yield
    vector reconstructs all substrate, product, water, and heat rates.

    ``ExcessKineticModel`` requires ``params.excess_kinetics`` and a
    stoichiometric model configured for ``EXCESS_COUPLED`` operation.
    """

    def _cache_parameters(self, params: SimulationParameters):
        """Cache shared and excess-kinetic parameters.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters containing an ``excess_kinetics`` section in
            addition to the shared kinetic parameters.

        Raises
        ------
        ValueError
            If ``params.excess_kinetics`` is not configured.
        """
        super()._cache_parameters(params)

        if params.excess_kinetics is None:
            raise ValueError("ExcessKineticModel requires excess_kinetics parameters")
        
        self.Y_Glc = float(params.excess_kinetics.Y_Glc)                # cell/mmol
        self.Y_Gln = float(params.excess_kinetics.Y_Gln)                # cell/mmol
        self.m_Glc = float(params.excess_kinetics.m_Glc)                # mmol/cell/h
        self.m_Gln = float(params.excess_kinetics.m_Gln)                # mmol/cell/h
        self.C_Glc_limit = float(params.excess_kinetics.C_Glc_limit)    # mmol/L
        self.C_Gln_limit = float(params.excess_kinetics.C_Gln_limit)    # mmol/L

        # Excess kinetics parameters
        self.K_Glc_Glc = float(params.excess_kinetics.K_Glc_Glc)                            # mmol/L
        self.delta_q_Glc_Glc_max = float(params.excess_kinetics.delta_q_Glc_Glc_max)        # mmol/cell/h
        self.K_Gln_Glc = float(params.excess_kinetics.K_Gln_Glc)                            # mmol/L
        self.delta_q_Gln_Glc_max = float(params.excess_kinetics.delta_q_Gln_Glc_max)        # mmol/cell/h
        self.K_Glc_Gln = float(params.excess_kinetics.K_Glc_Gln)                            # mmol/L
        self.delta_q_Glc_Gln_max = float(params.excess_kinetics.delta_q_Glc_Gln_max)        # mmol/cell/h
        self.K_Gln_Gln = float(params.excess_kinetics.K_Gln_Gln)                            # mmol/L
        self.delta_q_Gln_Gln_max = float(params.excess_kinetics.delta_q_Gln_Gln_max)        # mmol/cell/h
        
        # Basal catabolic closure assumption: maintenance catabolism converts glucose
        # to lactate at a fixed basal catabolic yield.
        self.Y_Lac_Glc_lim = 0.5

        # #  Deprecated lactate excess kinetic parameters - not currently used since lactate production is fitted to glucose uptake
        # self.m_Lac = float(params.excess_kinetics.m_Lac) if params.excess_kinetics.m_Lac is not None else None
        # self.Y_Lac = float(params.excess_kinetics.Y_Lac) if params.excess_kinetics.Y_Lac is not None else None
        # self.K_Glc_Lac = float(params.excess_kinetics.K_Glc_Lac) if params.excess_kinetics.K_Glc_Lac is not None else None
        # self.delta_q_Lac_max = float(params.excess_kinetics.delta_q_Lac_max) if params.excess_kinetics.delta_q_Lac_max is not None else None

        # # Deprecated ammonia excess kinetic parameters - not currently used since ammonia is calculated with stoichiometry
        # self.m_Amm = float(params.excess_kinetics.m_Amm) if params.excess_kinetics.m_Amm is not None else None
        # self.Y_Amm = float(params.excess_kinetics.Y_Amm) if params.excess_kinetics.Y_Amm is not None else None
        # self.K_Gln_Amm = float(params.excess_kinetics.K_Gln_Amm) if params.excess_kinetics.K_Gln_Amm is not None else None
        # self.delta_q_Amm_max = float(params.excess_kinetics.delta_q_Amm_max) if params.excess_kinetics.delta_q_Amm_max is not None else None

        # Compute lactate profile constants based on the excess kinetic parameters and stoichiometric model assumptions.
        self._precompute_excess_profile_constants()

        # Reset cached intermediate terms for catabolic yield calculations
        self._reset_cat_yield_term_state()

    def _precompute_excess_profile_constants(self) -> None:
        """Precompute invariant coefficients for excess catabolic-yield closure.

        The cached values convert empirical cell-based uptake parameters into
        the molar dry-cell-mass basis used by the stoichiometric model.
        """
        mmol_per_cell_from_mol_per_mol_dcm = 1000.0 / self.cells_per_mol

        # Anabolic yield coefficient from the fixed anabolic reaction.
        dcm_coeff = float(self.stoichiometric_model.anabolic_dict.get("DCM", 1.0))
        anabolic_glc_coeff = np.abs(float(self.stoichiometric_model.anabolic_dict.get("Glc", 0.0)))
        self.Y_Glc_ana = anabolic_glc_coeff / dcm_coeff if dcm_coeff != 0 else 0.0

        # Convert the empirical substrate-limited glucose coefficient from cell/mmol
        # to mol Glc / mol DCM so it can be combined with the stoichiometric model.
        self.Y_Glc_lim = 1 / self.Y_Glc / 1000.0 * self.cells_per_mol if self.Y_Glc != 0 else 0.0

        # Extract the basal catabolic glucose coefficient by removing the anabolic
        # glucose demand implied by the fixed anabolic reaction.
        self.Y_Glc_cat_lim = max(self.Y_Glc_lim - self.Y_Glc_ana, 0.0)

        # Lactate profile assumptions:
        # - basal catabolic closure converts glucose to lactate with a fixed
        #   substrate-limited yield Y_Lac_Glc_lim
        # - maintenance-associated lactate production is set to zero for now
        self.Y_Lac_lim = self.Y_Lac_Glc_lim * self.Y_Glc_cat_lim 
        self.m_Lac = 0.0
        self.K_Glc_Lac = self.K_Glc_Glc # Half saturation constant
        self.C_Lac_limit = self.C_Glc_limit # Substrate limiting concentration

        # The theoretical maximum catabolic lactate yield from glucose is
        # 2 mol Lac / mol catabolic Glc. Use the maximum catabolic glucose flux,
        # not the total glucose uptake, as the reference point for the lactate
        # Since glutamine excess uptake is not contributing to lactate production, 
        # it should be excluded from the calculation of the maximum lactate production in the
        # overflow profile.
        self.q_Glc_cat_max = (
            self.mu_max * self.Y_Glc_cat_lim * mmol_per_cell_from_mol_per_mol_dcm
            + self.m_Glc
            + self.delta_q_Glc_Glc_max
            # + self.delta_q_Gln_Glc_max # Exclude, but shown for clarity.
        )
        self.q_Lac_max = self.q_Glc_cat_max * 2.0

        self.delta_q_Lac_max = (
            self.q_Lac_max - self.Y_Lac_lim * mmol_per_cell_from_mol_per_mol_dcm * self.mu_max - self.m_Lac
        )

    def _reset_cat_yield_term_state(self) -> None:
        """Initialize inspectable excess-closure intermediate values to zero."""
        self.q_Lac_lim_current = 0.0
        self.q_Lac_exc_current = 0.0
        self.q_Glc_lim_current = 0.0
        self.q_Glc_ana_current = 0.0
        self.q_Glc_cat_lim_current = 0.0
        self.delta_q_Glc_Glc_current = 0.0
        self.delta_q_Gln_Glc_current = 0.0
        self.delta_q_Glc_Gln_current = 0.0
        self.delta_q_Gln_Gln_current = 0.0
        self.q_Glc_exc_current = 0.0
        self.q_Gln_exc_current = 0.0
        self.q_Glc_cat_current = 0.0
        self.q_Gln_cat_current = 0.0
        self.q_Lac_cat_current = 0.0
        self.Y_Lac_Glc_cat_current = 0.0
        self.Y_Gln_Glc_cat_current = 0.0
        self.Y_Lac_Glc_cat_actual_current = 0.0

    def get_cat_yield_terms(
        self,
        mu: Optional[float] = None,
        C_Glc: Optional[float] = None,
        C_Gln: Optional[float] = None,
    ) -> dict:
        """Return excess-closure intermediates from the latest evaluation.

        Supplying all three optional inputs performs a fresh catabolic-yield
        evaluation before the values are returned. Supplying only a subset is
        invalid because a complete physical state is required.

        Parameters
        ----------
        mu : float, optional
            Specific growth rate [1/h].
        C_Glc, C_Gln : float, optional
            Glucose and glutamine concentrations [mmol/L].

        Returns
        -------
        dict[str, float]
            Limited and excess flux components [mmol/cell/h] and apparent
            catabolic yield terms [mmol/mmol].

        Raises
        ------
        ValueError
            If only some of ``mu``, ``C_Glc``, and ``C_Gln`` are provided.
        """
        if mu is not None or C_Glc is not None or C_Gln is not None:
            if None in (mu, C_Glc, C_Gln):
                raise ValueError("mu, C_Glc, and C_Gln must all be provided together")
            self.calculate_cat_yields_from_excess(float(mu), float(C_Glc), float(C_Gln))

        return {
            "q_Lac_lim": self.q_Lac_lim_current,
            "q_Lac_exc": self.q_Lac_exc_current,
            "q_Glc_lim": self.q_Glc_lim_current,
            "q_Glc_ana": self.q_Glc_ana_current,
            "q_Glc_cat_lim": self.q_Glc_cat_lim_current,
            "delta_q_Glc_Glc": self.delta_q_Glc_Glc_current,
            "delta_q_Gln_Glc": self.delta_q_Gln_Glc_current,
            "delta_q_Glc_Gln": self.delta_q_Glc_Gln_current,
            "delta_q_Gln_Gln": self.delta_q_Gln_Gln_current,
            "q_Glc_exc": self.q_Glc_exc_current,
            "q_Gln_exc": self.q_Gln_exc_current,
            "q_Glc_cat": self.q_Glc_cat_current,
            "q_Gln_cat": self.q_Gln_cat_current,
            "q_Lac_cat": self.q_Lac_cat_current,
            "Y_Lac_Glc_cat": self.Y_Lac_Glc_cat_current,
            "Y_Gln_Glc_cat": self.Y_Gln_Glc_cat_current,
            "Y_Lac_Glc_cat_actual": self.Y_Lac_Glc_cat_actual_current,
        }
        
    def calculate_rates(self, t: float, y: List[float], C_CO2: float) -> None:
        """
          Calculate all specific rates using the excess catabolic-yield closure.

          The excess model first derives apparent catabolic yields from the current
          growth rate and substrate concentrations:

          1. Compute mu from Monod kinetics with inhibition.
          2. Build apparent catabolic Y_Lac/Glc and Y_Gln/Glc with
              calculate_cat_yields_from_excess().
          3. Pass those yields to stoichiometric_model.calculate_yields(...) for
              atom-balanced stoichiometric closure.
          4. Reconstruct q_Glc, q_Gln, q_Lac, q_O2, q_CO2, q_AA, q_Amm and q_H2O
              from the returned yield array.

        Parameters
        ----------
        t : float
            Current time [h].
        y : list[float]
            State vector ``[Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, ...]``.
        C_CO2 : float
            Dissolved CO2 concentration [mmol/L].

        Notes
        -----
        Updates cached growth, specific-rate, water, heat, and excess-closure
        intermediate values in place.
        """
        # Step 1: Calculate growth rate (Monod + inhibition, no yields needed)
        mu = self.calculate_mu(t, y[1], y[2], y[10], y[3], y[4], y[6], C_CO2, self.osm_current)
        cells_per_mmol = self.cells_per_mol / 1000  # convert cells / mol to cells/mmol

        # Step 2: Calculate catabolic rates for Glc, Gln, and Lac from empirical excess kinetics equations.
        Y_Lac_Glc_cat, Y_Gln_Glc_cat = self.calculate_cat_yields_from_excess(
            mu,
            y[1],
            y[2],
        )

        # Step 3: Update catabolic stoichiometry from catabolic fluxes and obtain full yield array
        yield_coefs = self.stoichiometric_model.calculate_yields(mu, (Y_Lac_Glc_cat, Y_Gln_Glc_cat))

        # Step 4: Derive rates from stoichiometric yields
        #  yield_coefs[0] = Y_Glc/X, [1] = Y_Lac/Glc, [2] = Y_Gln/Glc, [3] = Y_O2/Glc
        #  yield_coefs[4] = Y_CO2/Glc, [5] = Y_AA/Glc, [6] = Y_Amm/Gln
        #  yield_coefs[7] = Y_H2O/X
        q_Glc = (mu * yield_coefs[0]) / cells_per_mmol  # convert mmol/mmol/h to mmol/cell/h
        q_Lac = q_Glc * yield_coefs[1] 
        q_Gln = q_Glc * yield_coefs[2] 
        q_O2  = q_Glc * yield_coefs[3]
        q_CO2 = q_Glc * yield_coefs[4]
        q_AA  = q_Glc * yield_coefs[5]
        q_Amm = q_Gln * yield_coefs[6]
        q_H2O = (mu * yield_coefs[7]) / cells_per_mmol
        heat_rate = self._calculate_heat_rate(y[0])

        # Store all rates in state variables
        self.mu_current    = mu
        self.q_Glc_current = q_Glc
        self.q_Gln_current = q_Gln
        self.q_Lac_current = q_Lac
        self.q_Amm_current = q_Amm
        self.q_O2_current  = q_O2
        self.q_CO2_current = q_CO2
        self.q_AA_current  = q_AA
        self.q_H2O_current = q_H2O
        self.heat_rate_current = heat_rate

    def calculate_cat_yields_from_excess(
        self,
        mu: float,
        C_Glc: float,
        C_Gln: float,
    ) -> List[float]:
        """
        Calculate yields from excess rates for back-calculation in the stoichiometric model.

        Parameters
        ----------
        mu : float
            Specific growth rate [1/h].
        C_Glc, C_Gln : float
            Glucose and glutamine concentrations [mmol/L].

        Returns
        -------
        list[float]
            Feasibility-clamped apparent catabolic yields
            ``[Y_Lac_Glc_cat, Y_Gln_Glc_cat]`` [mmol/mmol].
        """
        q_Glc_cat, q_Gln_cat = self._calculate_catabolic_substrate_rates(mu, C_Glc, C_Gln)
        q_Lac_cat = self._calculate_lactate_catabolic_rate(mu, C_Glc)

        return self._calculate_feasible_catabolic_yields(q_Glc_cat, q_Gln_cat, q_Lac_cat)

    def _calculate_catabolic_substrate_rates(
        self,
        mu: float,
        C_Glc: float,
        C_Gln: float,
    ) -> Tuple[float, float]:
        """Calculate limited and overflow catabolic uptake rates.

        Parameters
        ----------
        mu : float
            Specific growth rate [1/h].
        C_Glc, C_Gln : float
            Glucose and glutamine concentrations [mmol/L].

        Returns
        -------
        tuple[float, float]
            Catabolic glucose and glutamine uptake rates [mmol/cell/h].
        """
        mmol_per_cell_from_mol_per_mol_dcm = 1000.0 / self.cells_per_mol

        # Basal catabolic glucose comes from the substrate-limited glucose uptake
        # after removing the anabolic glucose demand implied by the fixed anabolic
        # reaction. The maintenance term is assumed to be catabolic.
        q_Glc_lim = mu * self.Y_Glc_lim * mmol_per_cell_from_mol_per_mol_dcm + self.m_Glc
        q_Glc_ana = mu * self.Y_Glc_ana * mmol_per_cell_from_mol_per_mol_dcm
        q_Glc_cat_lim = mu * self.Y_Glc_cat_lim * mmol_per_cell_from_mol_per_mol_dcm + self.m_Glc
        self.q_Glc_lim_current = q_Glc_lim
        self.q_Glc_ana_current = q_Glc_ana
        self.q_Glc_cat_lim_current = q_Glc_cat_lim

        # Overflow catabolic terms from the excess functions.
        delta_q_Glc_Glc, delta_q_Gln_Glc, delta_q_Glc_Gln, delta_q_Gln_Gln = self.delta_q_excess_Glc_Gln(C_Glc, C_Gln)
        q_Glc_exc = delta_q_Glc_Glc + delta_q_Gln_Glc
        q_Gln_exc = delta_q_Glc_Gln + delta_q_Gln_Gln
        self.delta_q_Glc_Glc_current = delta_q_Glc_Glc
        self.delta_q_Gln_Glc_current = delta_q_Gln_Glc
        self.delta_q_Glc_Gln_current = delta_q_Glc_Gln
        self.delta_q_Gln_Gln_current = delta_q_Gln_Gln
        self.q_Glc_exc_current = q_Glc_exc
        self.q_Gln_exc_current = q_Gln_exc

        # Overall catabolic uptake rates: glucose is basal plus overflow, while
        # glutamine catabolism is assumed to arise only from excess-driven terms.
        q_Glc_cat = np.maximum(q_Glc_cat_lim + q_Glc_exc, 0.0)
        q_Gln_cat = np.maximum(q_Gln_exc, 0.0)
        self.q_Glc_cat_current = q_Glc_cat
        self.q_Gln_cat_current = q_Gln_cat

        return q_Glc_cat, q_Gln_cat

    def _calculate_lactate_catabolic_rate(self, mu: float, C_Glc: float) -> float:
        """Calculate the lactate formation rate used for catabolic closure.

        Parameters
        ----------
        mu : float
            Specific growth rate [1/h].
        C_Glc : float
            Glucose concentration [mmol/L].

        Returns
        -------
        float
            Apparent catabolic lactate formation rate [mmol/cell/h].
        """
        mmol_per_cell_from_mol_per_mol_dcm = 1000.0 / self.cells_per_mol

        # Lactate follows a fitted profile rather than a separate excess dataset.
        # The substrate-limited point comes from Y_Lac_Glc_lim, and the overflow
        # term follows the same glucose-excess shape used for glucose uptake.
        q_Lac_lim = self.Y_Lac_lim * mmol_per_cell_from_mol_per_mol_dcm * mu + self.m_Lac
        q_Lac_exc = np.maximum(
            self.delta_q_Lac_max
            * (C_Glc - self.C_Lac_limit)
            / (self.K_Glc_Lac + (C_Glc - self.C_Lac_limit) + 1e-9),
            0,
        )
        self.q_Lac_lim_current = q_Lac_lim
        self.q_Lac_exc_current = q_Lac_exc

        # Total lactate catabolism is the limited contribution plus the overflow
        # contribution, protected against negative values.
        q_Lac_cat = np.maximum(q_Lac_lim + q_Lac_exc, 0.0)
        self.q_Lac_cat_current = q_Lac_cat

        return q_Lac_cat

    def _calculate_feasible_catabolic_yields(
        self,
        q_Glc_cat: float,
        q_Gln_cat: float,
        q_Lac_cat: float,
    ) -> Tuple[float, float]:
        """Convert fluxes to apparent yields and enforce atom-balance feasibility.

        Parameters
        ----------
        q_Glc_cat, q_Gln_cat, q_Lac_cat : float
            Catabolic glucose uptake, glutamine uptake, and lactate formation
            rates [mmol/cell/h].

        Returns
        -------
        tuple[float, float]
            Feasible ``(Y_Lac_Glc_cat, Y_Gln_Glc_cat)`` [mmol/mmol]. A zero
            pair is returned when glucose catabolism is numerically negligible.
        """
        # Protect against numerical blow-up when catabolic glucose is near zero.
        if q_Glc_cat <= 1e-20:
            self.Y_Lac_Glc_cat_current = 0.0
            self.Y_Gln_Glc_cat_current = 0.0
            self.Y_Lac_Glc_cat_actual_current = 0.0
            return 0.0, 0.0

        Y_Lac_Glc_cat = max(q_Lac_cat / q_Glc_cat, 0.0)
        Y_Gln_Glc_cat = max(q_Gln_cat / q_Glc_cat, 0.0)
        self.Y_Lac_Glc_cat_current = Y_Lac_Glc_cat
        self.Y_Gln_Glc_cat_current = Y_Gln_Glc_cat

        # Enforce atom-balance feasibility for the catabolic closure used in
        # StoichiometricModel._atom_balance_catabolic:
        # O2_cons = 6 + 4.5*Y_Gln_Glc - 3*Y_Lac_Glc must be >= 0
        # => Y_Lac_Glc <= 2 + 1.5*Y_Gln_Glc.
        # Keep a tiny margin to prevent floating-point sign flips at the boundary.
        y_lac_max = max(2.0 + 1.5 * Y_Gln_Glc_cat - 1e-12, 0.0)
        Y_Lac_Glc_cat_actual = min(Y_Lac_Glc_cat, y_lac_max)
        self.Y_Lac_Glc_cat_actual_current = Y_Lac_Glc_cat_actual

        return Y_Lac_Glc_cat_actual, Y_Gln_Glc_cat


    def delta_q_excess_general(self, 
        C_S: float,
        C_S_limit: float,
        K: float,
        delta_q_max: float
    ) -> float:
        """Calculate a thresholded Monod-like excess uptake correction.

        Parameters
        ----------
        C_S : float
            Current substrate concentration [mmol/L].
        C_S_limit : float
            Concentration below which excess uptake is zero [mmol/L].
        K : float
            Excess-term half-saturation constant [mmol/L].
        delta_q_max : float
            Asymptotic excess uptake correction [mmol/cell/h].

        Returns
        -------
        float
            Excess uptake correction [mmol/cell/h].
        """
        # Use a true thresholded excess term (zero below limit) and keep a small
        # denominator epsilon for numerical robustness when K and C_excess are both zero.
        C_excess = np.maximum(C_S - C_S_limit, 0.0)

        return delta_q_max * C_excess / (K + C_excess + 1e-9)


    def delta_q_excess_Glc_Gln(self,
        C_Glc: float, 
        C_Gln: float
    ) -> List[float]:
        """Calculate all cross-substrate glucose and glutamine excess terms.

        Parameters
        ----------
        C_Glc, C_Gln : float
            Glucose and glutamine concentrations [mmol/L].

        Returns
        -------
        list[float]
            ``[delta_q_Glc_Glc, delta_q_Gln_Glc, delta_q_Glc_Gln,
            delta_q_Gln_Gln]`` [mmol/cell/h]. The first subscript identifies
            the concentration driving the term; the second identifies the
            affected substrate uptake rate.
        """
        return [
            self.delta_q_excess_general(C_Glc, self.C_Glc_limit, self.K_Glc_Glc, self.delta_q_Glc_Glc_max),
            self.delta_q_excess_general(C_Gln, self.C_Gln_limit, self.K_Gln_Glc, self.delta_q_Gln_Glc_max),
            self.delta_q_excess_general(C_Glc, self.C_Glc_limit, self.K_Glc_Gln, self.delta_q_Glc_Gln_max),
            self.delta_q_excess_general(C_Gln, self.C_Gln_limit, self.K_Gln_Gln, self.delta_q_Gln_Gln_max),
        ]



# Factory function to create kinetics models
def create_kinetic_model(
    model_type: Kinetics,
    params: SimulationParameters,
    stoichiometric_model: "StoichiometricModel",
    **kwargs,
) -> KineticModel:
    """Create the kinetic model selected by ``model_type``.

    Parameters
    ----------
    model_type : Kinetics
        Kinetic formulation to instantiate.
    params : SimulationParameters
        Simulation parameters used to cache kinetic constants.
    stoichiometric_model : StoichiometricModel
        Stoichiometric model used to retrieve or close yield coefficients.
    **kwargs
        Optional inhibition-model selectors forwarded unchanged to the
        kinetic-model constructor.

    Returns
    -------
    KineticModel
        A :class:`StoichiometricKinetics` or :class:`ExcessKineticModel`
        instance.

    Raises
    ------
    ValueError
        If ``model_type`` is not a supported :class:`Kinetics` member.
    """
    if model_type == Kinetics.STOICHIOMETRIC:
        return StoichiometricKinetics(params, stoichiometric_model, **kwargs)
        # In case multiple stoichiometric models are added in the future, we can add additional validation logic here for providing the stoichiometric model.
    elif model_type == Kinetics.EXCESS:
        return ExcessKineticModel(params, stoichiometric_model, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



# Test code or examples
if __name__ == "__main__":
    pass