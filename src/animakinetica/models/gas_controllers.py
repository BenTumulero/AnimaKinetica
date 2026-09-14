"""Algebraic dissolved-oxygen control and stirred-tank mass-transfer calculations.

The controllers in this module calculate the oxygen transfer rate (OTR) required
to maintain the configured dissolved-oxygen (DO) setpoint.  They then select
agitation speed, sparging rate, and—where enabled—inlet oxygen fraction within
their configured limits.  The ODE systems use the resulting ``OTR_current`` in
the dissolved-oxygen balance.

The implemented controller is for a stirred-tank reactor (STR).  It estimates
the volumetric oxygen mass-transfer coefficient, ``klO2a_current`` (1/h), with
the non-coalescing-medium van 't Riet correlation.  It also exposes power input
per liquid volume (W/m³) and superficial gas velocity (m/s) for logging.

Carbon-dioxide transfer is not controlled in this module.  ``pHControllerModel``
uses the current oxygen ``kLa`` to estimate CO2 transfer and to support its
separate pH-control calculation.
"""
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from animakinetica.core.parameters import SimulationParameters

__version__ = "1.0.0"

__all__ = [
    "GasController",
    "GasControllerModel",
    "STRCascadeControllerModel",
    "create_gas_controller_model",
]


class GasController(Enum):
    """Gas-control strategies accepted by :func:`create_gas_controller_model`.

    ``CASCADE_N_FG_XO2`` is implemented for STRs.  It increases agitation,
    then gas flow, then inlet oxygen fraction as oxygen demand rises.
    ``CASCADE_N_FG`` is reserved for a cascade without oxygen enrichment and
    currently raises :class:`NotImplementedError` in the factory.
    """
    CASCADE_N_FG_XO2 = "cascade_N_FG_XO2"  # Cascade control: first N, then gas flow, then O2 fraction
    CASCADE_N_FG = "cascade_N_FG"          # Cascade control: first N, then gas flow

class GasControllerModel(ABC):
    """Abstract interface for algebraic dissolved-oxygen controllers.

    Subclasses update the public current operating-point attributes used by the
    ODE systems and result logger: ``OTR_current`` (mmol/L/h),
    ``klO2a_current`` (1/h), ``N_current`` (rpm), ``Fg_current`` (vvh),
    ``y_O2_current`` (-), ``P_over_V_current`` (W/m³), and ``v_s_current``
    (m/s).  Here, vvh is gas volume per liquid volume per hour.

    Parameters
    ----------
    params : SimulationParameters
        Simulation configuration providing reactor geometry, operating limits,
        medium density, oxygen Henry constant, and the DO setpoint.
    """

    def __init__(self, params: SimulationParameters):
        
        # Operating state variables
        self.klO2a_current = 0.0       # Current mass transfer coefficient (1/h)
        self.OTR_current = 0.0       # Current Oxygen Transfer Rate (mmol/L/h) # To be calculated using calculate_gas_transfer_rates()
        self.V_current = float(params.reactor.V_batch_init)  # Initial volume (L)

        self.N_current = 0.0         # Current agitation speed (rpm)
        self.Fg_current = 0.0        # Current gas flow rate (vvh)
        self.y_O2_current = 0.0      # Current O2 mole fraction in
        self.P_over_V_current = 0.0  # Current power input per liquid volume (W/m^3)
        self.v_s_current = 0.0       # Current superficial gas velocity (m/s)

        # Reactor geometry and physical properties
        self.power_number = float(params.reactor.power_number)  # Assume power number for impeller
        self.D_reactor = float(params.reactor.D_reactor)  # Tank diameter (m)
        self.d_impeller = float(params.reactor.d_impeller)  # Impeller diameter (m)
        self.tank_crossarea = np.pi * (self.D_reactor / 2)**2  # Cross-sectional area of tank (m²)  
        self.P_operating = float(params.reactor.P_operating)  # Operating pressure (atm)
        self.density = float(params.shared_medium_properties.density)  # Density of the culture medium (kg/m³)
        # Pressure calculation
        self._update_P_avg()  # Sets P_avg based on initial volume

        # Gas transfer properties
        self.H_O2 = float(params.gas_transfer.H_O2)  # Henry's constant for O2 (mmol/L/atm)

        # Setpoint
        self.DO_setpoint = float(params.reactor.DO_setpoint) # Dissolved oxygen setpoint in mmol/L

    def _update_P_avg(self):
        """Update average liquid pressure from the current culture volume.

        The pressure is evaluated halfway down the liquid column and stored as
        ``P_avg`` in atm for Henry-law saturation calculations.

        Returns
        -------
        None
            Updates ``P_avg`` in place.
        """
        H_L_current = self.V_current / 1000 / self.tank_crossarea  # Current height of liquid column (m)
        self.P_avg = self.density * 9.81 * (H_L_current / 2) / 101325 + self.P_operating  # Pa (kg/m³ * m/s² * m) to atm

    def _calculate_power_per_volume(self, N: float, V: float | None = None) -> float:
        """Return ungassed impeller power input per liquid volume.

        Parameters
        ----------
        N : float
            Agitation speed (rpm).
        V : float | None, optional
            Culture volume (L). Uses ``V_current`` when omitted.

        Returns
        -------
        float
            Power input per liquid volume (W/m³), calculated from the power
            number, medium density, impeller diameter, speed, and volume.
        """
        V_current = self.V_current if V is None else V
        V_m3 = V_current / 1000
        return self.P_ex_N * np.power(N / 60, 3) / V_m3

    def _calculate_superficial_gas_velocity(self, Fg: float, V: float | None = None) -> float:
        """Return superficial gas velocity for a sparging rate.

        Parameters
        ----------
        Fg : float
            Gas-flow rate (vvh; gas volume per liquid volume per hour).
        V : float | None, optional
            Culture volume (L). Uses ``V_current`` when omitted.

        Returns
        -------
        float
            Superficial gas velocity (m/s).
        """
        V_current = self.V_current if V is None else V
        V_m3 = V_current / 1000
        Fg_per_sec = Fg / 3600
        return Fg_per_sec * V_m3 / self.tank_crossarea

    def _update_hydrodynamics(self) -> None:
        """Update logged power-density and superficial-velocity attributes.

        Returns
        -------
        None
            Updates ``P_over_V_current`` and ``v_s_current`` in place.
        """
        self.P_over_V_current = self._calculate_power_per_volume(self.N_current)
        self.v_s_current = self._calculate_superficial_gas_velocity(self.Fg_current)

    @abstractmethod
    def control_DO(self, V: float, D: float, C_O2_in: float, C_O2: float, q_O2: float, Xv: float):  
        """Update the operating point needed to supply dissolved oxygen.

        Parameters
        ----------
        V : float
            Current culture volume (L).
        D : float
            Total inlet or dilution rate represented in the oxygen balance
            (1/h), including chemical-addition flow where applicable.
        C_O2_in : float
            Effective dissolved-oxygen concentration of incoming liquid
            (mmol/L).
        C_O2 : float
            Current dissolved-oxygen concentration (mmol/L).
        q_O2 : float
            Cell-specific oxygen uptake rate (mmol/cell/h).
        Xv : float
            Viable-cell concentration (cells/L).

        Returns
        -------
        None
            Implementations update their current OTR, ``kLa``, and gas
            operating-point attributes in place.
        """
        pass


class STRCascadeControllerModel(GasControllerModel):
    """Algebraic DO cascade controller for a stirred-tank reactor.

    The controller evaluates the OTR required by the oxygen balance at the DO
    setpoint.  It follows the cascade ``N → Fg → y_O2``: minimum agitation and
    gas flow are used first, agitation is increased to ``N_max``, gas flow is
    then increased to ``Fg_max``, and finally the inlet oxygen mole fraction is
    enriched to ``y_O2_max``.  If demand exceeds the attainable OTR, all three
    actuators are capped and the oxygen ODE is allowed to fall below setpoint.

    ``kLa`` is estimated using the non-coalescing-medium van 't Riet
    correlation.  CO2 transfer and pH regulation are handled by
    :class:`animakinetica.models.pH_controller.pHControllerModel`.

    Parameters
    ----------
    params : SimulationParameters
        Simulation configuration containing STR geometry, gas-control limits,
        operating pressure, medium density, the oxygen Henry constant, and the
        DO setpoint.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__(params)

        # Control limits for cascade control
        self.N_min = float(params.reactor.N_min)    # Minimum agitation speed (rpm)
        self.N_max = float(params.reactor.N_max)    # Maximum agitation speed (rpm)
        self.Fg_min = float(params.reactor.Fg_min)    # Minimum gas flow rate (vvh)
        self.Fg_max = float(params.reactor.Fg_max)    # Maximum gas flow rate (vvh)
        self.y_O2_min = float(params.reactor.y_O2_min)  # Minimum O2 mole fraction (air)
        self.y_O2_max = float(params.reactor.y_O2_max)   # Maximum O2 mole fraction (pure O2)

        # Initialize current operating conditions at minimums
        self.N_current = self.N_min
        self.Fg_current = self.Fg_min
        self.y_O2_current = self.y_O2_min
        
        # Pre-calculate power expression constant
        self.P_ex_N = self.density * self.power_number * np.power(self.d_impeller, 5) # Pre-calculate part of power expression for performance

        # Pre-calculate kLa boundaries for cascade control and set initial kLa
        self._calculate_klO2a_boundaries()  # Pre-calculate kLa at boundary conditions
        self.klO2a_current = self.klO2a_at_min_N_min_Fg

        # Update saturation concentrations
        self._update_saturation_concentrations() # Update saturation concentrations based on initial P_avg and y_O2 boundaries
        self._update_hydrodynamics()

    def _calculate_klO2a_boundaries(self):
        """Calculate cascade-boundary oxygen ``kLa`` values for current volume.

        Returns
        -------
        None
            Updates the four ``klO2a_at_*`` boundary attributes in place.
        """
        self.klO2a_at_min_N_min_Fg = self._calculate_klO2a_from_correlation(
            self.N_min, self.Fg_min)
        self.klO2a_at_max_N_max_Fg = self._calculate_klO2a_from_correlation(
            self.N_max, self.Fg_max)
        self.klO2a_at_max_N_min_Fg = self._calculate_klO2a_from_correlation(
            self.N_max, self.Fg_min)
        self.klO2a_at_min_N_max_Fg = self._calculate_klO2a_from_correlation(
            self.N_min, self.Fg_max)


    def _update_saturation_concentrations(self):
        """Update oxygen saturation concentrations at current pressure.

        Returns
        -------
        None
            Updates minimum, maximum, and current oxygen saturation
            concentrations (mmol/L) using Henry's law.
        """
        self.C_O2_sat_min = self.H_O2 * self.P_avg * self.y_O2_min  # Saturation concentration of O2 (mmol/L) at minimum O2 fraction
        self.C_O2_sat_max = self.H_O2 * self.P_avg * self.y_O2_max  # Max saturation concentration of O2 (mmol/L) at maximum O2 fraction
        self.C_O2_sat_current = self.H_O2 * self.P_avg * self.y_O2_current 

    def _calculate_klO2a_from_correlation(
        self, 
        N: float, 
        Fg: float,
    ) -> float:
        """Return oxygen ``kLa`` from the van 't Riet correlation.
        
        Parameters
        ----------
        N : float
            Agitation speed (rpm).
        Fg : float
            Sparging rate (vvh; gas volume per liquid volume per hour).
            
        Returns
        -------
        float
            Oxygen volumetric mass-transfer coefficient (1/h).
            
        Notes
        -----
        Reference: van 't Riet, K. (1979). "Review of measuring methods and
        results in nonviscous gas-liquid mass transfer in stirred vessels."
        *Industrial & Engineering Chemistry Process Design and Development*,
        18(3).

        Van't Riet correlation: kLa = C * (P/V)^α * v_s^β
        where P/V is power per volume and v_s is superficial gas velocity.
        For coalescing (low-ionic-strength) media,
        kLa_O2 [1/s] = 0.026 * (P/V)^0.4 * v_s^0.5. For non-coalescing
        (high-ionic-strength, animal-cell-culture) media, the implemented
        relation is kLa_O2 [1/s] = 0.002 * (P/V)^0.7 * v_s^0.2.

        The controller approximates P/V from N^3 * d^5 / V.
        """
        P_over_V = self._calculate_power_per_volume(N)
        v_s = self._calculate_superficial_gas_velocity(Fg)
        
        kLa_O2_per_sec = 0.002 * np.power(P_over_V, 0.7) * np.power(v_s, 0.2)  # Non-coalescing medium correlation
        kLa_O2 = kLa_O2_per_sec * 3600  # Convert to 1/h
    
        return kLa_O2

    def _calculate_N_from_kLa(self) -> float:
        """Return agitation speed required for ``klO2a_current`` at current gas flow.

        Returns
        -------
        float
            Required agitation speed (rpm), obtained by rearranging the van
            't Riet correlation. The caller is responsible for applying limits.
        """
        V = self.V_current
        klO2a_target = self.klO2a_current
        Fg = self.Fg_current
        # Rearranged van't Riet correlation (approximate)
        v_s = self._calculate_superficial_gas_velocity(Fg)
        P_over_V = np.power((klO2a_target / 3600) / (0.002 * np.power(v_s, 0.2)), 1/0.7) # W/m³
        V_m3 = V / 1000
        N = np.power(P_over_V * V_m3 / self.P_ex_N, 1/3) * 60 # rpm
        return N

    def _calculate_Fg_from_kLa(self) -> float:
        """Return gas flow required for ``klO2a_current`` at current agitation.

        Returns
        -------
        float
            Required sparging rate (vvh), obtained by rearranging the van
            't Riet correlation. The caller is responsible for applying limits.
        """
        V = self.V_current
        klO2a_target = self.klO2a_current
        N = self.N_current
        # Rearranged van't Riet correlation (approximate)
        P_over_V = self._calculate_power_per_volume(N, V=V)
        v_s = np.power((klO2a_target / 3600) / (np.power(P_over_V, 0.7) * 0.002), 5)  # m/s ; 5 = 1 / 0.2
        V_m3 = V / 1000
        Fg_per_sec = v_s * self.tank_crossarea / V_m3  # vvs (L/L/sec)
        Fg = Fg_per_sec * 3600  # Convert from L/L/sec to L/L/h
        return Fg

    def control_DO(self, V: float, D: float, C_O2_in: float, C_O2: float, q_O2: float, Xv: float            
        ):
        """Update the STR oxygen cascade for the current process state.

        The required transfer rate is calculated as
        ``D * (DO_setpoint - C_O2_in) + Xv * q_O2``.  At volumes that differ
        from the previously controlled volume, hydrostatic pressure, oxygen
        saturation concentrations, and cascade ``kLa`` boundaries are updated
        before selecting the operating point.

        Parameters
        ----------
        V : float
            Current culture volume (L).
        D : float
            Total inlet or dilution rate in the oxygen balance (1/h).
        C_O2_in : float
            Effective dissolved-oxygen concentration of incoming liquid
            (mmol/L).
        C_O2 : float
            Current dissolved-oxygen concentration (mmol/L).
        q_O2 : float
            Cell-specific oxygen uptake rate (mmol/cell/h).
        Xv : float
            Viable-cell concentration (cells/L).

        Returns
        -------
        None
            Updates ``OTR_current``, ``klO2a_current``, cascade actuators,
            and logged hydrodynamic attributes in place.
        """
        if V != self.V_current: # This means the operating volume is changing. This has consequences for P_avg
            self.V_current = V
            self._update_P_avg()  # Update average pressure based on current volume. Also updates saturation concentrations
            self._calculate_klO2a_boundaries()  # Update kLa boundaries based on current volume
            self._update_saturation_concentrations()

        # Use a balanced ODE to calculate OTR and CTR in the gas control model to maintain SP, but drop after max has been reached.
        # Use full, not simplified balances to have the same code be compatible with any operating mode.
        OTR_required = D * (-C_O2_in + self.DO_setpoint) + Xv * q_O2 #Calculates required OTR to maintain DO setpoint. OTR = In - out - consumption (negative q_O2)
        # Effective bulk DO cannot exceed the saturation concentration achievable at the
        # current cascade limit. Safetiy clamp to avoid a negative O2 transfer driving force when
        # the actual dissolved O2 is temporarily above that attainable saturation level.        
        C_O2_out_eff_min = min(C_O2, self.C_O2_sat_min)  
        C_O2_out_eff_max = min(C_O2, self.C_O2_sat_max)
        F_driving_min = self.C_O2_sat_min - C_O2_out_eff_min # Driving force available at the minimum oxygenation setting (min y_O2).
        F_driving_max = self.C_O2_sat_max - C_O2_out_eff_max # Driving force available at the maximum oxygenation setting (max y_O2).

        
        # Control cascade conditionals
        # First check if OTR can be met at minimum N, Fg, and y_O2. 
        if OTR_required <= self.klO2a_at_min_N_min_Fg * F_driving_min:
            # OTR can be met at minimum N and Fg
            self.N_current = self.N_min
            self.Fg_current = self.Fg_min
            self.y_O2_current = self.y_O2_min
            self.klO2a_current = self.klO2a_at_min_N_min_Fg
            self.OTR_current = OTR_required
        elif OTR_required <= self.klO2a_at_max_N_min_Fg * F_driving_min:
            # OTR can be met by increasing N only
            self.Fg_current = self.Fg_min
            self.y_O2_current = self.y_O2_min
            self.klO2a_current = OTR_required / F_driving_min # Find required kLa to meet OTR
            self.N_current = self._calculate_N_from_kLa() # Calculate required N to meet kLa for required OTR
            self.OTR_current = OTR_required
        elif OTR_required <= self.klO2a_at_max_N_max_Fg * F_driving_min:
            # OTR can be met by setting N_max and adjusting Fg
            self.N_current = self.N_max
            self.y_O2_current = self.y_O2_min
            self.klO2a_current = OTR_required / F_driving_min
            # First try increasing Fg at max N
            self.Fg_current = self._calculate_Fg_from_kLa()
            self.OTR_current = OTR_required
        elif OTR_required <= self.klO2a_at_max_N_max_Fg * F_driving_max:
            # OTR can be met by setting N_max, Fg_max and adjusting y_O2
            self.N_current = self.N_max
            self.Fg_current = self.Fg_max
            self.klO2a_current = self.klO2a_at_max_N_max_Fg
            self.C_O2_sat_current = OTR_required / self.klO2a_current + C_O2_out_eff_max
            self.y_O2_current = np.clip(
                self.C_O2_sat_current / (self.H_O2 * self.P_avg),
                self.y_O2_min,
                self.y_O2_max,
            )
            self.OTR_current = OTR_required
        elif np.isnan(OTR_required) or np.isinf(OTR_required):
            pass # keep previous values. This is likely due to zero cell density and zero q_O2
        else:
            # OTR requirement exceeds maximum capabilities. Cap at maximum and let DO drop.
            self.N_current = self.N_max
            self.Fg_current = self.Fg_max
            self.y_O2_current = self.y_O2_max
            self.C_O2_sat_current = self.C_O2_sat_max
            self.klO2a_current = self.klO2a_at_max_N_max_Fg
            self.OTR_current = self.klO2a_current * F_driving_max

        self._update_saturation_concentrations()
        self._update_hydrodynamics()
        # Catch any numerical issues in kLa calculation and keep previous value if so
        self.klO2a_current = self._calculate_klO2a_from_correlation(
            self.N_current, self.Fg_current
        )

def create_gas_controller_model(
    controller_type: GasController,
    params: SimulationParameters,
    reactor_type: str = "STR",  # Placeholder for reactor type variable. Only STR implemented currently.
) -> GasControllerModel:
    """Create the gas controller for a selected strategy and reactor type.

    Parameters
    ----------
    controller_type : GasController
        Requested gas-control strategy. Only
        :attr:`GasController.CASCADE_N_FG_XO2` is implemented.
    params : SimulationParameters
        Simulation configuration passed to the controller constructor.
    reactor_type : str, optional
        Reactor implementation. Only ``"STR"`` is supported; defaults to
        ``"STR"``.

    Returns
    -------
    GasControllerModel
        A configured :class:`STRCascadeControllerModel`.

    Raises
    ------
    NotImplementedError
        If ``CASCADE_N_FG`` is requested for an STR.
    ValueError
        If the controller/reactor combination is unsupported.
    """
    if controller_type == GasController.CASCADE_N_FG_XO2 and reactor_type == "STR":
        return STRCascadeControllerModel(params)
    elif controller_type == GasController.CASCADE_N_FG and reactor_type == "STR":
        # Placeholder for another controller model
        raise NotImplementedError("CASCADE_N_FG controller not implemented yet.")
    else:
        raise ValueError(f"Unknown gas controller type: {controller_type}")