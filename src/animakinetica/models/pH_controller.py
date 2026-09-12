"""Algebraic pH control, carbonate equilibrium, and buffer calculations.

:class:`pHControllerModel` calculates the base or acid dilution rate and
gas-phase CO2 fraction needed to regulate the configured pH setpoint. It
balances lactate and ammonia production, carbonate speciation and transfer,
medium dilution, and optional non-carbonate buffers. The controller exposes
these calculated values to the operating-mode ODE systems; it does not
integrate pH independently.
"""
import numpy as np
from typing import Optional
from animakinetica.core.parameters import SimulationParameters
from animakinetica.models.gas_controllers import GasControllerModel
from animakinetica.core.parameters import BufferDefinition

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = ["pHControllerModel"]


PKA_LACTATE = 3.86
PKA_CO2_1 = 6.35
PKA_CO2_2 = 10.33
PKA_AMMONIA = 9.25


class pHControllerModel():
    """Maintain a culture pH setpoint through algebraic chemical and gas control.

    The controller updates ``D_base``, ``D_acid``, ``CTR_current``, and
    ``y_CO2_current`` when :meth:`control_pH` is called. Carbonate species are
    calculated by :meth:`calculate_equilibrium`. Optional non-carbonate buffer
    totals are ODE states; their acid/base speciation is handled algebraically.
    Setpoint regulation is limited by the configured gas-phase CO2 and chemical
    addition bounds.
    """

    def __init__(self, params: SimulationParameters, gas_controller_model: GasControllerModel):
        """Initialize pH-control, carbonate, and buffer properties.

        Parameters
        ----------
        params : SimulationParameters
            Simulation parameters providing acid/base, gas-transfer, reactor,
            initial-condition, and optional buffer settings.
        gas_controller_model : GasControllerModel
            Gas controller supplying the current oxygen mass-transfer
            coefficient and average reactor pressure used to derive CO2
            transfer and saturation conditions.
        """
        # Get pH calculation constants
        self.pKa_lactate = PKA_LACTATE
        self.pKa_CO2_1 = PKA_CO2_1
        self.pKa_CO2_2 = PKA_CO2_2
        self.pKa_ammonia = PKA_AMMONIA
        self.acid_concentration = float(params.acid_base_properties.acid_concentration)
        self.base_concentration = float(params.acid_base_properties.base_concentration) 
        # Get gas controller model for CO2 stripping calculations
        self.gas_controller = gas_controller_model
        self.H_CO2 = float(params.gas_transfer.H_CO2)  # Henry's constant for CO2 (mmol/L/atm)
        self.D_L_CO2 = float(params.gas_transfer.D_L_CO2)  # Diffusivity of CO2 in water (cm2/s)
        self.D_L_O2 = float(params.gas_transfer.D_L_O2)  # Diffusivity of O2 in water (cm2/s)
        self._update_klCO2a_from_klO2a() # Needs to be called when kLO2a is changed.
        # pH control parameters
        self.C_H_setpoint = float(params.reactor.C_H_setpoint)  # H+ concentration at setpoint (mol/L)
        self.C_H_init = float(params.initial_conditions.C_H_init)  # Initialize current H+ concentration at setpoint (mol/L)
        self.C_H_current = self.C_H_init  # Current H+ concentration (mol/L)
        self.C_CT_init = float(params.initial_conditions.C_CT_init)  # Initial total dissolved carbon concentration (mmol/L)
        self.C_CT_current = self.C_CT_init  # Current total dissolved carbon concentration (mmol/L)
        self.C_Base = self.base_concentration  # Concentration of base solution (mol/L)
        self.C_OH_base = self.C_Base  # Assuming NaOH, OH- concentration equals base concentration (mol/L)
        self.C_Na_base = self.C_Base  # Assuming NaOH, Na+ concentration equals base concentration (mol/L)
        # Maximum base addition (dilution) rate for pH control (1/h). Defaults to
        # 5% of the reactor D_max when not configured (see SimulationParameters).
        _D_base_max = params.acid_base_properties.D_base_max
        self.D_base_max = float(_D_base_max) if _D_base_max is not None else np.inf
        self.y_CO2_min = float(params.reactor.y_CO2_min)  # Minimum mole fraction of CO2 in gas phase
        self.y_CO2_max = float(params.reactor.y_CO2_max)  # Maximum mole fraction of CO2 in gas phase
        self.y_CO2_current = self.y_CO2_min  # Initialize current CO2 mole fraction at minimum
        self._update_saturation_concentrations() # Update CO2 saturation concentrations based on P_avg and y_CO2 boundaries
        self.Ka_lactate = 10**(-self.pKa_lactate)
        self.Ka_CO2_1 = 10**(-self.pKa_CO2_1)
        self.Ka_CO2_2 = 10**(-self.pKa_CO2_2)
        self.Ka_ammonia = 10**(-self.pKa_ammonia)
        self.CTR_current = 0.0  # Current CO2 transfer rate (L/h). To be calcuted by control_pH method.
        self.D_base = 0.0  # Required base addition rate (h-1). To be calculated by control_pH method.
        self.dC_H_dt_from_Amm = 0.0  # H+ rate from ammonia production (mmol/L/h); set by control_pH
        self.V_current = float(params.reactor.V_batch_init)  # Current culture volume (L)

        # Acid addition parameters (symmetric to base, for alkalinization control)
        self.C_H_acid = self.acid_concentration  # H+ from fully dissociated strong acid (mol/L)
        self.C_Na_acid = self.acid_concentration  # Counterion from fully dissociated strong acid (mol/L)
        _D_acid_max = params.acid_base_properties.D_acid_max
        self.D_acid_max = float(_D_acid_max) if _D_acid_max is not None else self.D_base_max
        self.D_acid = 0.0  # Required acid addition rate (h-1). To be calculated by control_pH method.

        # Initialize equilibrium parameters
        self._update_alpha_L(self.C_H_current)
        self._update_alpha_ammonia(self.C_H_current)
        self._update_alpha_carbonate(self.C_H_current)
        self.calculate_equilibrium(self.C_CT_init, self.C_H_current)

        # --- Non-carbonate buffer system ---
        # Buffers are tracked as ODE state variables (one per buffer) whose total
        # concentration (HA + A-) changes only via feed input and dilution/washout.
        # Buffer "consumption" is a speciation shift (A- -> HA) handled algebraically
        # via the alpha fractions at the pH setpoint; total buffer mass is conserved.
        self.buffers: list = []
        self._buffer_Ka: list = []                  # acid dissociation constant Ka for each buffer
        self._buffer_alpha_base: list = []          # alpha_base at culture setpoint pH (frozen reference)
        self._buffer_alpha_base_feed: list = []     # alpha_base at feed medium pH (frozen reference)
        self._buffer_alpha_base_current: list = []  # alpha_base at current (possibly drifting) C_H
        self._buffer_C_init: list = []              # initial concentration (mmol/L), cached
        self._buffer_C_feed: list = []              # feed concentration (mmol/L), cached
        self.dC_H_dt_from_buffer: float = 0.0       # H+ rate from buffer re-equilibration (mmol/L/h)

        buf_defs = params.acid_base_properties.buffers
        if buf_defs:
            for buf in buf_defs:
                if not isinstance(buf, BufferDefinition):
                    continue
                Ka = 10.0 ** (-float(buf.pKa))
                C_init = float(buf.C_init)
                C_feed = float(buf.C_feed)

                # alpha_base = fraction in base (deprotonated) form = Ka/(Ka + C_H)
                alpha_culture = Ka / (Ka + self.C_H_setpoint)
                alpha_feed = Ka / (Ka + float(params.shared_medium_properties.C_H))

                self.buffers.append(buf)
                self._buffer_Ka.append(Ka)
                self._buffer_alpha_base.append(alpha_culture)
                self._buffer_alpha_base_feed.append(alpha_feed)
                self._buffer_alpha_base_current.append(alpha_culture)  # init at setpoint
                self._buffer_C_init.append(C_init)
                self._buffer_C_feed.append(C_feed)

        # Cache the number of non-carbonate buffer state variables (for ODE sizing)
        self.n_buffers: int = len(self.buffers)
    
    def _update_klCO2a_from_klO2a(self):
        """Update the CO2 volumetric mass-transfer coefficient.

        The estimate is ``kLCO2a = kLO2a * (D_L_CO2 / D_L_O2)**0.5``, where
        diffusivities are in cm²/s. The result is stored in
        ``kLCO2a_current`` in 1/h.
        """
        self.kLCO2a_current =  self.gas_controller.klO2a_current * np.power(self.D_L_CO2 / self.D_L_O2, 0.5) # Current CO2 volumetric mass transfer coefficient (h-1)

    def _update_saturation_concentrations(self):
        """Update CO2 saturation concentrations at the gas-fraction limits.

        Uses Henry's law and the current average pressure from
        :attr:`gas_controller` to update ``C_CO2_sat_min`` and
        ``C_CO2_sat_max`` in mmol/L.
        """
        self.C_CO2_sat_min = self.H_CO2 * self.gas_controller.P_avg * self.y_CO2_min  # Saturation concentration of CO2 (mmol/L)
        self.C_CO2_sat_max = self.H_CO2 * self.gas_controller.P_avg * self.y_CO2_max  # Max saturation concentration of CO2 (mmol/L)

    def _update_buffer_alphas(self, C_H: float):
        """Recompute non-carbonate buffer speciation fractions at the current C_H.

        Called whenever ``C_H`` changes through :meth:`calculate_equilibrium`
        or :meth:`reset_buffer_state`, so that the buffer alphas track the actual
        culture pH rather than remaining frozen at the setpoint.

        Parameters
        ----------
        C_H : float
            Current H+ concentration (mol/L).
        """
        for i, Ka in enumerate(self._buffer_Ka):
            self._buffer_alpha_base_current[i] = Ka / (Ka + C_H)

    def compute_buffer_capacity(self, C_buffer_list: list, C_H: float) -> float:
        """Return total non-carbonate buffer capacity β_total (dimensionless).

        For each buffer *i* with acid dissociation constant ``Ka_i`` and total
        concentration ``C_total,i`` (mmol/L), the buffer capacity is
        ``β_i = Ka_i * C_total,i / (Ka_i + [H+])**2`` after converting the
        concentration to mol/L.

        ``β_total`` is the sum over all configured buffers.  When pH drifts
        away from setpoint, the effective H+ rate seen by the ODE is reduced
        by a factor ``1 / (1 + β_total)`` — this is the classic buffering
        effect that resists pH change.

        Parameters
        ----------
        C_buffer_list : list[float]
            Current total concentrations of each buffer (mmol/L), from the ODE
            state vector.
        C_H : float
            Current H+ concentration (mol/L).

        Returns
        -------
        float
            Total buffer capacity β_total (dimensionless, ≥ 0).
        """
        if not self.buffers:
            return 0.0
        beta = 0.0
        for i, Ka in enumerate(self._buffer_Ka):
            C_total = C_buffer_list[i] if i < len(C_buffer_list) else 0.0
            beta += Ka * (C_total / 1000) / (Ka + C_H) ** 2
        return beta

    # Non-carbonate buffer methods
    # ------------------------------------------------------------------
    def compute_buffer_H_flux(self, C_buffer_list: list, D_medium: float) -> float:
        """Compute H⁺ production rate from buffer re-equilibration due to medium dilution.

        When medium at feed pH enters the culture (held at setpoint pH), the
        non-carbonate buffer species re-equilibrate, releasing or consuming H⁺.

        ``dH/dt = Σ (α_feed,i − α_current,i) * C_buffer_total,i * D_medium``.

        Parameters
        ----------
        C_buffer_list : list[float]
            Current total concentrations of each buffer (mmol/L), from the ODE
            state vector.
        D_medium : float
            Medium dilution rate (feed + perfusion, excluding base) in 1/h.

        Returns
        -------
        float
            H⁺ production rate in mmol/L/h (positive = acidification).
        """
        if not self.buffers or D_medium <= 0:
            return 0.0

        dH = 0.0
        for i in range(len(self.buffers)):
            delta_alpha = self._buffer_alpha_base_feed[i] - self._buffer_alpha_base_current[i]
            C_total = C_buffer_list[i]
            # delta_alpha > 0 → feed has MORE deprotonated form → absorbs H⁺ (−dH).
            dH += delta_alpha * C_total * D_medium
        return dH

    def get_buffer_osm_contribution(self, C_buffer_list: list) -> float:
        """Return total buffer osmolality contribution (mOsm).

        Both acid and base forms contribute equally to osmolality, so the total
        buffer concentration is the osmolar contribution.

        Parameters
        ----------
        C_buffer_list : list[float]
            Current total concentrations of each buffer (mmol/L), from the ODE
            state vector.

        Returns
        -------
        float
            Osmolality contribution in mOsm (sum of all buffer species).
        """
        if not self.buffers:
            return 0.0
        return sum(C_buffer_list)

    def reset_buffer_state(self, V_init: float):
        """Reset the controller's cached state for a new batch.

        Buffer total concentrations are ODE state variables (reset via the
        initial-conditions vector), so this method only resets the controller's
        cached scalar state that persists across batches. It also restores the
        carbonate equilibrium at the initial dissolved-carbon and H+ values.

        Parameters
        ----------
        V_init : float
            Initial culture volume for the new batch (L).
        """
        self.V_current = V_init
        self.C_H_current = self.C_H_init
        self.C_CT_current = self.C_CT_init
        self.D_base = 0.0
        self.D_acid = 0.0
        self.y_CO2_current = self.y_CO2_min
        self.CTR_current = 0.0
        self.dC_H_dt_from_buffer = 0.0
        self.dC_H_dt_from_Amm = 0.0
        # Re-derive equilibrium-dependent parameters at the reset pH
        self._update_alpha_L(self.C_H_current)
        self._update_alpha_ammonia(self.C_H_current)
        self._update_alpha_carbonate(self.C_H_current)
        self._update_buffer_alphas(self.C_H_current)
        self.calculate_equilibrium(self.C_CT_init, self.C_H_init)

    def _update_alpha_L(self, C_H: float):
        """Update the dissociated lactate fraction at the supplied H+ concentration.

        Parameters
        ----------
        C_H : float
            Hydrogen-ion concentration in mol/L.
        """
        self.alpha_L = self.Ka_lactate / (self.Ka_lactate + C_H)

    def _update_alpha_ammonia(self, C_H: float):
        """Update the unprotonated ammonia fraction at the supplied H+ concentration.

        ``alpha_NH3 = Ka / (Ka + C_H)`` is the fraction in the basic
        deprotonated form. Production of this fraction consumes H+.

        Parameters
        ----------
        C_H : float
            Hydrogen-ion concentration in mol/L.
        """
        self.alpha_NH3 = self.Ka_ammonia / (self.Ka_ammonia + C_H)

    def _update_alpha_carbonate(self, C_H: float):
        """Update carbonate-species fractions at the supplied H+ concentration.

        Parameters
        ----------
        C_H : float
            Hydrogen-ion concentration in mol/L.
        """
    
        denominator = C_H**2 + self.Ka_CO2_1 * C_H + self.Ka_CO2_1 * self.Ka_CO2_2
        
        # Fraction as H2CO3 (and dissolved CO2)
        self.alpha_H2CO3 = C_H**2 / denominator

        # Fraction as HCO3-
        self.alpha_HCO3 = (self.Ka_CO2_1 * C_H) / denominator
        
        # Fraction as CO3^2-
        self.alpha_CO3 = (self.Ka_CO2_1 * self.Ka_CO2_2) / denominator

        # Update alpha dependent parameters
        self._update_H_per_carbonate()

    def _update_H_per_carbonate(self):
        """Update proton equivalents associated with carbonate re-equilibration.

        The values ``H_per_CO2``, ``H_per_HCO3``, and ``H_per_CO3`` express
        the proton change per mmol/L change of the corresponding carbonate
        species at the current pH.
        """
        self.H_per_CO2 =  self.alpha_HCO3 * 1 + self.alpha_CO3 * 2
        self.H_per_HCO3 =  self.alpha_H2CO3 * -1 + self.alpha_CO3 * 1  # Some HCO3- will convert to CO2, consuming H+. Some will convert to CO3^2-, releasing H+.
        self.H_per_CO3 = self.alpha_H2CO3 * -2 + self.alpha_HCO3 * -1  # Some CO3^2- will convert to HCO3-, consuming H+. Some will convert to CO2, consuming 2 H+.


    def _calculate_dH_carbonate(self, dC_CO2_dt: float, dC_HCO3_dt: float, dC_CO3_dt: float):
        """Update the proton flux caused by carbonate-species changes.

        Changes from production, dilution, or gas transfer re-equilibrate at
        the current pH. The resulting flux is stored in
        ``dC_H_dt_from_carbonate``.

        Parameters
        ----------
        dC_CO2_dt : float
            Rate of change in dissolved CO2 concentration (mmol/L/h)
        dC_HCO3_dt : float
            Rate of change in bicarbonate concentration (mmol/L/h)
        dC_CO3_dt : float
            Rate of change in carbonate concentration (mmol/L/h)
            
        """
        
        # Calculate H+ change from each carbonate species change
        dH_from_CO2 = dC_CO2_dt * self.H_per_CO2
        dH_from_HCO3 = dC_HCO3_dt * self.H_per_HCO3
        dH_from_CO3 = dC_CO3_dt * self.H_per_CO3
        
        # Total H+ change from carbonate system
        dH_total = dH_from_CO2 + dH_from_HCO3 + dH_from_CO3
        
        self.dC_H_dt_from_carbonate = dH_total  


    def calculate_equilibrium(self, C_CT, C_H):
        """Update carbonate equilibrium for total dissolved carbon and H+ concentration.

        The method applies the two carbonate dissociation equilibria and
        updates ``C_H2CO3`` (including dissolved CO2), ``C_HCO3``, and
        ``C_CO3`` in mmol/L. H+ values within ``1e-10`` mol/L of the configured
        setpoint are treated as the setpoint to limit numerical drift.

        Parameters
        ----------
        C_CT : float
            Total dissolved carbon concentration (mmol/L)
        C_H : float
            H+ concentration (mol/L)
        """
        # Clamp C_H to setpoint for algebraic control consistency
        C_H_effective = self.C_H_setpoint if abs(C_H - self.C_H_setpoint) < 1e-10 else C_H

        if C_H_effective != self.C_H_current:
        #Update current pH based on H+ concentration
            self.C_H_current = C_H_effective
            self._update_alpha_L(self.C_H_current)
            self._update_alpha_ammonia(self.C_H_current)
            self._update_alpha_carbonate(self.C_H_current)
            self._update_buffer_alphas(self.C_H_current)
        
        self.C_CT_current = C_CT  # Total dissolved carbon concentration (mmol/L)
        # Calculate individual species concentrations
        self.C_H2CO3 = self.alpha_H2CO3 * C_CT  # H2CO3 + dissolved CO2 (mmol/L)
        self.C_HCO3 = self.alpha_HCO3 * C_CT  # Bicarbonate (mmol/L)
        self.C_CO3 = self.alpha_CO3 * C_CT  # Carbonate (mmol/L)


    def control_pH(
        self,
        V: float,
        D: float,
        C_H_in: float,
        C_CO2_in: float,
        C_HCO3_in: float,
        C_CO3_in: float,
        q_Lac: float,
        q_CO2: float,
        q_Amm: float,
        Xv: float,
        C_buffer_list: Optional[list] = None,
    ):
        """Update gas CO2, base, and acid control actions for the pH setpoint.

        The method first accounts for lactate, ammonia, carbonate, medium, and
        buffer proton fluxes. It then uses CO2 transfer within the configured
        gas-phase CO2 bounds before commanding chemical addition for any
        residual acidification or alkalinization. Results are retained in
        ``y_CO2_current``, ``CTR_current``, ``D_base``, and ``D_acid``.

        Parameters
        ----------
        V : float
            Current culture volume (L)
        D : float
            Dilution rate (h-1)
        C_H_in : float
            H+ concentration in inlet medium (mol/L)
        C_CO2_in : float
            Dissolved CO2/H2CO3 concentration in inlet medium (mmol/L)
        C_HCO3_in : float
            Bicarbonate concentration in inlet medium (mmol/L)
        C_CO3_in : float
            Carbonate concentration in inlet medium (mmol/L)
        q_Lac : float
            Specific lactate production rate (mmol/cell/h)
        q_CO2 : float
            Specific CO2 production rate (mmol/cell/h)
        q_Amm : float
            Specific ammonia production rate (mmol/cell/h)
        Xv : float
            Viable cell density in cells/L.
        C_buffer_list : list[float], optional
            Current total concentrations of each non-carbonate buffer (mmol/L),
            from the ODE state vector.  Required when buffers are configured.
            
        Notes
        -----
        This method updates controller attributes in place and returns
        ``None``. :class:`~animakinetica.models.odesystems.ODESystem` calls
        :meth:`calculate_equilibrium` before this method on each shared-model
        update.
        """
        # Update current volume and recalculate dependent parameters if volume has changed
        if V != self.V_current:  # Volume has changed
            self.V_current = V
            self._update_klCO2a_from_klO2a()  # Update klCO2a
            self._update_saturation_concentrations() # P_avg has been updated in the gas_controller. Update saturation concentrations.
        
        # Be sure to update pH and equilibrium by calling calculate_equilibrium before this method

        ## Carbonate calulations ##
        # New total carbon (C_CT) is produced/added to the system during this time step. This CO2 will equilibrate with the carbonate system, releasing H+ and lowering pH. Calculate the amount of CO2 added to the system (without gas-liquid transfer yet):
        dC_CO2_dt = Xv * q_CO2 + D * (C_CO2_in - self.C_H2CO3)  # Rate of change in dissolved CO2 concentration without CTR (mmol/L/h)
        dC_HCO3_dt = D * (C_HCO3_in - self.C_HCO3)  # Rate of change in bicarbonate concentration (mmol/L/h)
        dC_CO3_dt = D * (C_CO3_in - self.C_CO3)  # Rate of change in carbonate concentration (mmol/L/h)
        # The system equilibrates instantaneously. Calculate the rate of change of H+ due to the addition of these carbonates.
        self._calculate_dH_carbonate(dC_CO2_dt, dC_HCO3_dt, dC_CO3_dt) # Updates self.dC_H_dt_from_carbonate (mmol/L/h)
               
        # Calculate the H+ produced from lactate production
        self.dC_H_dt_from_Lac = self.alpha_L * Xv * q_Lac  # Rate of H+ production from lactate production (mmol/L/h)

        # Calculate the H+ consumed by ammonia production (ammonia is basic).
        # alpha_NH3 is the fraction produced as NH3 (deprotonated), which
        # consumes H+ (raises pH), so the term is negative.
        self.dC_H_dt_from_Amm = -self.alpha_NH3 * Xv * q_Amm  # Rate of H+ consumption from ammonia production (mmol/L/h)

        # H+ from non-carbonate buffer re-equilibration on medium dilution.
        # The D argument passed to control_pH is the total hydraulic turnover
        # (feed + base).  Buffer only enters via the medium feed, so recover
        # the medium-only dilution rate.  At this point self.D_base still
        # holds the previous timepoint value (not yet recomputed below).
        # Buffer total concentrations are ODE state variables, passed in
        # via C_buffer_list.
        D_medium = max(D - self.D_base, 0.0)
        if self.buffers:
            if C_buffer_list is None:
                C_buffer_list = [0.0] * len(self.buffers)
            self.dC_H_dt_from_buffer = self.compute_buffer_H_flux(C_buffer_list, D_medium)
        else:
            self.dC_H_dt_from_buffer = 0.0

        # Calculate the change in H+ concentration from all sources, excluding carbon stripping. Convert C_H from mol/L to mmol/L for unit consistency
        dC_H_dt = (C_H_in - self.C_H_current) * D * 1000 + self.dC_H_dt_from_Lac + self.dC_H_dt_from_Amm + self.dC_H_dt_from_carbonate + self.dC_H_dt_from_buffer  # (mmol/L/h)

        # Calculate the required CO2 transfer rate (CTR) to maintain pH setpoint
        CTR_required = -dC_H_dt / self.H_per_CO2 # (mmol/L/h); CTR should have the opposite sign because if C_H increases, pH decreases, CO2 should be removed to remove some H+, and vice versa

        # Convert the pH-driven CTR target into the gas-phase CO2 fraction needed to achieve it.
        # This allows total dissolved carbon to change dynamically instead of forcing dC_CT/dt >= 0.
        C_CO2_sat_required = CTR_required / self.kLCO2a_current + self.C_H2CO3 # (mmol/L)
        C_CO2_sat_required = max(C_CO2_sat_required, 0.0) # Saturation concentration clamped to 0 as it cannot be negative. This means that the C_H2CO3 concentration is too small for CO2 to be stripped.
        y_CO2_required = C_CO2_sat_required / (self.H_CO2 * self.gas_controller.P_avg) # (mole fraction)

        # Now there is three situations:
        # 1. The required y_CO2 is larger than the achievable range (max y_CO2). In this case, set y_CO2 to achieve this y_CO2_max, resulting in CTR = CTR_max. As y_CO2 is not in equilibrium with the dissolved CO2, CO2 will be evolved and pH will drop. pH will rise above setpoint. Acid needs to be added to counteract.
        # 2. The required y_CO2 is within the achievable range. In this case, set y_CO2 to achieve this y_CO2, resulting in CTR maintaining pH at setpoint.
        # 3. The required y_CO2 is below the achievable range. In this case, set y_CO2 to min. If the H2CO3 concentration becomes larger than saturation, CO2 is stripped from the medium. Any remaining H+ must be neutralized by base addition.
        # 3. is also reached when the H2CO3 concentration is so small that the required CO2 saturation concentration becomes negative. There cannot be any CO2 evolved from the medium, so the only way to maintain pH is to add base. 

        # Handle NaN or inf values (can occur with numerical issues but should not stop simulation)

        # 1.
        if y_CO2_required > self.y_CO2_max:
            self.y_CO2_current = self.y_CO2_max
            self.C_CO2_sat_current = self.C_CO2_sat_max  # Saturation concentration of CO2 (mmol/L)
            self.CTR_current = self.kLCO2a_current * (self.C_CO2_sat_current - self.C_H2CO3)  # Current CO2 transfer rate (mmol/L/h)
            self.D_base = 0.0  # No base addition required
            # Check if alkalinization persists after max CO2 enrichment.
            # residual < 0 means more CO2 is evolved than necessary as dissolved CO2 cannot be equilibrated with y_CO2. H+ removed --> pH will rise above setpoint. Acid needs to be added to counteract.
            residual = dC_H_dt + self.H_per_CO2 * self.CTR_current
            if residual < 0:
                self.D_acid = -residual / self.C_H_acid / 1000  # h-1; convert mmol/L/h → mol/L/h then / (mol/L)
                self.D_acid = min(self.D_acid, self.D_acid_max)
            else:
                self.D_acid = 0.0
        # 2.
        elif self.y_CO2_min <= y_CO2_required <= self.y_CO2_max:
            self.y_CO2_current = y_CO2_required
            self.C_CO2_sat_current = self.H_CO2 * self.gas_controller.P_avg * self.y_CO2_current
            self.CTR_current = CTR_required  # Current CO2 transfer rate (mmol/L/h)
            self.D_base = 0.0
            self.D_acid = 0.0
        # 3.
        elif y_CO2_required < self.y_CO2_min:
            self.y_CO2_current = self.y_CO2_min
            self.C_CO2_sat_current = self.C_CO2_sat_min
            self.CTR_current = self.kLCO2a_current * (self.C_CO2_sat_current - self.C_H2CO3)  # Current CO2 transfer rate (mmol/L/h)
            # Calculate the required amount of base to add to neutralize the extra H+
            self.D_base = (dC_H_dt + self.H_per_CO2 * self.CTR_current) / self.C_OH_base / 1000 # Required base addition rate (h-1)
            # Enforce maximum base addition rate; excess H+ is left unneutralized (pH deviates from setpoint)
            self.D_base = min(self.D_base, self.D_base_max)
        elif np.isnan(y_CO2_required) or np.isinf(y_CO2_required):
            pass # keep previous values
        else:  # Should not reach here
            raise ValueError(f"y_CO2_required calculation error in pHControllerModel.control_pH. Value: {y_CO2_required}")
        