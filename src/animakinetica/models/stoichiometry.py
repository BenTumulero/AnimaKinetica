"""Stoichiometric yield calculation and elemental-balance utilities.

:class:`StoichiometricModel` combines the configured anabolic reaction with a
catabolic reaction to calculate substrate and product yields. The model supports
constant, growth-rate-dependent, and excess-kinetics-coupled stoichiometry.
In :attr:`StoichiometryMode.EXCESS_COUPLED`, apparent lactate and glutamine
yields are converted into a CHON-balanced catabolic reaction at each update.

The model is created by
:class:`~animakinetica.core.simulation.CellCultureSimulation` and consumed by
the kinetic models through :meth:`StoichiometricModel.calculate_yields`.
"""

# Imports
from typing import Dict, Tuple, Optional
from enum import Enum
import numpy as np

from ..core.parameters import  SimulationParameters

__version__ = "1.0.0"

__all__ = ["StoichiometricModel", "StoichiometryMode"]


class StoichiometryMode(Enum):
    """Modes controlling the yield-calculation update policy.

    ``CONSTANT`` caches yields calculated at ``cell_properties.mu_max``.
    ``DYNAMIC`` recalculates them at each supplied growth rate. In
    ``EXCESS_COUPLED`` mode, catabolic stoichiometry is updated from apparent
    catabolic yields supplied to :meth:`StoichiometricModel.calculate_yields`.
    """
    CONSTANT      = "constant"       # Pre-calculated at mu_max, never updated during simulation
    DYNAMIC       = "dynamic"        # Recalculated each ODE step using the current growth rate
    EXCESS_COUPLED = "excess_coupled" # No catabolic file; stoichiometry back-calculated each ODE step from excess-model rates

# TODO: Future versions may automatically calculate the anabolic coefficients based on the biomass composition using the Humbird method.

# Standard formation enthalpies for the seven catabolic species [kJ/mol].
# Sources match the values used in the Humbird et al. (2020) Excel-based stoichiometry tool.
# Used to derive delta_Hf_catabolic when the yield-based catabolic mode is active.
_CATABOLIC_SPECIES_DHF: Dict[str, float] = {
    'Glc': -1262.19,   # Glucose (s)
    'O2':    -11.7,    # O2 (aq)
    'Gln':  -805.0,    # Glutamine (aq)
    'NH3':   -80.29,   # Ammonia (aq)
    'Lac':  -686.64,   # Lactate (aq)
    'CO2':  -413.8,    # CO2 (aq)
    'H2O':  -285.83,   # Water (l)
}

_ZERO_CATABOLIC_SPECIES = ('Glc', 'Gln', 'O2', 'NH3', 'Lac', 'CO2', 'H2O')
_YIELD_SPECIES = ('Glc', 'Gln', 'Lac', 'NH3', 'O2', 'CO2', 'DCM', 'Glu', 'H2O')

_ELEMENTS_CHON = ('C', 'H', 'O', 'N')
_SPECIES_FORMULAS_CHON: Dict[str, Dict[str, float]] = {
    'Glc': {'C': 6.0, 'H': 12.0, 'O': 6.0, 'N': 0.0},
    'Arg': {'C': 6.0, 'H': 14.0, 'O': 2.0, 'N': 4.0},
    'Cys': {'C': 3.0, 'H': 7.0, 'O': 2.0, 'N': 1.0},
    'Gln': {'C': 5.0, 'H': 10.0, 'O': 3.0, 'N': 2.0},
    'His': {'C': 6.0, 'H': 9.0, 'O': 2.0, 'N': 3.0},
    'Ile': {'C': 6.0, 'H': 13.0, 'O': 2.0, 'N': 1.0},
    'Lys': {'C': 6.0, 'H': 14.0, 'O': 2.0, 'N': 2.0},
    'Met': {'C': 5.0, 'H': 11.0, 'O': 2.0, 'N': 1.0},
    'Phe': {'C': 9.0, 'H': 11.0, 'O': 2.0, 'N': 1.0},
    'Thr': {'C': 4.0, 'H': 9.0, 'O': 3.0, 'N': 1.0},
    'Trp': {'C': 11.0, 'H': 12.0, 'O': 2.0, 'N': 2.0},
    'Tyr': {'C': 9.0, 'H': 11.0, 'O': 3.0, 'N': 1.0},
    'Val': {'C': 5.0, 'H': 11.0, 'O': 2.0, 'N': 1.0},
    'Ala': {'C': 3.0, 'H': 7.0, 'O': 2.0, 'N': 1.0},
    'Asn': {'C': 4.0, 'H': 8.0, 'O': 3.0, 'N': 2.0},
    'Asp': {'C': 4.0, 'H': 7.0, 'O': 4.0, 'N': 1.0},
    'Gly': {'C': 2.0, 'H': 5.0, 'O': 2.0, 'N': 1.0},
    'Leu': {'C': 6.0, 'H': 13.0, 'O': 2.0, 'N': 1.0},
    'Pro': {'C': 5.0, 'H': 9.0, 'O': 2.0, 'N': 1.0},
    'Ser': {'C': 3.0, 'H': 7.0, 'O': 3.0, 'N': 1.0},
    'Glu': {'C': 5.0, 'H': 9.0, 'O': 4.0, 'N': 1.0},
    'O2': {'C': 0.0, 'H': 0.0, 'O': 2.0, 'N': 0.0},
    'NH3': {'C': 0.0, 'H': 3.0, 'O': 0.0, 'N': 1.0},
    'Lac': {'C': 3.0, 'H': 6.0, 'O': 3.0, 'N': 0.0},
    'CO2': {'C': 1.0, 'H': 0.0, 'O': 2.0, 'N': 0.0},
    'H2O': {'C': 0.0, 'H': 2.0, 'O': 1.0, 'N': 0.0},
}
_REDUCED_BASIS_MW_G_PER_MOL: Dict[str, float] = {
    'Glc': 180.1572,
    'Gln': 146.1456,
    'O2': 31.998,
    'Lac': 90.078,
    'NH3': 17.031,
    'CO2': 44.01,
    'H2O': 18.01528,
}


class StoichiometricModel:
    """Calculate metabolic yield coefficients from anabolic and catabolic reactions.

    The model aligns reaction coefficients on a unified species basis, combines
    anabolic and maintenance requirements at a specified growth rate, and
    returns eight yield coefficients. In excess-coupled mode,
    :meth:`set_catabolic_yields` atom-balances the catabolic reaction before
    yield extraction.

    Attributes
    ----------
    species_list : list[str]
        Unified species names used to align the reaction arrays.
    anabolic_array, catabolic_array : numpy.ndarray
        Aligned anabolic and catabolic stoichiometric coefficients.
    yields_current : numpy.ndarray
        Most recently calculated eight-element yield vector.
    """
    
    def __init__(self, 
                 params: SimulationParameters,
                 mode: StoichiometryMode = StoichiometryMode.CONSTANT):
        """Initialize the model from configured reaction and cell properties.

        ``CONSTANT`` and ``DYNAMIC`` modes initialize catabolic stoichiometry
        from ``params.catabolic_reaction``. ``EXCESS_COUPLED`` initializes a
        zero-valued catabolic scaffold; its reaction is established by the
        first call to :meth:`calculate_yields` with ``cat_yields``.

        Parameters
        ----------
        params : SimulationParameters
            Parameters containing the anabolic reaction, cell properties, and,
            except in excess-coupled mode, catabolic-reaction data.
        mode : StoichiometryMode, default=StoichiometryMode.CONSTANT
            Policy controlling when catabolic requirements and yields update.
        """
        self.params = params
        self.mode = mode
        self.anabolic_dict = self._anabolic_to_dict()
        self.yields_current = np.zeros(8)

        # Per-yield scalar mirrors for lightweight logging access from ODE systems.
        self.Y_Glc_X_current = 0.0
        self.Y_Lac_Glc_current = 0.0
        self.Y_Gln_Glc_current = 0.0
        self.Y_O2_Glc_current = 0.0
        self.Y_CO2_Glc_current = 0.0
        self.Y_AA_Glc_current = 0.0
        self.Y_Amm_Gln_current = 0.0
        self.Y_H2O_X_current = 0.0

        # Cache metabolic power once — it depends only on cell properties which never
        # change during a simulation.  Used by _calculate_catabolic_extent().
        self._metabolic_power = self._calculate_metabolic_power()  # J h-1 Cmol-1

        if mode == StoichiometryMode.EXCESS_COUPLED:
            self._initialize_excess_coupled_state()
        else:
            self._initialize_standard_state()

        self._build_species_caches()

        if mode == StoichiometryMode.EXCESS_COUPLED:
            self.yields = np.zeros(8)
            self._set_current_yields(self.yields)
        else:
            self._refresh_cached_yields()

    def _set_current_yields(self, yields: np.ndarray) -> np.ndarray:
        """Store the latest yield vector and expose scalar aliases."""
        y = np.asarray(yields, dtype=float)
        if y.shape != (8,):
            y = np.zeros(8, dtype=float)

        self.yields_current = y
        self.Y_Glc_X_current = float(y[0])
        self.Y_Lac_Glc_current = float(y[1])
        self.Y_Gln_Glc_current = float(y[2])
        self.Y_O2_Glc_current = float(y[3])
        self.Y_CO2_Glc_current = float(y[4])
        self.Y_AA_Glc_current = float(y[5])
        self.Y_Amm_Gln_current = float(y[6])
        self.Y_H2O_X_current = float(y[7])
        return y

    def _initialize_excess_coupled_state(self) -> None:
        """Initialize EXCESS_COUPLED mode before the first back-calculation."""
        self.catabolic_dict = self._zero_catabolic_dict()
        self.cat_extent = 0.0

    def _initialize_standard_state(self) -> None:
        """Initialize CONSTANT/DYNAMIC catabolic state from parameter data."""
        self.catabolic_dict = self._catabolic_to_dict()
        self.cat_extent = self._calculate_catabolic_extent()

    def _build_species_caches(self) -> None:
        """Build the species-alignment and yield-extraction caches."""
        self._initialize_unified_arrays()
        self.AA_anabolic_coeff = self._calculate_AA_coefficient()
        self._cache_yield_indices()

    def _refresh_cached_yields(self) -> None:
        """Refresh the cached baseline yields at mu_max for CONSTANT/DYNAMIC modes."""
        self.yields = None
        self.yields = self.calculate_yields(mu=self.params.cell_properties.mu_max.value)
        self._set_current_yields(self.yields)

    def _fallback_yields(self) -> np.ndarray:
        """Return the last valid yields, or zeros if none have been cached yet."""
        fallback = self.yields if self.yields is not None else np.zeros(8)
        return self._set_current_yields(fallback)

    def _anabolic_to_dict(self) -> Dict[str, float]:
        """Return configured anabolic coefficients keyed by species.

        Returns
        -------
        dict[str, float]
            Anabolic stoichiometric coefficients, including biomass (``DCM``)
            and amino-acid species.
        """
        anabolic_params = self.params.anabolic_reaction
        return {
            'Glc': anabolic_params.Glc.value,
            'Arg': anabolic_params.Arg.value,
            'Cys': anabolic_params.Cys.value,
            'Gln': anabolic_params.Gln.value,
            'His': anabolic_params.His.value,
            'Ile': anabolic_params.Ile.value,
            'Lys': anabolic_params.Lys.value,
            'Met': anabolic_params.Met.value,
            'Phe': anabolic_params.Phe.value,
            'Thr': anabolic_params.Thr.value,
            'Trp': anabolic_params.Trp.value,
            'Tyr': anabolic_params.Tyr.value,
            'Val': anabolic_params.Val.value,
            'Ala': anabolic_params.Ala.value,
            'Asn': anabolic_params.Asn.value,
            'Asp': anabolic_params.Asp.value,
            'Gly': anabolic_params.Gly.value,
            'Leu': anabolic_params.Leu.value,
            'Pro': anabolic_params.Pro.value,
            'Ser': anabolic_params.Ser.value,
            'DCM': anabolic_params.DCM.value,
            'Glu': anabolic_params.Glu.value,
            'CO2': anabolic_params.CO2.value,
            'H2O': anabolic_params.H2O.value
        }

    def _catabolic_to_dict(self) -> Dict[str, float]:
        """Return configured or atom-balanced catabolic coefficients.

        Dispatches to the appropriate path depending on whether yield-based
        parameters (Y_Lac_Glc / Y_Gln_Glc) or explicit coefficients are present
        in the catabolic_reaction parameter section.

        Returns
        -------
        dict[str, float]
            Coefficients normalized to 1 mol glucose consumed. Negative values
            denote consumed species and positive values denote products.
        """
        cat = self.params.catabolic_reaction

        # Yield-based path: derive full stoichiometry from two degrees of freedom
        if cat.Y_Lac_Glc is not None and cat.Y_Gln_Glc is not None:
            return self._atom_balance_catabolic(cat.Y_Lac_Glc.value, cat.Y_Gln_Glc.value)

        # Explicit-coefficient path (existing behaviour)
        return {
            'Glc': cat.Glc.value,
            'Gln': cat.Gln.value,
            'O2':  cat.O2.value,
            'NH3': cat.NH3.value,
            'Lac': cat.Lac.value,
            'CO2': cat.CO2.value,
            'H2O': cat.H2O.value,
        }

    @staticmethod
    def _zero_catabolic_dict() -> Dict[str, float]:
        """Return a zero-valued catabolic stoichiometry scaffold.

        Used by EXCESS_COUPLED mode before the first back-calculation so the
        unified species arrays and cached indices include all catabolic species.
        """
        return {species: 0.0 for species in _ZERO_CATABOLIC_SPECIES}

    @staticmethod
    def _atom_balance_catabolic(Y_Lac_Glc: float, Y_Gln_Glc: float) -> Dict[str, float]:
        """
        Derive full catabolic stoichiometry from two apparent yield coefficients
        using elemental atom balances (C, H, O, N), normalised to 1 mol glucose.

        Balance equations (per mol Glc consumed):

        * N-balance: NH3 = 2 * Y_Gln/Glc
        * C-balance: CO2 = 6 + 5*Y_Gln/Glc - 3*Y_Lac/Glc
        * H-balance: H2O = (12 + 4*Y_Gln/Glc - 6*Y_Lac/Glc) / 2
        * O-balance: O2  = (3*Y_Lac/Glc + 2*CO2 + H2O - 6 - 3*Y_Gln/Glc) / 2

        Parameters
        ----------
        Y_Lac_Glc : float
            Apparent lactate yield on glucose in mol Lac/mol Glc.
        Y_Gln_Glc : float
            Apparent glutamine yield on glucose in mol Gln/mol Glc.

        Returns
        -------
        dict[str, float]
            Atom-balanced catabolic coefficients normalized to 1 mol glucose
            consumed. Negative values denote substrates.
        """
        # N-balance
        NH3 = 2.0 * Y_Gln_Glc
        # C-balance
        CO2 = 6.0 + 5.0 * Y_Gln_Glc - 3.0 * Y_Lac_Glc
        # H-balance
        H2O = (12.0 + 4.0 * Y_Gln_Glc - 6.0 * Y_Lac_Glc) / 2.0
        # O-balance
        O2 = (3.0 * Y_Lac_Glc + 2.0 * CO2 + H2O - 6.0 - 3.0 * Y_Gln_Glc) / 2.0

        return {
            'Glc': -1.0,
            'Gln': -Y_Gln_Glc,
            'O2':  -O2,
            'NH3':  NH3,
            'Lac':  Y_Lac_Glc,
            'CO2':  CO2,
            'H2O':  H2O,
        }

    @staticmethod
    def _calculate_dhf_from_stoich(catabolic_dict: Dict[str, float]) -> float:
        """
        Calculate the enthalpy of the catabolic reaction from its stoichiometry
        using the module-level standard formation enthalpies ``_CATABOLIC_SPECIES_DHF``.

        Parameters
        ----------
        catabolic_dict : dict[str, float]
            Coefficients normalized to 1 mol glucose, with negative values for
            consumed species.

        Returns
        -------
        float
            Catabolic reaction enthalpy in kJ/mol glucose.
        """
        return sum(
            nu * _CATABOLIC_SPECIES_DHF.get(species, 0.0)
            for species, nu in catabolic_dict.items()
        )


    def _initialize_unified_arrays(self):
        """Create a unified species basis and aligned reaction arrays.

        Missing species are represented by zero coefficients. Both
        ``anabolic_dict`` and ``catabolic_dict`` must be initialized before
        this method is called.
        """
        # Create unified species list (all unique species across both reactions)
        self.species_list = sorted(set(self.anabolic_dict.keys()) | set(self.catabolic_dict.keys()))
        
        # Create aligned arrays with zeros for missing species
        self.anabolic_array = np.array([self.anabolic_dict.get(s, 0.0) for s in self.species_list])
        self.catabolic_array = np.array([self.catabolic_dict.get(s, 0.0) for s in self.species_list])
        
        # Create species index mapping for fast lookups (used in calculate_yields)
        self._species_idx = {species: idx for idx, species in enumerate(self.species_list)}

        # Build index mapping from catabolic_dict keys into the unified species_list
        # so _update_catabolic_array_inplace() can write directly without rebuilding.
        self._catabolic_species_indices = {
            species: self._species_idx[species]
            for species in self.catabolic_dict
            if species in self._species_idx
        }

    def _update_catabolic_array_inplace(self) -> None:
        """Update ``catabolic_array`` from the current catabolic dictionary.

        Only the slots corresponding to catabolic species are overwritten.
        The species_list, anabolic_array, and _species_idx remain unchanged
        because the anabolic reaction never changes during a simulation.
        """
        if any(species not in self._catabolic_species_indices for species in self.catabolic_dict):
            self._build_species_caches()
            return

        for species, idx in self._catabolic_species_indices.items():
            self.catabolic_array[idx] = self.catabolic_dict.get(species, 0.0)

    def _cache_yield_indices(self) -> None:
        """Pre-compute numpy integer index array for the tracked yield species.

        Called once after _initialize_unified_arrays().  The indices are stable
        because the species_list never changes after initialisation.
        Stores a boolean mask for species that are present (always True in
        practice, but guards against missing species for safety).
        """
        raw = [self._species_idx.get(s) for s in _YIELD_SPECIES]
        # Build int index array + mask for None (species not in unified list)
        self._yield_indices = np.array([i if i is not None else 0 for i in raw], dtype=np.intp)
        self._yield_mask = np.array([i is not None for i in raw], dtype=bool)


    def _commit_catabolic_state(self) -> None:
        """Commit catabolic coefficients to the aligned array and extent cache."""
        self._update_catabolic_array_inplace()
        self.cat_extent = self._calculate_catabolic_extent()
    
    
    def _calculate_AA_coefficient(self) -> float:
        """Calculate the lumped non-glutamine amino-acid coefficient.

        The coefficient contains all configured amino acids except glutamine,
        which is tracked separately. Glutamate retains its stoichiometric sign,
        so glutamate production reduces the net requirement.

        Returns
        -------
        float
            Net amino-acid coefficient in mol AA/mol DCM.
        """
        amino_acids = ['Arg', 'Cys', 'His', 'Ile', 'Lys', 'Met', 'Phe', 'Thr', 'Trp', 'Tyr', 
                       'Val', 'Ala', 'Asn', 'Asp', 'Gly', 'Leu', 'Pro', 'Ser', 'Glu']
        
        AA_coeff = sum(self.anabolic_dict.get(aa, 0.0) for aa in amino_acids)
        return AA_coeff
    
    
    def _wests_relation(self, M_biomass: float) -> float:
        """Calculate cellular metabolic power using West's allometric relation.

        Parameters
        ----------
        M_biomass : float
            Wet biomass mass in pg/cell.

        Returns
        -------
        float
            Metabolic power in pW/cell.
        """
        return 0.148 * np.power(M_biomass, 0.75)  # West relation used in Humbird et al. (2020)
    

    def _calculate_metabolic_power(self) -> float:
        """Calculate metabolic power per C-mol of biomass.

        Returns
        -------
        float
            Metabolic power in J/(h Cmol). This is also the heat-release basis
            used by the culture model.
        """
        WCM = self.params.cell_properties.wet_cell_mass * 1e12  # g/cell to pg/cell
        DCM = self.params.cell_properties.dry_cell_mass * 1e12  # g/cell to pg/cell
        M_X = self.params.cell_properties.M_biomass # gDCW / Cmol
        metabolic_power = self._wests_relation(WCM)  # pW/cell
        metabolic_power = metabolic_power * 3600 / DCM  # pW / cell --> pJ h-1 cell-1 --> J h-1 gDCW-1
        metabolic_power = metabolic_power * M_X  # J h-1 Cmol-1
        return metabolic_power


    def _calculate_catabolic_extent(self) -> float:
        """Calculate catabolic extent from metabolic power and reaction enthalpy.

        In explicit-coefficient mode, delta_Hf_catabolic is read directly from the
        catabolic_reaction parameter.  In yield-based mode (Y_Lac_Glc / Y_Gln_Glc
        provided), it is derived on the fly from the atom-balanced stoichiometry
        using the hardcoded ``_CATABOLIC_SPECIES_DHF`` formation enthalpies.

        The calculation is ``cat_extent = P / (-delta_Hf_catabolic)`` after
        converting the enthalpy from kJ/mol to J/mol.

        Returns
        -------
        float
            Catabolic extent in mol/(h Cmol).
        """
        metabolic_power = self._metabolic_power  # J h-1 Cmol-1 (cached at init)

        cat = self.params.catabolic_reaction
        if cat is not None and cat.delta_Hf is not None:
            # Explicit-coefficient mode: use the value stored in the YAML
            delta_Hf_catabolic = cat.delta_Hf.value  # kJ/mol
        else:
            # Yield-based or EXCESS_COUPLED mode: derive from current atom-balanced stoichiometry
            delta_Hf_catabolic = self._calculate_dhf_from_stoich(self.catabolic_dict)  # kJ/mol

        delta_Hf_catabolic_J = delta_Hf_catabolic * 1000  # kJ/mol → J/mol
        cat_extent = metabolic_power / (-delta_Hf_catabolic_J)  # mol h-1 Cmol-1
        return cat_extent
    

    def _update_overall_reaction_array(self, mu: float) -> None:
        """Recompute ``overall_reaction_array`` at a specified growth rate.

        Used by the hot path (:meth:`calculate_yields`)
        to avoid a dict allocation on every ODE step.

        Parameters
        ----------
        mu : float
            Specific growth rate in 1/h. The caller must ensure it is positive.
        """
        cat_requirement = self.cat_extent / mu
        self.overall_reaction_array = self.anabolic_array + self.catabolic_array * cat_requirement
        self._overall_reaction_dict_dirty = True


    @property
    def overall_reaction_dict(self) -> Dict[str, float]:
        """Return a lazily built mapping for the latest overall reaction.

        Only constructed when explicitly accessed (e.g. for display / export),
        avoiding a dict allocation on every ODE step.

        Returns
        -------
        dict[str, float]
            Species coefficients from the most recently calculated reaction.
        """
        if getattr(self, '_overall_reaction_dict_dirty', True):
            self._overall_reaction_dict_cache = dict(zip(self.species_list, self.overall_reaction_array))
            self._overall_reaction_dict_dirty = False
        return self._overall_reaction_dict_cache

    def _get_reaction_coefficients(
        self,
        mu: float,
        cat_yields: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """Return overall-reaction coefficients at a specified growth rate.

        This diagnostic helper always evaluates the reaction at ``mu``, unlike
        :meth:`calculate_yields`, which returns a cache in constant mode.

        Parameters
        ----------
        mu : float
            Specific growth rate in 1/h. Must be positive.
        cat_yields : tuple[float, float], optional
            Apparent ``(Y_Lac_Glc, Y_Gln_Glc)`` values to commit before
            calculating the reaction.

        Returns
        -------
        dict[str, float]
            Overall stoichiometric coefficients on the current DCM basis.

        Raises
        ------
        ValueError
            If ``mu`` is not positive or supplied catabolic yields are invalid.
        """
        if mu <= 0:
            raise ValueError("Reaction closure requires mu > 0.")

        if cat_yields is not None:
            self.set_catabolic_yields(cat_yields[0], cat_yields[1])

        self._update_overall_reaction_array(mu)
        return dict(self.overall_reaction_dict)

    def get_biomass_chon_formula(self) -> Dict[str, float]:
        """Return the configured biomass CHON formula on the DCM basis.

        Returns
        -------
        dict[str, float]
            Elemental coefficients keyed by ``C``, ``H``, ``O``, and ``N``.

        Raises
        ------
        ValueError
            If the configured biomass carbon coefficient is zero.
        """
        if hasattr(self, '_biomass_chon_formula_cache'):
            return dict(self._biomass_chon_formula_cache)

        formula = {
            'C': float(self.params.cell_properties.biomass_C),
            'H': float(self.params.cell_properties.biomass_H),
            'O': float(self.params.cell_properties.biomass_O),
            'N': float(self.params.cell_properties.biomass_N),
        }
        if formula['C'] == 0:
            raise ValueError("Configured biomass CHON formula must have a non-zero carbon basis.")

        self._biomass_chon_formula_cache = dict(formula)
        return dict(self._biomass_chon_formula_cache)

    def calculate_elemental_closure(
        self,
        mu: float,
        cat_yields: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate CHON elemental closure for an overall reaction.

        Parameters
        ----------
        mu : float
            Specific growth rate in 1/h. Must be positive.
        cat_yields : tuple[float, float], optional
            Apparent ``(Y_Lac_Glc, Y_Gln_Glc)`` values to commit before the
            closure calculation.

        Returns
        -------
        dict[str, object]
            Closure report containing the DCM basis, growth rate, biomass CHON
            formula, per-element reactant/product totals and residuals, and
            ``max_abs_residual``.

        Raises
        ------
        ValueError
            If the reaction cannot be evaluated or a required CHON species
            formula is unavailable.
        """
        reaction = self._get_reaction_coefficients(mu, cat_yields=cat_yields)
        dcm_formula = self.get_biomass_chon_formula()

        elemental_balance: Dict[str, Dict[str, float]] = {}
        for element in _ELEMENTS_CHON:
            reactants = 0.0
            products = 0.0
            for species, coeff in reaction.items():
                species_formula = dcm_formula if species == 'DCM' else _SPECIES_FORMULAS_CHON.get(species)
                if species_formula is None:
                    raise ValueError(f"No CHON formula is defined for species '{species}'.")

                atoms = species_formula.get(element, 0.0)
                if coeff < 0:
                    reactants += -coeff * atoms
                elif coeff > 0:
                    products += coeff * atoms

            elemental_balance[element] = {
                'reactants': reactants,
                'products': products,
                'residual': products - reactants,
            }

        max_abs_residual = max(abs(item['residual']) for item in elemental_balance.values())
        return {
            'basis': 'per_Cmol_DCM',
            'mu': mu,
            'dcm_chon_formula': dcm_formula,
            'elements': elemental_balance,
            'max_abs_residual': max_abs_residual,
        }

    @staticmethod
    def _extract_yields_from_coeffs(coeffs: np.ndarray, abs_AA_coeff: float) -> np.ndarray:
        """Convert tracked stoichiometric coefficients into yield values.

        Pure helper with no attribute access, so it adds minimal overhead per call.

        Parameters
        ----------
        coeffs : numpy.ndarray
            Nine coefficients ordered as ``[Glc, Gln, Lac, NH3, O2, CO2, DCM,
            Glu, H2O]``.
        abs_AA_coeff : float
            Absolute precomputed non-glutamine amino-acid coefficient.

        Returns
        -------
        numpy.ndarray
            Eight yield coefficients in the order returned by
            :meth:`calculate_yields`.
        """
        Glc_coeff, Gln_coeff, Lac_coeff, NH3_coeff, O2_coeff, CO2_coeff, DCM_coeff, Glu_coeff, H2O_coeff = coeffs

        abs_Glc = np.abs(Glc_coeff)
        abs_Gln = np.abs(Gln_coeff)

        yields = np.zeros(8)

        if DCM_coeff != 0:
            yields[0] = abs_Glc / DCM_coeff
            yields[7] = H2O_coeff / DCM_coeff

        if abs_Glc != 0:
            inv_abs_Glc = 1.0 / abs_Glc
            yields[1] = Lac_coeff * inv_abs_Glc
            yields[2] = abs_Gln * inv_abs_Glc
            yields[3] = np.abs(O2_coeff) * inv_abs_Glc
            yields[4] = CO2_coeff * inv_abs_Glc
            yields[5] = abs_AA_coeff * inv_abs_Glc

        if abs_Gln != 0:
            yields[6] = NH3_coeff / abs_Gln

        return yields

    def _fast_extract_coeffs(self, reaction_array: np.ndarray) -> np.ndarray:
        """Extract tracked yield-species coefficients using cached indices.

        This replaces per-call dictionary lookups with a NumPy fancy-index
        operation.

        Parameters
        ----------
        reaction_array : numpy.ndarray
            Aligned overall-reaction coefficient array.

        Returns
        -------
        numpy.ndarray
            Nine coefficients ordered according to the internal tracked-yield
            species list.
        """
        coeffs = reaction_array[self._yield_indices]
        # Zero-out any species that were absent from the unified species list
        coeffs[~self._yield_mask] = 0.0
        return coeffs

    def calculate_yields(self, mu: float, cat_yields: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Calculate yield coefficients from the overall reaction at ``mu``.

        In constant mode, the cached yield vector calculated at ``mu_max`` is
        returned unless ``cat_yields`` is supplied. Dynamic and excess-coupled
        modes recalculate the overall reaction at the requested growth rate.
        A non-positive ``mu`` returns the last valid yield vector to avoid the
        undefined maintenance-to-growth ratio.

        Parameters
        ----------
        mu : float
            Specific growth rate in 1/h.
        cat_yields : tuple[float, float], optional
            Apparent ``(Y_Lac_Glc, Y_Gln_Glc)`` values in mol/mol. When
            supplied, the values are committed through
            :meth:`set_catabolic_yields` before calculating the yield vector.

        Returns
        -------
        numpy.ndarray
            Eight yields, ordered as ``[Y_Glc_X, Y_Lac_Glc, Y_Gln_Glc,
            Y_O2_Glc, Y_CO2_Glc, Y_AA_Glc, Y_Amm_Gln, Y_H2O_X]``. Their units
            are mol/mol, with the denominators indicated by each name.

        Raises
        ------
        ValueError
            If supplied apparent catabolic yields yield oxygen production or
            are unsupported by the loaded catabolic parameterization.
        """
        
        # Apply optional catabolic-yield override first so state is updated even
        # in CONSTANT mode (used by sweep/debug workflows).
        if cat_yields is not None:
            self.set_catabolic_yields(cat_yields[0], cat_yields[1])

        # CONSTANT mode: return pre-cached yields immediately when no override
        # was requested for this call.
        if self.mode == StoichiometryMode.CONSTANT and self.yields is not None and cat_yields is None:
            return self._set_current_yields(self.yields)

        # DYNAMIC mode guard: cat_requirement = cat_extent / mu would blow up at mu <= 0
        # (e.g. during lag phase or near washout). Return last valid yields as fallback.
        if mu <= 0:
            return self._fallback_yields()

        # Update overall_reaction_array at given growth rate and optional new catabolic yields (array only, no dict)
        self._update_overall_reaction_array(mu)
        
        # Extract tracked yield-species coefficients via cached numpy index array
        coeffs = self._fast_extract_coeffs(self.overall_reaction_array)

        # AA_Anabolic coeff is the net lumped amino-acid term excluding only
        # glutamine; only water is surfaced from the non-state pseudo products.
        yields = self._extract_yields_from_coeffs(coeffs, np.abs(self.AA_anabolic_coeff))
        self._set_current_yields(yields)
        return self.yields_current

    def set_catabolic_yields(self, Y_Lac_Glc: float, Y_Gln_Glc: float) -> None:
        """Commit apparent catabolic yields and refresh derived stoichiometry.

        The method atom-balances lactate and glutamine yields to update the
        catabolic reaction, aligned coefficient array, and catabolic extent.
        In constant mode it also refreshes the cached yield vector. For modes
        other than :attr:`StoichiometryMode.EXCESS_COUPLED`, this updates the
        yield fields in ``params.catabolic_reaction``.

        Parameters
        ----------
        Y_Lac_Glc : float
            Apparent lactate yield on glucose in mol Lac/mol Glc.
        Y_Gln_Glc : float
            Apparent glutamine yield on glucose in mol Gln/mol Glc.

        Raises
        ------
        ValueError
            If the apparent yields imply oxygen production, or if the loaded
            non-excess catabolic parameterization lacks yield-based fields.
        """
        # Validate physical stoichiometry before committing (O2 must be consumed, not produced)
        proposed_catabolic_dict = self._atom_balance_catabolic(Y_Lac_Glc, Y_Gln_Glc)
        if proposed_catabolic_dict['O2'] > 0:
            raise ValueError(
                f"Unphysical catabolic stoichiometry: O2 coefficient = {proposed_catabolic_dict['O2']:.4f} > 0 "
                f"(O2 would be produced rather than consumed) "
                f"for Y_Lac_Glc={Y_Lac_Glc}, Y_Gln_Glc={Y_Gln_Glc}.  "
                "Reduce Y_Lac_Glc or Y_Gln_Glc."
            )

        if self.mode == StoichiometryMode.EXCESS_COUPLED:
            # EXCESS_COUPLED: use the atom-balanced dict directly; no params to write back
            self.catabolic_dict = proposed_catabolic_dict
        else:
            cat = self.params.catabolic_reaction
            if cat.Y_Lac_Glc is None or cat.Y_Gln_Glc is None:
                raise ValueError(
                    "set_catabolic_yields() requires the yield-based catabolic configuration. "
                    "Load a catabolic YAML that provides Y_Lac_Glc and Y_Gln_Glc fields."
                )
            # Commit new yield values to the parameter objects
            cat.Y_Lac_Glc.value = Y_Lac_Glc
            cat.Y_Gln_Glc.value = Y_Gln_Glc
            self.catabolic_dict = self._catabolic_to_dict()

        self._commit_catabolic_state()

        # Keep CONSTANT-mode cache coherent after runtime updates.
        if self.mode == StoichiometryMode.CONSTANT:
            self._refresh_cached_yields()