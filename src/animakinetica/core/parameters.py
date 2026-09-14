"""Parameter definitions, YAML I/O, and derived simulation configuration.

YAML parameter sections are represented by dataclasses. Individual scalar
values are stored as :class:`Parameter` objects, preserving their display name
and unit while supporting numeric operations. :class:`SimulationParameters`
loads and merges one or more YAML files, then calculates dependent values such
as ``cell_properties.mu_max``, reactor working volumes, and initial feed flow.

Use :meth:`SimulationParameters.load_parameters` to load a complete or modular
configuration. Retrieve a :class:`Parameter` with ``params["reactor.V_max"]``
or its raw value with
``params.get_parameter_value("reactor.V_max")``. For independent parameter
variations, use :meth:`SimulationParameters.create_modified_parameter_object`.
"""
# Imports
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, fields
from pathlib import Path
from datetime import datetime
from time import sleep
import numpy as np
import yaml
import copy

__version__ = "1.0.0"

__all__ = [
    "Parameter",
    "ParameterValidationError",
    "ParameterIO",
    "CellProperties",
    "AnabolicReaction",
    "CatabolicReaction",
    "SharedMediumProperties",
    "BufferDefinition",
    "AcidBaseProperties",
    "KineticsParameters",
    "ExcessKineticsParameters",
    "ReactorParameters",
    "InitialConditions",
    "FedBatchParameters",
    "ChemostatParameters",
    "PerfusionParameters",
    "ProcessSchedule",
    "GasTransferConstants",
    "SimulationParameters",
    "list_parameter_sets",
]


# Module level constants
CONFIG_DIR = Path(__file__).parent.parent / "config"
PARAMETER_SETS_DIR = CONFIG_DIR / "parameter_sets"
BASE_PARAMS_FILE = PARAMETER_SETS_DIR / "base_parameters.yaml"
SCHEMA_FILE = CONFIG_DIR / "schemas" / "parameter_schema.yaml"

# Create directories if they don't exist
PARAMETER_SETS_DIR.mkdir(parents=True, exist_ok=True)

# Module functions
def list_parameter_sets() -> List[str]:
    """Return names of YAML parameter sets directly in the default directory.

    Returns
    -------
    list of str
        Filename stems for ``*.yaml`` files in :data:`PARAMETER_SETS_DIR`.
        Subdirectories are not searched.
    """
    if not PARAMETER_SETS_DIR.exists():
        print(f"No parameter sets directory found at {PARAMETER_SETS_DIR}")
        return []
    
    yaml_files = list(PARAMETER_SETS_DIR.glob("*.yaml"))
    if not yaml_files:
        print("No parameter set files found.\n")
        return []
    
    return [f.stem for f in yaml_files]

@dataclass
class Parameter:
    """Numeric value with a human-readable label and unit.

    Parameters
    ----------
    name : str
        Display name of the parameter.
    value : Any
        Stored value. Arithmetic operators assume it can be converted to
        ``float``.
    unit : str
        Display unit. Use ``"-"`` for dimensionless values.

    Notes
    -----
    Arithmetic and comparisons operate on ``value`` and return numeric results.
    Use ``float(parameter)`` explicitly when passing a value to NumPy or SciPy.
    """
    name: str
    value: Any
    unit: str
    
    def __float__(self):
        """Return the stored value as ``float``."""
        return float(self.value)
    
    def __int__(self):
        """Return the stored value as ``int``."""
        return int(self.value)
    
    def __repr__(self):
        """String representation"""
        unit_str = f" {self.unit}" if self.unit != "-" else ""
        return f"{self.value}{unit_str}"
    
    def __str__(self):
        return self.__repr__()
    
    def __format__(self, format_spec):
        """Handle string formatting operations like f'{param:.2f}'"""
        if format_spec:
            # Apply the format specification to the value
            return format(float(self.value), format_spec)
        else:
            # No format spec, return string representation
            return str(self)

    # Add arithmetic operators so Parameter objects can be used in calculations
    def __add__(self, other):
        """Addition operator"""
        if isinstance(other, Parameter):
            return float(self.value) + float(other.value)
        return float(self.value) + other
    
    def __radd__(self, other):
        """Reverse addition"""
        return other + float(self.value)
    
    def __sub__(self, other):
        """Subtraction operator"""
        if isinstance(other, Parameter):
            return float(self.value) - float(other.value)
        return float(self.value) - other
    
    def __rsub__(self, other):
        """Reverse subtraction"""
        return other - float(self.value)
    
    def __mul__(self, other):
        """Multiplication operator"""
        if isinstance(other, Parameter):
            return float(self.value) * float(other.value)
        return float(self.value) * other
    
    def __rmul__(self, other):
        """Reverse multiplication"""
        return other * float(self.value)
    
    def __truediv__(self, other):
        """Division operator"""
        if isinstance(other, Parameter):
            return float(self.value) / float(other.value)
        return float(self.value) / other
    
    def __rtruediv__(self, other):
        """Reverse division"""
        return other / float(self.value)
    
    def __pow__(self, other):
        """Power operator"""
        if isinstance(other, Parameter):
            return float(self.value) ** float(other.value)
        return float(self.value) ** other
    
    def __rpow__(self, other):
        """Reverse power"""
        return other ** float(self.value)
    
    # Add comparison operators
    def __eq__(self, other):
        """Equality operator"""
        if isinstance(other, Parameter):
            return float(self.value) == float(other.value)
        return float(self.value) == other
    
    def __ne__(self, other):
        """Not equal operator"""
        return not self.__eq__(other)
    
    def __lt__(self, other):
        """Less than operator"""
        if isinstance(other, Parameter):
            return float(self.value) < float(other.value)
        return float(self.value) < other
    
    def __le__(self, other):
        """Less than or equal operator"""
        if isinstance(other, Parameter):
            return float(self.value) <= float(other.value)
        return float(self.value) <= other
    
    def __gt__(self, other):
        """Greater than operator"""
        if isinstance(other, Parameter):
            return float(self.value) > float(other.value)
        return float(self.value) > other
    
    def __ge__(self, other):
        """Greater than or equal operator"""
        if isinstance(other, Parameter):
            return float(self.value) >= float(other.value)
        return float(self.value) >= other
    
    # Add reverse comparison operators for when Parameter is on the right side
    def __rlt__(self, other):
        """Reverse less than"""
        return other < float(self.value)
    
    def __rle__(self, other):
        """Reverse less than or equal"""
        return other <= float(self.value)
    
    def __rgt__(self, other):
        """Reverse greater than"""
        return other > float(self.value)
    
    def __rge__(self, other):
        """Reverse greater than or equal"""
        return other >= float(self.value)

    # Add unary operators
    def __neg__(self):
        """Unary negation operator"""
        return -float(self.value)

    def __pos__(self):
        """Unary positive operator"""
        return float(self.value)

    def __abs__(self):
        """Absolute value operator"""
        return abs(float(self.value))

    @property
    def full_label(self) -> str:
        """Get full label with name and unit"""
        unit_str = f"({self.unit})" if self.unit != "-" else ""
        return f"{self.name} {unit_str}".strip()


class ParameterValidationError(Exception):
    """Raised when parsing or validation of parameter data fails."""
    pass


class ParameterIO:
    """Read, transform, validate, and write structured parameter YAML.

    The class provides stateless helpers. YAML scalar entries may be raw values
    or mappings with ``name``, ``value``, and ``unit`` keys.
    """
    
    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML mapping from a file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            YAML file to load.

        Returns
        -------
        dict of str to Any
            Parsed YAML mapping.

        Raises
        ------
        FileNotFoundError
            If ``filepath`` does not exist.
        ParameterValidationError
            If the YAML content cannot be parsed.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Parameter file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ParameterValidationError(f"Error parsing YAML file: {e}")
        
    @staticmethod
    def create_parameter(param_data: Union[Dict, Any], param_name: str) -> Parameter:
        """Create a :class:`Parameter` from a YAML value or mapping.

        Parameters
        ----------
        param_data : dict or Any
            Mapping containing ``value`` and optional ``name`` and ``unit``, or
            a legacy raw value.
        param_name : str
            Fallback name used when ``param_data`` does not define ``name``.

        Returns
        -------
        Parameter
            Converted parameter. Legacy raw values receive unit ``"-"``.
        """
        if isinstance(param_data, dict) and 'value' in param_data:
            return Parameter(
                name=param_data.get('name', param_name.replace('_', ' ').title()),
                value=param_data['value'],
                unit=param_data.get('unit', '-')
            )
        else:
            # Backwards compatibility: treat as raw value
            return Parameter(
                name=param_name.replace('_', ' ').title(),
                value=param_data,
                unit='-'
            )

    @staticmethod
    def extract_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert scalar YAML entries in each section to :class:`Parameter`.

        Parameters
        ----------
        data : dict of str to Any
            Structured YAML data.

        Returns
        -------
        dict of str to Any
            Data with scalar section entries converted to ``Parameter`` objects.
            ``metadata`` and lists are preserved for later conversion.
        """
        result = {}

        for section_name, section_data in data.items():
            if section_name == 'metadata':
                result[section_name] = section_data
                continue

            result[section_name] = {}
            for param_name, param_data in section_data.items():
                # Lists (e.g. acid_base_properties.buffers) are kept as-is;
                # they are converted to their target type by the caller.
                if isinstance(param_data, list):
                    result[section_name][param_name] = param_data
                else:
                    result[section_name][param_name] = ParameterIO.create_parameter(
                        param_data, param_name
                    )

        return result
    
    
    @staticmethod
    def convert_to_structured_format(params: 'SimulationParameters') -> Dict[str, Any]:
        """Convert a parameter object to the structured YAML representation.

        Parameters
        ----------
        params : SimulationParameters
            Parameter configuration to serialize.

        Returns
        -------
        dict of str to Any
            YAML-compatible data with ``name``, ``value``, and ``unit`` entries.
        """
        def convert_value(value):
            """Convert numpy types to native Python types"""
            if isinstance(value, np.generic):
                return value.item()
            elif isinstance(value, np.ndarray):
                return value.tolist()
            return value

        result = {'metadata': params.metadata.copy()}
        
        # Dynamically get all sections from SimulationParameters fields
        for field_info in fields(params):
            section_name = field_info.name
            
            # Skip metadata and None values
            if section_name == 'metadata':
                continue
            
            section_obj = getattr(params, section_name)
            if section_obj is None:
                continue
            
            # Get only fields that are part of __init__ (init=True)
            init_fields = {f.name for f in fields(section_obj) if f.init}
            
            result[section_name] = {}
            for param_name in dir(section_obj):
                if not param_name.startswith('_') and param_name in init_fields:
                    param = getattr(section_obj, param_name)
                    if isinstance(param, Parameter):
                        result[section_name][param_name] = {
                            'name': param.name,
                            'value': convert_value(param.value),
                            'unit': param.unit
                        }
                    elif isinstance(param, list):
                        # Serialise list of BufferDefinition objects
                        buffer_list = []
                        for entry in param:
                            if isinstance(entry, BufferDefinition):
                                buffer_list.append({
                                    'name': entry.name,
                                    'pKa': {
                                        'name': entry.pKa.name,
                                        'value': convert_value(entry.pKa.value),
                                        'unit': entry.pKa.unit,
                                    },
                                    'C_init': {
                                        'name': entry.C_init.name,
                                        'value': convert_value(entry.C_init.value),
                                        'unit': entry.C_init.unit,
                                    },
                                    'C_feed': {
                                        'name': entry.C_feed.name,
                                        'value': convert_value(entry.C_feed.value),
                                        'unit': entry.C_feed.unit,
                                    },
                                    'type': entry.type,
                                })
                        result[section_name][param_name] = buffer_list
        return result
    

    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: Union[str, Path], 
                  overwrite: bool = False):
        """Write YAML data to a file.

        Parameters
        ----------
        data : dict of str to Any
            YAML-compatible data to write.
        filepath : str or pathlib.Path
            Destination file. Parent directories are created when needed.
        overwrite : bool, default=False
            Whether to replace an existing file.

        Raises
        ------
        FileExistsError
            If the destination exists and ``overwrite`` is ``False``.
        FileNotFoundError
            If the destination parent repeatedly disappears while the file is
            being created.
        """
        filepath = Path(filepath)
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}. Use overwrite=True to replace.")

        # Cloud-sync clients can briefly remove or replace a newly-created
        # directory between mkdir() and open() on Windows. This is particularly
        # likely immediately after a companion Excel workbook and pickle are
        # written to the same timestamped export directory. Recreate its
        # parent and retry long enough for the directory to stabilize.
        max_attempts = 8
        for attempt in range(max_attempts):
            filepath.parent.mkdir(parents=True, exist_ok=True)
            try:
                with filepath.open('w', encoding='utf-8') as file:
                    yaml.dump(data, file, default_flow_style=False, sort_keys=False)
                break
            except FileNotFoundError:
                if attempt == max_attempts - 1:
                    raise
                sleep(0.5 * (attempt + 1))
        
        print(f"Parameters saved to: {filepath}")
    
    # Placeholder for validation method, currently not used
    @staticmethod
    def validate_parameters(params: Dict[str, Any], 
                          schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate parameter data against a schema when one is available.

        Parameters
        ----------
        params : dict of str to Any
            Parameter data to validate.
        schema : dict of str to Any, optional
            Schema mapping. If omitted, :data:`SCHEMA_FILE` is loaded when it
            exists.

        Returns
        -------
        bool
            ``True`` when validation completes without collected errors.

        Notes
        -----
        Detailed schema validation is not implemented; this method currently
        performs schema availability checks only.
        """
        if schema is None:
            if SCHEMA_FILE.exists():
                schema = ParameterIO.load_yaml(SCHEMA_FILE)
            else:
                print(f"Warning: Schema file not found at {SCHEMA_FILE}. Skipping validation.")
                return True
        
        # Add this check right after loading schema
        if not schema:
            print("Warning: Empty schema. Skipping validation.")
            return True
        
        errors = []
        
        # Validate metadata
        if 'metadata' in schema and 'metadata' in params:
            # Add metadata validation logic here if needed
            pass
        
        # Validate parameter sections
        for section, section_schema in schema.items():
            if section == 'metadata':
                continue
            # Add section validation logic here if needed
        
        if errors:
            raise ParameterValidationError(f"Validation errors: {errors}")
        
        return True

# Dataclass definitions. Data is obtained from YAML files, and converted to these structures.
@dataclass
class CellProperties:
    """Cell-specific composition, mass, and growth parameters.

    Notes
    -----
    ``mu_max``, ``dry_cell_mass``, and ``cells_per_mol`` are calculated from
    the supplied doubling time and biomass properties during initialization.
    """
    td: Parameter
    mu_max: Parameter = field(init=False)
    cell_diameter: Parameter # in micrometers; # not used in current model but included for potential future use in volume/mass conversions
    cell_density: Parameter # in g/mL; # not used in current model but included for potential future use in volume/mass conversions
    wet_cell_mass: Parameter # in g/cell
    dry_mass_fraction: Parameter # fraction of wet mass that is dry mass
    protein_fraction: Parameter # fraction of dry mass that is protein
    lipid_fraction: Parameter # fraction of dry mass that is lipid
    carbohydrate_fraction: Parameter # fraction of dry mass that is carbohydrate
    nucleic_acid_fraction: Parameter # fraction of dry mass that is nucleic acids 
    M_biomass: Parameter # in g/Cmol
    M_protein: Parameter # in g/Cmol
    M_AAs_ex_glutamine: Parameter # in g/mol
    biomass_C: Parameter # empirical biomass carbon coefficient on the configured DCM basis
    biomass_H: Parameter # empirical biomass hydrogen coefficient on the configured DCM basis
    biomass_O: Parameter # empirical biomass oxygen coefficient on the configured DCM basis
    biomass_N: Parameter # empirical biomass nitrogen coefficient on the configured DCM basis
    Xv_min: Parameter # in cells/mL
    X_viscosity_limit: Parameter # in cells/L
    dry_cell_mass: Parameter = field(init=False) # in g / cell
    cells_per_mol: Parameter = field(init=False) # in cells/mol
    
    def __post_init__(self):
        # Calculate derived parameters
        value_mu_max = np.log(2) / float(self.td)
        self.mu_max = Parameter(
            name="Maximum Specific Growth Rate",
            value=value_mu_max,
            unit="1/h"
        )

        value_dry_cell_mass = float(self.wet_cell_mass) * float(self.dry_mass_fraction) # in g / cell
        self.dry_cell_mass = Parameter(
            name="Dry Cell Mass",
            value=value_dry_cell_mass,
            unit="g/cell"
        )

        value_cells_per_mol = float(self.M_biomass) / float(self.dry_cell_mass)
        self.cells_per_mol = Parameter(
            name="Cells per mol Biomass",
            value=value_cells_per_mol,
            unit="cells/mol"
        )

@dataclass
class AnabolicReaction:
    """
    Stoichiometric coefficients for anabolic (growth) reaction.
    Represents the conversion of substrates to biomass and metabolic products.
    
    ``delta_Hf`` and ``delta_Gf`` are formation enthalpy and Gibbs free energy,
    respectively, in kJ/mol. All remaining fields are dimensionless
    stoichiometric coefficients. Negative coefficients indicate consumption and
    positive coefficients indicate production.
    """
    # Substrates (negative values)
    Glc: Parameter
    Arg: Parameter
    Cys: Parameter
    Gln: Parameter
    His: Parameter
    Ile: Parameter
    Lys: Parameter
    Met: Parameter
    Phe: Parameter
    Thr: Parameter
    Trp: Parameter
    Tyr: Parameter
    Val: Parameter
    Ala: Parameter
    Asn: Parameter
    Asp: Parameter
    Gly: Parameter
    Leu: Parameter
    Pro: Parameter
    Ser: Parameter
    
    # Products (positive values)
    DCM: Parameter  # Dry cell mass
    Glu: Parameter
    CO2: Parameter
    H2O: Parameter
    
    # Thermodynamic properties
    delta_Hf: Parameter
    delta_Gf: Parameter

@dataclass
class CatabolicReaction:
    """
    Stoichiometric coefficients for catabolic (maintenance) reaction.
    Represents the conversion of substrates to energy and waste products.

        Notes
        -----
        Two input forms are supported:

        - *Explicit-coefficient mode* provides all reaction coefficients and
            ``delta_Hf``/``delta_Gf``.
        - *Yield-based mode* provides ``Y_Lac_Glc`` and ``Y_Gln_Glc``. The
            stoichiometric model derives the remaining coefficients by elemental atom
            balancing and calculates the catabolic formation enthalpy from standard
            formation enthalpies.

        Explicit fields are optional to support yield-based YAML files. Negative
        coefficients indicate consumption; positive coefficients indicate
        production.
    """
    # --- Explicit-coefficient mode (all Optional to support yield-based YAMLs) ---
    # Substrates (negative values)
    Glc: Optional[Parameter] = None
    Gln: Optional[Parameter] = None
    O2: Optional[Parameter] = None

    # Products (positive values)
    NH3: Optional[Parameter] = None
    Lac: Optional[Parameter] = None
    CO2: Optional[Parameter] = None
    H2O: Optional[Parameter] = None

    # Thermodynamic properties (used in explicit mode; derived from stoichiometry in yield-based mode)
    delta_Hf: Optional[Parameter] = None
    delta_Gf: Optional[Parameter] = None

    # --- Yield-based mode (new) ---
    Y_Lac_Glc: Optional[Parameter] = None  # Apparent lactate yield on glucose [mol Lac / mol Glc]
    Y_Gln_Glc: Optional[Parameter] = None  # Apparent glutamine yield on glucose [mol Gln / mol Glc]

@dataclass
class SharedMediumProperties:
    """Shared medium properties used across media definitions.

    This section stores medium properties that remain shared between initial and
    feed media definitions (e.g., density, pH reference, dissolved gases).
    Osmolality baseline is intentionally defined under ``initial_conditions`` so
    initial-medium and feed-medium baselines can be configured independently.
    """
    density: Parameter
    pH: Parameter
    C_H: Parameter = field(init=False)
    C_O2: Parameter
    C_CT: Parameter
    
    def __post_init__(self):
        # Calculate H+ concentration from pH
        value_C_H = 10**(-float(self.pH)) # in mol/L
        self.C_H = Parameter(
            name="H+ Concentration",
            value=value_C_H,
            unit="mol/L"
        )

@dataclass
class BufferDefinition:
    """Definition of a non-carbonate buffer species in the culture medium.

    Supports monoprotic buffers such as HEPES (pKa ≈ 7.55) and the
    H₂PO₄⁻/HPO₄²⁻ pair of phosphate (pKa₂ ≈ 7.20).  Multi-protic buffers
    can be modelled by declaring separate ``BufferDefinition`` entries for
    each relevant equilibrium.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. "HEPES", "Phosphate").
    pKa : Parameter
        Acid dissociation constant (unitless, -log10 scale).
    C_init : Parameter
        Total buffer concentration in the initial culture medium (mmol/L).
    C_feed : Parameter
        Total buffer concentration in the feed / perfusion medium (mmol/L).
    type : str
        Buffer type (``"monoprotic"`` only; diprotic/triprotic deferred).
    """
    name: str
    pKa: Parameter
    C_init: Parameter
    C_feed: Parameter
    type: str = "monoprotic"


@dataclass
class AcidBaseProperties:
    """Acid, base, and optional buffer properties used by pH control.

    Notes
    -----
    If omitted, ``D_base_max`` and ``D_acid_max`` default to 5% of
    ``reactor.D_max``. ``buffers`` may contain :class:`BufferDefinition`
    instances for non-carbonate buffer equilibria.
    """
    acid_concentration: Parameter
    #acid_pKa: Parameter
    base_concentration: Parameter # Assume strong base, so no pKb needed. Full dissociation.
    #base_pKa: Parameter
    # Maximum base addition (dilution) rate for pH control (1/h). If not provided,
    # defaults to 5% of the reactor D_max (set in SimulationParameters.__post_init__).
    D_base_max: Optional[Parameter] = None
    # Maximum acid addition (dilution) rate for pH control (1/h).  Symmetric to
    # D_base_max — used to counteract alkalinization when the buffer system
    # drives pH above setpoint and CO₂ enrichment is saturated.
    # If not provided, defaults to 5% of the reactor D_max.
    D_acid_max: Optional[Parameter] = None
    # Optional non-carbonate buffers (HEPES, phosphate, etc.).  Each entry is a
    # BufferDefinition describing a single equilibrium.  None or empty list means
    # only the carbonate system is active (backward compatible).
    buffers: Optional[List[BufferDefinition]] = None

# Example of how to modify other dataclasses
@dataclass
class KineticsParameters:
    """Growth, limitation, and inhibition parameters shared by kinetic models.

    Optional inhibition fields are required only by the corresponding selected
    inhibition model.
    """

    t_lag: Parameter
    K_Glc: Parameter
    K_Gln: Parameter
    K_O2: Parameter

    # These parameters are required for the default inhibition models, so they are not optional
    # O2
    IC50_O2: Parameter
    n_O2: Parameter
    # CO2
    IC100_CO2: Parameter
    n_CO2: Parameter
    # Lactate
    phi_power_decay_Lac: Parameter
    IC100_Lac: Parameter # Also used for toxicity-based-step perfusion strategy
    # Ammonium
    phi_power_decay_Amm: Parameter
    IC100_Amm: Parameter # Also used for toxicity-based-step perfusion strategy
    # Osmolality
    IC100_Osm: Parameter
    Osm_baseline_inhibition: Parameter

    # Optional inhibition parameters - required based on inhibition model chosen
    C_O2_inh_min: Optional[Parameter] = None
    IC50_CO2: Optional[Parameter] = None
    IC50_Lac: Optional[Parameter] = None
    IC50_Amm: Optional[Parameter] = None
    IC50_Osm: Optional[Parameter] = None
    phi_power_decay_Osm: Optional[Parameter] = None


@dataclass
class ExcessKineticsParameters:
    """Parameters used by the EXCESS kinetic model.

    Several lactate and ammonia fields are retained for backward-compatible
    YAML loading but are not used by the current stoichiometric closure.
    """
        # Deprecated Growth yields and maintenance
    Y_Glc: Parameter
    Y_Gln: Parameter
    m_Glc: Parameter
    m_Gln: Parameter
    
    # Limitation thresholds
    C_Glc_limit: Parameter
    C_Gln_limit: Parameter
    
    # Glucose excess parameters
    K_Glc_Glc: Parameter
    delta_q_Glc_Glc_max: Parameter
    K_Gln_Glc: Parameter
    delta_q_Gln_Glc_max: Parameter
    
    # Glutamine excess parameters
    K_Glc_Gln: Parameter
    delta_q_Glc_Gln_max: Parameter
    K_Gln_Gln: Parameter
    delta_q_Gln_Gln_max: Parameter
    
    # Lactate parameters --> Deprecated, Lac is calculated from glucose excess profile within stoichiometric boundaries
    m_Lac: Optional[Parameter] = None
    Y_Lac: Optional[Parameter] = None
    K_Glc_Lac: Optional[Parameter] = None
    delta_q_Lac_max: Optional[Parameter] = None
    
    # Ammonia parameters --> Deprecated, Amm is now calculated via stoichiometric balancing
    m_Amm: Optional[Parameter] = None
    Y_Amm: Optional[Parameter] = None
    K_Gln_Amm: Optional[Parameter] = None
    delta_q_Amm_max: Optional[Parameter] = None

@dataclass
class ReactorParameters:
    """Reactor geometry, operating setpoints, and control limits.

    Notes
    -----
    Initialization derives hydrogen-ion limits, working volumes, maximum feed
    flow, and ``D_max`` from the supplied design and operating parameters.
    """
    # pH control parameters
    pH_setpoint: Parameter
    C_H_setpoint: Parameter = field(init=False)
    y_CO2_min: Parameter
    y_CO2_max: Parameter
    
    # Reactor dimensions
    V_reactor: Parameter
    D_reactor: Parameter
    H_reactor: Parameter = field(init=False)
    P_operating: Parameter # Operating pressure in headspace
    # STR impeller properties
    d_impeller: Parameter
    power_number: Parameter
    # Gas controller properties --> Might be moved to separate gas controller dataclass later
    N_min: Parameter
    N_max: Parameter
    Fg_min: Parameter
    Fg_max: Parameter
    y_O2_min: Parameter
    y_O2_max: Parameter
    DO_setpoint: Parameter

    # Operational limits
    max_fill_fraction: Parameter
    batch_init_fraction: Parameter
    V_max: Parameter = field(init=False)
    V_batch_init: Parameter = field(init=False)
    pH_tolerance: Parameter
    C_H_upper: Parameter = field(init=False)
    C_H_lower: Parameter = field(init=False)
    F_L_max_specific: Parameter # in L/h per L of max batch volume. Also serves as D_max for continuous modes.
    F_L_max: Parameter = field(init=False)
    D_max: Parameter = field(init=False)


    def __post_init__(self):
        # Calculate H+ concentration setpoint from pH setpoint
        value_C_H_setpoint = 10**(-float(self.pH_setpoint)) # in mol/L
        self.C_H_setpoint = Parameter(
            name="H+ Concentration Setpoint",
            value=value_C_H_setpoint,
            unit="mol/L"
        )

        # Pre-Calculate upper and lower H+ concentration limits
        value_C_H_upper = 10**(-(float(self.pH_setpoint) - float(self.pH_tolerance))) # in mol/L. Lower pH is higher H+ concentration
        self.C_H_upper = Parameter(
            name="Upper H+ Concentration Limit",
            value=value_C_H_upper,
            unit="mol/L"
        )

        value_C_H_lower = 10**(-(float(self.pH_setpoint) + float(self.pH_tolerance))) # in mol/L. Higher pH is lower H+ concentration
        self.C_H_lower = Parameter(
            name="Lower H+ Concentration Limit",
            value=value_C_H_lower,
            unit="mol/L"
        )

        # Calculate derived height
        value_H = (4 * float(self.V_reactor) / 1000) / (np.pi * float(self.D_reactor)**2)
        self.H_reactor = Parameter(
            name="Reactor Height",
            value=value_H,
            unit="m"
        )
        # Calculate derived volumes
        value_V_max = float(self.max_fill_fraction) * float(self.V_reactor)
        self.V_max = Parameter(
            name="Maximum Reactor Volume",
            value=value_V_max,
            unit="L"
        )
        value_V_batch_init = float(self.batch_init_fraction) * value_V_max
        self.V_batch_init = Parameter(
            name="Initial Batch Volume",
            value=value_V_batch_init,
            unit="L"
        )
        # Calculate maximum feed flow rate from specific max feed and max batch volume
        value_F_L_max = float(self.F_L_max_specific) * value_V_max
        self.F_L_max = Parameter(
            name="Maximum Feed Flow Rate",
            value=value_F_L_max,
            unit="L/h"
        )
        # Maximum dilution rate (1/h) for continuous modes (chemostat, perfusion).
        # Numerically equal to F_L_max_specific since D = F/V and at V_max the
        # specific rate is already normalised to 1/h.
        self.D_max = Parameter(
            name="Maximum Dilution Rate",
            value=float(self.F_L_max_specific),
            unit="1/h"
        )

@dataclass
class InitialConditions:
    """Initial state definitions for a new batch at ``t = 0``.

    Notes
    -----
    - ``C_*_init`` values serve a dual role:
      1) ODE initial concentrations for the batch start, and
      2) Fresh batch-medium baseline concentrations used in multi-batch
         inoculum/medium mixing logic in :mod:`animakinetica.core.simulation`.
    - In single-batch simulations these values act only as initial conditions.
    - ``pH_init`` and ``C_O2_init`` are optional overrides; if omitted,
      :class:`SimulationParameters` defaults them to reactor setpoints.
    """
    X_inoc: Parameter
    C_Glc_init: Parameter  # Also used as fresh batch-medium glucose baseline (mmol/L)
    C_Gln_init: Parameter  # Also used as fresh batch-medium glutamine baseline (mmol/L)
    C_Lac_init: Parameter  # Also used as fresh batch-medium lactate baseline (mmol/L)
    C_Amm_init: Parameter  # Also used as fresh batch-medium ammonia baseline (mmol/L)
    C_AA_init: Parameter  # Also used as fresh batch-medium amino-acid baseline (mmol/L)
    C_CT_init: Parameter  # Also used as fresh batch-medium total dissolved carbon baseline (mmol/L)
    C_Na_init: Parameter  # Also used as fresh batch-medium sodium baseline (mmol/L)
    osm_bg_init: Parameter  # Initial/batch-medium osmolality baseline (mOsm)
    C_H_init: Parameter = field(init=False)
    pH_init: Optional[Parameter] = None # Optional initial pH override; defaults to reactor.pH_setpoint
    C_O2_init: Optional[Parameter] = None  # Optional initial DO override; defaults to reactor.DO_setpoint


    def __post_init__(self):
        # Calculate initial H+ concentration from pH
        if self.pH_init is not None:
            value_C_H = 10**(-float(self.pH_init)) # Convert from mol/L to mol/L
            self.C_H_init = Parameter(
                name="Initial H+ Concentration",
                value=value_C_H,
                unit="mol/L"
            )

@dataclass
class FedBatchParameters:
    """Feed composition, initial flow, and optional feed-control settings.

    Notes
    -----
    ``F_L0`` is derived from ``F_L0_specific`` and the reactor initial batch
    volume. The legacy ``F_L_max_specific`` and ``F_L_max`` fields are accepted
    during YAML loading but ignored; their active equivalents are in
    :class:`ReactorParameters`.
    """
    C_Glc_feed: Parameter
    C_Gln_feed: Parameter
    C_AA_feed: Parameter
    osm_bg_feed: Parameter
    F_L0_specific: Parameter # in L/h per L of initial batch volume. Basically a dilution rate, but expressed in terms of initial batch volume to make it more intuitive for fed-batch operation.
    F_L0: Parameter = field(init=False)

    # Batch-phase trigger: glucose concentration (mmol/L) at which the initial
    # closed batch phase ends and feeding begins.  When set, a BatchODESystem
    # phase is prepended to FEDBATCH, CHEMOSTAT, CONCENTRATED_FEDBATCH, and
    # CONTINUOUS_PERFUSION modes.  Must be > 0.001 mmol/L (the base
    # glucose-depletion event threshold) and < C_Glc_init to be effective.
    C_Glc_batch_trigger: Optional[Parameter] = None

    # Feed strategy parameters (optional, depends on strategy used)
    r: Optional[Parameter] = None  # For exponential feed
    feed_slope: Optional[Parameter] = None  # For linear feed
    CSFR: Optional[Parameter] = None  # For cell-specific feed rate
    # For glucose setpoint feed strategy
    C_Glc_setpoint: Optional[Parameter] = None
    C_Glc_setpoint_Kp: Optional[Parameter] = None
    C_Glc_setpoint_deadband: Optional[Parameter] = None
    C_Glc_setpoint_slew_rate: Optional[Parameter] = None
    C_Glc_setpoint_den_epsilon: Optional[Parameter] = None

    # DEPRECATED: F_L_max_specific and F_L_max have been centralised to ReactorParameters.
    # Kept as Optional for backward-compatible YAML loading; ignored at runtime.
    F_L_max_specific: Optional[Parameter] = None
    F_L_max: Optional[Parameter] = field(init=False, default=None)


    def calculate_scaled_flowrate(self, V_init: Parameter):
        """Calculate scaled flowrate parameters from reactor reference volumes."""
        value = float(self.F_L0_specific) * float(V_init)
        self.F_L0 = Parameter(
            name="Initial Feed Flow Rate",
            value=value,
            unit="L/h"
    )

@dataclass
class ChemostatParameters:
    """Medium composition, dilution rate, and optional chemostat trigger."""
    D: Parameter
    C_Glc_feed: Parameter
    C_Gln_feed: Parameter
    C_AA_feed: Parameter
    osm_bg_feed: Parameter
    
    # In case alternative switching strategy is used, these parameters can be used instead of max volume
    Xv_chemostat_trig: Optional[Parameter] = None  # Trigger cell density for switching to chemostat mode


@dataclass
class PerfusionParameters:
    """Medium composition, separator, dilution-control, and trigger settings.

    Notes
    -----
    ``D_1`` and ``D_2`` are derived from ``D_baseline`` and their respective
    multipliers. The legacy ``D_max`` field is accepted during YAML loading but
    ignored; the active limit is ``reactor.D_max``.
    """
    # General perfusion parameters
    eff_separator: Parameter
    B: Parameter
    
    # Feed concentrations
    C_Glc_feed: Parameter
    C_Gln_feed: Parameter
    C_AA_feed: Parameter
    osm_bg_feed: Parameter
    # Constant strategy
    D_baseline: Parameter
    # Toxicity-based step strategy
    D_1_multiplier: Parameter
    D_2_multiplier: Parameter
    D_1: Parameter = field(init=False)
    D_2: Parameter = field(init=False)
    # Toxicity-based proportional strategy
    k_Lac: Parameter
    k_Amm: Parameter
    # Cell density based rate specific perfusion rate (CSPR)
    CSPR: Parameter
    # Glucose setpoint perfusion strategy
    C_Glc_setpoint: Optional[Parameter] = None
    C_Glc_setpoint_Kp: Optional[Parameter] = None
    C_Glc_setpoint_deadband: Optional[Parameter] = None
    C_Glc_setpoint_slew_rate: Optional[Parameter] = None
    C_Glc_setpoint_den_epsilon: Optional[Parameter] = None
    # Perfusion trigger parameters (optional, depends on strategy used)
    C_Lac_pf_trig: Optional[Parameter] = None
    C_Amm_pf_trig: Optional[Parameter] = None
    Xv_pf_trig: Optional[Parameter] = None
    Xv_harvest_stop: Optional[Parameter] = None
    min_perfusion_runtime_h: Optional[Parameter] = None

    # DEPRECATED: D_max has been centralised to ReactorParameters.
    # Kept as Optional for backward-compatible YAML loading; ignored at runtime.
    D_max: Optional[Parameter] = None

    def __post_init__(self):
        self.D_1 = Parameter(
            name="Dilution Rate 1",
            value=float(self.D_1_multiplier) * float(self.D_baseline),
            unit="1/h"
        )
        self.D_2 = Parameter(
            name="Dilution Rate 2",
            value=float(self.D_2_multiplier) * float(self.D_baseline),
            unit="1/h"
        )
    
@dataclass
class ProcessSchedule:
    """Annual production availability and maximum culture-age parameters."""
    yearly_uptime: Parameter
    max_culture_age: Parameter

@dataclass
class GasTransferConstants:
    """Henry-law and liquid-diffusivity constants for gas transfer."""
    H_O2: Parameter
    H_CO2: Parameter
    D_L_O2: Parameter
    D_L_CO2: Parameter

@dataclass
class SimulationParameters:
    """Complete parameter configuration used to construct a simulation.

    Parameters
    ----------
    cell_properties : CellProperties
        Cell composition, growth, and conversion properties.
    anabolic_reaction : AnabolicReaction
        Growth-reaction stoichiometry.
    catabolic_reaction : CatabolicReaction
        Catabolic stoichiometry or yield-based closure inputs.
    shared_medium_properties : SharedMediumProperties
        Medium properties shared by initial and feed media.
    acid_base_properties : AcidBaseProperties
        Acid/base and buffer properties.
    kinetics : KineticsParameters
        Kinetic limitation and inhibition settings.
    reactor : ReactorParameters
        Reactor geometry, setpoints, and constraints.
    initial_conditions : InitialConditions
        Initial culture-medium state.
    process_schedule : ProcessSchedule
        Cultivation and scheduling durations.
    gas_transfer : GasTransferConstants
        Gas-transfer physical constants.
    fed_batch : FedBatchParameters, optional
        Fed-batch settings.
    chemostat : ChemostatParameters, optional
        Chemostat settings.
    perfusion : PerfusionParameters, optional
        Perfusion settings.
    excess_kinetics : ExcessKineticsParameters, optional
        EXCESS-model settings.
    metadata : dict of str to Any, optional
        Source and user metadata.

    Notes
    -----
    The object is mutable; use
    :meth:`create_modified_parameter_object` to produce an independent modified
    copy for optimization or sensitivity analysis. Use
    :meth:`load_parameters` to load a complete file or compose a configuration
    from ordered YAML files; later sections overwrite earlier values.
    """
    cell_properties: CellProperties
    anabolic_reaction: AnabolicReaction
    catabolic_reaction: CatabolicReaction
    shared_medium_properties: SharedMediumProperties
    acid_base_properties: AcidBaseProperties
    kinetics: KineticsParameters
    reactor: ReactorParameters
    initial_conditions: InitialConditions 
    process_schedule: ProcessSchedule
    gas_transfer: GasTransferConstants
    fed_batch: Optional[FedBatchParameters] = None
    chemostat: Optional[ChemostatParameters] = None
    perfusion: Optional[PerfusionParameters] = None
    excess_kinetics: Optional[ExcessKineticsParameters] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Calculate scaled flowrate if fed_batch parameters are present
        if self.fed_batch is not None:
            self.fed_batch.calculate_scaled_flowrate(self.reactor.V_batch_init)
        
        # Set initial DO to DO setpoint if not explicitly provided in initial conditions
        if self.initial_conditions.C_O2_init is None:
            self.initial_conditions.C_O2_init = self.reactor.DO_setpoint
        
        if self.initial_conditions.pH_init is None:
            self.initial_conditions.pH_init = self.reactor.pH_setpoint
            self.initial_conditions.C_H_init = self.reactor.C_H_setpoint

        # Default the maximum base addition (dilution) rate for pH control.
        # Recomputed here (and after section merges) because perfusion may be
        # loaded in a later file than acid_base_properties.
        self._apply_D_base_max_default()
        self._apply_D_acid_max_default()

        # Add creation timestamp if not present
        if 'created' not in self.metadata:
            self.metadata['created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      
    def __getitem__(self, param_path: str) -> Parameter:
        """
        Allow dictionary-style access to parameters.
        
        Examples:
            >>> params = SimulationParameters.load_parameters()
            >>> mu_max = params['kinetics.mu_max']
            >>> print(mu_max.value, mu_max.unit)
        """
        return self.get_parameter(param_path)
    
    def __setitem__(self, param_path: str, value: Any):
        """
        Allow dictionary-style setting of parameter values.
        
        Examples:
            >>> params = SimulationParameters.load_parameters()
            >>> params['kinetics.td'] = 48.0
        """
        self.set_parameter_value(param_path, value)

    # Methods to get and set parameters using dot notation
    def get_parameter(self, param_path: str) -> Parameter:
        """Return a :class:`Parameter` identified by a dotted path.

        Parameters
        ----------
        param_path : str
            Path in ``"section.attribute"`` form, for example
            ``"cell_properties.mu_max"``.

        Returns
        -------
        Parameter
            Requested parameter object.

        Raises
        ------
        ValueError
            If the path is malformed, the section or attribute is absent, or the
            resolved attribute is not a ``Parameter``.
        """
        if '.' not in param_path:
            raise ValueError(f"Parameter path must be in format 'section.attribute', got: {param_path}")
        
        section, attribute = param_path.split('.', 1)
        
        section_obj = getattr(self, section, None)
        if section_obj is None:
            raise ValueError(f"Section '{section}' not found in parameters")
        
        param_obj = getattr(section_obj, attribute, None)
        if param_obj is None:
            raise ValueError(f"Parameter '{attribute}' not found in section '{section}'")
        
        if not isinstance(param_obj, Parameter):
            raise ValueError(f"'{param_path}' is not a Parameter object")
        
        return param_obj
    
    def get_parameter_value(self, param_path: str) -> Any:
        """Return the raw value of a parameter identified by a dotted path.

        Parameters
        ----------
        param_path : str
            Path accepted by :meth:`get_parameter`.

        Returns
        -------
        Any
            The ``value`` attribute of the requested parameter.
        """
        return self.get_parameter(param_path).value
    
    def set_parameter_value(self, param_path: str, value: Any):
        """Set a parameter's raw value and update modification metadata.

        Parameters
        ----------
        param_path : str
            Path accepted by :meth:`get_parameter`.
        value : Any
            Replacement raw value.

        Notes
        -----
        This method does not recalculate dependent values. Call
        :meth:`recompute_derived_parameters` after direct mutations, or use
        :meth:`create_modified_parameter_object`.
        """
        param = self.get_parameter(param_path)
        param.value = value
        
        # Update metadata
        self.metadata['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    ### Method to save parameters to YAML file in case they need to be exported
    def save_parameters(self, name: str, description: str = "", overwrite: bool = False):
        """Save this configuration in :data:`PARAMETER_SETS_DIR`.

        Parameters
        ----------
        name : str
            Destination filename stem.
        description : str, default=""
            Optional metadata description to save.
        overwrite : bool, default=False
            Whether to replace an existing ``{name}.yaml`` file.

        Returns
        -------
        pathlib.Path
            Saved YAML file path.
        """
        # Update metadata
        self.metadata['name'] = name
        if description:
            self.metadata['description'] = description
        self.metadata['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert to YAML format
        yaml_dict = self._to_yaml_dict()
        
        # Save to file
        filepath = PARAMETER_SETS_DIR / f"{name}.yaml"
        ParameterIO.save_yaml(yaml_dict, filepath, overwrite=overwrite)
    
        return filepath

    def _apply_D_base_max_default(self):
        """Set the default maximum base addition (dilution) rate for pH control.

        When ``acid_base_properties.D_base_max`` is not explicitly configured,
        default it to 5% of the reactor maximum dilution rate (``reactor.D_max``).
        Safe to call repeatedly: an explicitly configured value is preserved, and
        an already-applied default is not overwritten.
        """
        if (self.acid_base_properties is not None
                and self.acid_base_properties.D_base_max is None
                and self.perfusion is not None
                and self.reactor.D_max is not None):
            self.acid_base_properties.D_base_max = Parameter(
                name="Maximum Base Addition Rate",
                value=0.05 * float(self.reactor.D_max),
                unit="1/h"
            )

    def _apply_D_acid_max_default(self):
        """Set the default maximum acid addition (dilution) rate for pH control.

        When ``acid_base_properties.D_acid_max`` is not explicitly configured,
        default it to 5% of the reactor maximum dilution rate (``reactor.D_max``).
        Safe to call repeatedly: an explicitly configured value is preserved, and
        an already-applied default is not overwritten.
        """
        if (self.acid_base_properties is not None
                and self.acid_base_properties.D_acid_max is None
                and self.reactor.D_max is not None):
            self.acid_base_properties.D_acid_max = Parameter(
                name="Maximum Acid Addition Rate",
                value=0.05 * float(self.reactor.D_max),
                unit="1/h"
            )

    def recompute_derived_parameters(self):
        """Recalculate all derived fields after direct value mutations.

        The recalculation order respects dependencies among cell, medium,
        reactor, initial-condition, optional perfusion, and container-level
        fed-batch fields.
        """
        # Rebuild section-level derived values first.
        self.cell_properties.__post_init__()
        self.shared_medium_properties.__post_init__()
        self.reactor.__post_init__()
        self.initial_conditions.__post_init__()

        # Optional sections with derived parameters.
        if self.perfusion is not None:
            self.perfusion.__post_init__()

        # Rebuild container-level derived values (fed-batch scaled flow rates).
        self.__post_init__()
    
    def create_modified_parameter_object(self, modifications: Dict[str, Any]) -> 'SimulationParameters':
        """Return a deep-copied configuration with requested values replaced.

        Parameters
        ----------
        modifications : dict of str to Any
            Mapping from paths accepted by :meth:`set_parameter_value` to their
            replacement raw values.

        Returns
        -------
        SimulationParameters
            Independent modified copy with dependent fields recalculated.
        """
        # Create a deep copy of base parameters
        modified_param_obj = copy.deepcopy(self)
        
        # Apply optimized values
        for param_key, value in modifications.items():
            modified_param_obj.set_parameter_value(param_key, value)

        # Recompute all derived values after modifications.
        modified_param_obj.recompute_derived_parameters()
        
        # Update metadata
        modified_param_obj.metadata['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        modified_param_obj.metadata['optimization_applied'] = True
        
        return modified_param_obj


    def create_mode_adjusted_copy(self, operating_mode: Any) -> 'SimulationParameters':
        """Return a copy with operating-mode-specific defaults applied.

        Parameters
        ----------
        operating_mode : Any
            Operating-mode enum or object with a ``name`` attribute.

        Returns
        -------
        SimulationParameters
            ``self`` for non-batch modes. For ``BATCH``, an independent copy
            with ``reactor.batch_init_fraction`` set to 1.0.
        """
        if getattr(operating_mode, "name", None) != "BATCH":
            return self

        adjusted_params = copy.deepcopy(self)
        adjusted_params.reactor.batch_init_fraction.value = 1.0
        adjusted_params.reactor.__post_init__()


        adjusted_params.metadata['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        adjusted_params.metadata['operating_mode_defaults_applied'] = 'BATCH'
        return adjusted_params


    ### Modular parameter loading methods
    @classmethod
    def load_parameters(cls, 
                       parameter_set: Optional[Union[str, Path, List[Union[str, Path, Tuple[str, List[str]]]]]] = None,
                       sections: Optional[List[str]] = None,
                       validate: bool = True) -> 'SimulationParameters':
        """Load a complete configuration from one or more YAML parameter files.

        Parameters
        ----------
        parameter_set : str, pathlib.Path, list, or None, optional
            Source specification. ``None`` loads ``base_parameters``. A string
            or path loads one file. A list loads sources in order; later files
            overwrite matching fields. List items may also be
            ``(source, sections)`` tuples to restrict one update to selected
            sections.
        sections : list of str, optional
            Sections to load from a single source. Ignored when
            ``parameter_set`` is a list; use tuple list items instead.
        validate : bool, default=True
            Reserved validation flag. Schema validation is currently not invoked
            by this loading path.

        Returns
        -------
        SimulationParameters
            Loaded configuration with derived fields calculated.

        Raises
        ------
        FileNotFoundError
            If a requested parameter file cannot be resolved.
        ParameterValidationError
            If a source YAML file cannot be parsed.

        Examples
        --------
        Load a modular fed-batch configuration::

            params = SimulationParameters.load_parameters([
                "base_parameters",
                "operating_modes/fed_batch",
            ])
        """
        # Handle multiple file loading
        if isinstance(parameter_set, list):
            return cls._load_multiple_files(parameter_set, validate=validate)
        
        # Single file loading - use centralized path resolution for all cases
        if parameter_set is None:
            # Load base parameters (default)
            print(f"No parameter set specified. Loading base parameters...")
            filepath = cls._resolve_path("base_parameters")
        else:
            # Use centralized path resolution with recursive subdirectory search
            print(f"Searching for parameter set: {parameter_set}")
            filepath = cls._resolve_path(parameter_set)
        
        print(f"Loading parameters from: {filepath}")
        return cls._from_yaml(filepath, sections=sections, validate=validate)

    @classmethod
    def _load_multiple_files(cls,
                            file_list: List[Union[str, Path, Tuple[str, List[str]]]],
                            validate: bool = True) -> 'SimulationParameters':
        """Load, order, and merge multiple parameter sources.

        A source containing all required core sections is used first. If none is
        supplied, ``base_parameters`` is prepended before applying updates.

        Parameters
        ----------
        file_list : list
            Paths, names, or ``(source, sections)`` tuples to merge.
        validate : bool, default=True
            Reserved validation flag passed through the loading API.

        Returns
        -------
        SimulationParameters
            Fully merged parameter configuration.

        Raises
        ------
        ValueError
            If ``file_list`` is empty.
        """
        if not file_list:
            raise ValueError("file_list cannot be empty")

        # Required sections for a complete parameter set
        REQUIRED_SECTIONS = {
            'shared_medium_properties', 'acid_base_properties',
            'kinetics', 'reactor', 'initial_conditions',
            'process_schedule', 'gas_transfer'
        }

        # Ensure a complete file is first: scan the list for the first file that
        # contains all required sections.  Works regardless of filename or location.
        complete_index = None
        for idx, item in enumerate(file_list):
            file_name = item[0] if isinstance(item, tuple) else item
            try:
                candidate_sections = set(ParameterIO.load_yaml(cls._resolve_path(file_name)).keys()) - {'metadata'}
                if REQUIRED_SECTIONS.issubset(candidate_sections):
                    complete_index = idx
                    break
            except Exception:
                pass

        if complete_index is None:
            # No complete file found - prepend the standard base_parameters
            file_list = ['base_parameters'] + list(file_list)
        elif complete_index > 0:
            # Complete file found but not first - move it to the front
            file_list = list(file_list)
            file_list.insert(0, file_list.pop(complete_index))

        print(f"Loading parameters from {len(file_list)} file(s) sequentially...")

        # Load all files: first as base, the rest as incremental updates
        for idx, item in enumerate(file_list):
            file_name, sections = item if isinstance(item, tuple) else (item, None)
            resolved_path = cls._resolve_path(file_name)

            if idx == 0:
                print(f"  [1/{len(file_list)}] Base: {resolved_path}")
                instance = cls._from_yaml(resolved_path, sections=sections, validate=False)
                instance._update_metadata(source_file=str(resolved_path))
            else:
                print(f"  [{idx + 1}/{len(file_list)}] Updating with: {resolved_path}")
                instance.update(resolved_path, sections=sections)

        print(f"Successfully loaded parameters from {len(file_list)} file(s).")
        return instance
    
    @staticmethod
    def _resolve_path(parameter_set: Union[str, Path]) -> Path:
        """Resolve a parameter-file name or path.

        Absolute paths are used directly. Relative sources are resolved in
        :data:`PARAMETER_SETS_DIR`; exact relative paths are preferred, followed
        by recursive filename matching. A missing extension is treated as
        ``.yaml``.

        Parameters
        ----------
        parameter_set : str or pathlib.Path
            Parameter-file name or path, with or without ``.yaml``.

        Returns
        -------
        pathlib.Path
            Existing resolved parameter-file path.

        Raises
        ------
        FileNotFoundError
            If no matching file is found.
        """
        filepath = Path(parameter_set)
        
        # If absolute path, use directly
        if filepath.is_absolute():
            if filepath.exists():
                return filepath
            raise FileNotFoundError(f"Parameter file not found: {filepath}")
        
        # Helper function to search recursively in parameter_sets directory
        def search_recursive(base_dir: Path, filename: str) -> Optional[Path]:
            """Search for filename recursively in base_dir.
            Returns first match found, prioritizing shallower paths.
            """
            # Try exact match first (direct child)
            exact_path = base_dir / filename
            if exact_path.exists() and exact_path.is_file():
                return exact_path
            
            # Then search recursively in subdirectories
            matches = list(base_dir.rglob(filename))
            if matches:
                # Filter to only files (not directories)
                file_matches = [p for p in matches if p.is_file()]
                if file_matches:
                    # Return the match with the shortest path (shallowest directory)
                    return min(file_matches, key=lambda p: len(p.parts))
            return None
        
        # Determine filename(s) to search for
        if filepath.suffix:
            # Has extension - search as-is
            search_names = [str(filepath), filepath.name]
        else:
            # No extension - try with .yaml
            search_names = [f"{filepath}.yaml", f"{filepath.name}.yaml"]
        
        # Try each search name in PARAMETER_SETS_DIR
        for search_name in search_names:
            # If user specified a relative path (e.g., "subdir/params"), try it relative to PARAMETER_SETS_DIR
            if '/' in str(filepath) or '\\' in str(filepath):
                param_sets_path = PARAMETER_SETS_DIR / search_name
                if param_sets_path.exists() and param_sets_path.is_file():
                    return param_sets_path
            
            # Search recursively in PARAMETER_SETS_DIR (tries exact match first, then subdirectories)
            result = search_recursive(PARAMETER_SETS_DIR, Path(search_name).name)
            if result:
                return result
        
        # File not found in any location
        raise FileNotFoundError(
            f"Parameter file not found: '{parameter_set}'\n"
            f"Searched recursively in: {PARAMETER_SETS_DIR} (and subdirectories)\n"
            f"Tried filenames: {', '.join(search_names)}"
        )

    @staticmethod
    def _convert_buffers(data: Dict[str, Any]) -> None:
        """Convert acid/base buffer mappings to :class:`BufferDefinition`.

        Parameters
        ----------
        data : dict of str to Any
            Extracted parameter data. The mapping is modified in place.

        Notes
        -----
        Call after :meth:`ParameterIO.extract_parameters`. Entries in
        ``acid_base_properties.buffers`` are replaced with fully constructed
        ``BufferDefinition`` objects.
        """
        ab_data = data.get('acid_base_properties')
        if ab_data is None:
            return
        raw_buffers = ab_data.get('buffers')
        if raw_buffers is None or not isinstance(raw_buffers, list):
            return

        converted = []
        for entry in raw_buffers:
            if isinstance(entry, BufferDefinition):
                converted.append(entry)
                continue
            # Convert each field to a Parameter (handles raw scalar values)
            param_fields = {}
            for key in ('pKa', 'C_init', 'C_feed'):
                val = entry.get(key)
                param_fields[key] = (
                    ParameterIO.create_parameter(val, key)
                    if not isinstance(val, Parameter) else val
                )
            converted.append(BufferDefinition(
                name=entry.get('name', 'Buffer'),
                type=entry.get('type', 'monoprotic'),
                **param_fields,
            ))
        ab_data['buffers'] = converted


    @classmethod
    def _from_yaml(cls, 
                   filepath: Union[str, Path], 
                   sections: Optional[List[str]] = None,
                   validate: bool = True) -> 'SimulationParameters':
        """Construct a parameter configuration from one complete YAML file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            YAML file to parse.
        sections : list of str, optional
            Sections to retain before constructing the configuration.
        validate : bool, default=True
            Reserved validation flag; validation is not currently invoked.

        Returns
        -------
        SimulationParameters
            Parsed configuration.

        Notes
        -----
        This internal constructor requires all non-optional sections needed by
        :class:`SimulationParameters`. Use :meth:`load_parameters` for modular
        loading.
        """
        raw_data = ParameterIO.load_yaml(filepath)
        
        # Filter sections if requested
        if sections is not None:
            filtered_data = {'metadata': raw_data.get('metadata', {})}
            for section in sections:
                if section in raw_data:
                    filtered_data[section] = raw_data[section]
                else:
                    print(f"Warning: Requested section '{section}' not found in {filepath}")
            raw_data = filtered_data
        
        # Validation placeholder (not implemented)
        # if validate:
        #     ParameterIO.validate_parameters(raw_data)
        
        # Extract values only for parameter creation
        data = ParameterIO.extract_parameters(raw_data)

        # Convert raw buffer dicts -> BufferDefinition objects
        cls._convert_buffers(data)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        
        # Create dataclass instances
        cell_props = CellProperties(**data['cell_properties']) if 'cell_properties' in data else None
        anabolic_reaction = AnabolicReaction(**data['anabolic_reaction']) if 'anabolic_reaction' in data else None
        catabolic_reaction = CatabolicReaction(**data['catabolic_reaction']) if 'catabolic_reaction' in data else None
        shared_medium_props = SharedMediumProperties(**data['shared_medium_properties']) if 'shared_medium_properties' in data else None
        acid_base_props = AcidBaseProperties(**data['acid_base_properties']) if 'acid_base_properties' in data else None
        kinetics = KineticsParameters(**data['kinetics']) if 'kinetics' in data else None
        excess_kin = ExcessKineticsParameters(**data['excess_kinetics']) if 'excess_kinetics' in data else None
        reactor = ReactorParameters(**data['reactor']) if 'reactor' in data else None
        initial = InitialConditions(**data['initial_conditions']) if 'initial_conditions' in data else None
        fed_batch = FedBatchParameters(**data['fed_batch']) if 'fed_batch' in data else None
        chemostat = ChemostatParameters(**data['chemostat']) if 'chemostat' in data else None
        perfusion = PerfusionParameters(**data['perfusion']) if 'perfusion' in data else None
        schedule_data = dict(data['process_schedule']) if 'process_schedule' in data else None
        if schedule_data is not None:
            # ``setup_time`` was previously used to insert turnaround gaps
            # between fresh vessel lifetimes. Yearly uptime now represents the
            # available production horizon directly, so retain load
            # compatibility with old parameter files while ignoring the field.
            schedule_data.pop('setup_time', None)
            schedule = ProcessSchedule(**schedule_data)
        else:
            schedule = None
        gas_transfer = GasTransferConstants(**data['gas_transfer']) if 'gas_transfer' in data else None
        
        # Create instance
        instance = cls(
            cell_properties=cell_props,
            anabolic_reaction=anabolic_reaction,
            catabolic_reaction=catabolic_reaction,
            shared_medium_properties=shared_medium_props,
            acid_base_properties=acid_base_props,
            kinetics=kinetics,
            excess_kinetics=excess_kin,
            reactor=reactor,
            initial_conditions=initial,
            fed_batch=fed_batch,
            chemostat=chemostat,
            perfusion=perfusion,
            process_schedule=schedule,
            gas_transfer=gas_transfer,
            metadata=metadata
        )
        return instance
    
    def update(self, 
               parameter_file: Union[str, Path],
               sections: Optional[List[str]] = None) -> 'SimulationParameters':
        """Merge parameter sections from a YAML file into this object.

        Parameters
        ----------
        parameter_file : str or pathlib.Path
            Parameter-set name or path resolved by :meth:`_resolve_path`.
        sections : list of str, optional
            Sections to merge. By default, every non-metadata section present in
            the source file is merged.

        Returns
        -------
        SimulationParameters
            This modified object, enabling method chaining.

        Notes
        -----
        Each updated section is reconstructed so its derived fields are current.
        Metadata records the source file and modification time.
        """
        # Resolve file path
        filepath = self._resolve_path(parameter_file)
        
        # Load YAML data
        raw_data = ParameterIO.load_yaml(filepath)
        
        # Filter sections if requested
        sections_to_update = sections if sections is not None else list(raw_data.keys())
        if 'metadata' in sections_to_update:
            sections_to_update.remove('metadata')  # Handle metadata separately
        
        # Extract parameters
        data = ParameterIO.extract_parameters(raw_data)
        
        # Convert raw buffer dicts -> BufferDefinition objects
        self._convert_buffers(data)

        # Update each section
        for section_name in sections_to_update:
            if section_name == 'metadata':
                continue
            
            if section_name in data:
                self._merge_section(section_name, data[section_name])
            else:
                print(f"Warning: Requested section '{section_name}' not found in {filepath}")
        
        # Update metadata
        self._update_metadata(source_file=str(filepath))
        
        return self
    
    def _merge_section(self, section_name: str, new_section_data: Dict[str, Parameter]):
        """Merge one section and reconstruct its corresponding dataclass.

        Parameters
        ----------
        section_name : str
            Name of a :class:`SimulationParameters` section.
        new_section_data : dict of str to Parameter
            Incoming values for that section. Buffer entries are represented by
            :class:`BufferDefinition` objects.

        Notes
        -----
        Reconstructing the dataclass recalculates its derived fields. New
        ``acid_base_properties.buffers`` entries replace existing entries with
        the same name and otherwise are appended.
        """
        # Get current section object
        current_section = getattr(self, section_name, None)
        
        # Prepare merged data
        if current_section is None:
            # Section doesn't exist, create new from provided data
            merged_data = new_section_data
        else:
            # Merge: start with current values, overwrite with new values
            merged_data = {}
            
            # Get all init fields from current section
            section_fields = {f.name: f for f in fields(current_section) if f.init}
            
            for field_name in section_fields:
                if field_name in new_section_data:
                    # Special handling for buffers list in acid_base_properties:
                    # Append new buffers to existing ones, replacing any with the same name
                    if section_name == 'acid_base_properties' and field_name == 'buffers':
                        current_buffers = getattr(current_section, field_name) or []
                        new_buffers = new_section_data[field_name] or []
                        # Build a dict keyed by buffer name for deduplication
                        buf_dict = {b.name: b for b in current_buffers}
                        for new_buf in new_buffers:
                            buf_dict[new_buf.name] = new_buf
                        merged_data[field_name] = list(buf_dict.values())
                    else:
                        # Use new value
                        merged_data[field_name] = new_section_data[field_name]
                else:
                    # Keep current value
                    merged_data[field_name] = getattr(current_section, field_name)
        
        # Get the appropriate dataclass type
        section_class_map = {
            'cell_properties': CellProperties,
            'anabolic_reaction': AnabolicReaction,
            'catabolic_reaction': CatabolicReaction,
            'shared_medium_properties': SharedMediumProperties,
            'acid_base_properties': AcidBaseProperties,
            'kinetics': KineticsParameters,
            'excess_kinetics': ExcessKineticsParameters,
            'reactor': ReactorParameters,
            'initial_conditions': InitialConditions,
            'fed_batch': FedBatchParameters,
            'chemostat': ChemostatParameters,
            'perfusion': PerfusionParameters,
            'process_schedule': ProcessSchedule,
            'gas_transfer': GasTransferConstants,
        }
        
        if section_name not in section_class_map:
            print(f"Warning: Unknown section '{section_name}', skipping.")
            return
        
        # Recreate section dataclass with merged data
        section_class = section_class_map[section_name]
        new_section = section_class(**merged_data)
        
        # Update the attribute
        setattr(self, section_name, new_section)
        
        # Handle special post-update calculations
        # Check if fed_batch exists and if either fed_batch or reactor was updated,
        # since fed_batch flowrate depends on reactor volume.
        if self.fed_batch is not None and (section_name == 'fed_batch' or section_name == 'reactor'):
            self.fed_batch.calculate_scaled_flowrate(self.reactor.V_batch_init)

        # The base-addition rate limit default depends on reactor.D_max, which
        # may only become available after a later file merge.
        if section_name in ('perfusion', 'acid_base_properties'):
            self._apply_D_base_max_default()
            self._apply_D_acid_max_default()
    
    def _update_metadata(self, source_file: Optional[str] = None):
        """Update modification timestamp and optionally record a source file.

        Parameters
        ----------
        source_file : str, optional
            Resolved source file to append to ``metadata["source_files"]`` if it
            is not already present.
        """
        # Update timestamp
        self.metadata['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append source file to list
        if source_file:
            if 'source_files' not in self.metadata:
                self.metadata['source_files'] = []
            if source_file not in self.metadata['source_files']:
                self.metadata['source_files'].append(source_file)
    
    def print_summary(self):
        """Print all configured parameter sections and metadata to stdout."""
        print(f"=== Loaded Parameters ===")
        
        # Iterate through all fields of SimulationParameters
        for field_info in fields(self):
            section_name = field_info.name
            
            # Skip metadata as it's handled separately
            if section_name == 'metadata':
                continue
            
            section_obj = getattr(self, section_name)
            
            # Skip if section is None (optional sections)
            if section_obj is None:
                continue
            
            # Format section name for display
            display_name = section_name.replace('_', ' ').title()
            print(f"\n{display_name}:")
            
            # Print all parameters in this section
            for key, value in section_obj.__dict__.items():
                # Skip private attributes
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
        
        # Optionally print metadata if present
        if self.metadata:
            print(f"\nMetadata:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")


    def _to_yaml_dict(self) -> Dict[str, Any]:
        """Return this configuration in a YAML-compatible structured format.

        Returns
        -------
        dict of str to Any
            Structured representation generated by
            :meth:`ParameterIO.convert_to_structured_format`.
        """
        return ParameterIO.convert_to_structured_format(self)
    

if __name__ == "__main__":
    # Example usage: Load base parameters (most common)
    params = SimulationParameters.load_parameters()
    params.print_summary()
    
    print("\n" + "="*60 + "\n")
    
    # Example usage: Load from specific file path
    params_from_file = SimulationParameters._from_yaml(BASE_PARAMS_FILE)
    
    # Example usage: Load custom parameter set by name
    # params_custom = SimulationParameters.load("my_custom_params")
    
    # Example of parameter access
    print(f"\nExample access: mu_max = {params.kinetics.mu_max}")
    print(f"Example access: X_inoc = {params.initial_conditions.X_inoc}")
    