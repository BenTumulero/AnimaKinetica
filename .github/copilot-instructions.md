# AnimaKinetica: AI Agent Instructions

## Project Overview
AnimaKinetica is a **comprehensive bioprocess simulation framework** for animal cell culture (e.g., CHO cells, avian cells) modeling growth kinetics, substrate consumption, waste metabolite production, and process control in bioreactors. Developed at **Delft University of Technology, Bioprocess Engineering Section**.

**Core Purpose**: Optimize cell culture processes through mechanistic simulation, parameter optimization, sensitivity analysis, and visualization to maximize Space-Time Yield (STY), productivity, and process efficiency.

## Architecture & Key Components

### Package Structure (`animakinetica/src/animakinetica/`)
```
core/
  ├── simulation.py          # CellCultureSimulation - main simulation engine with multi-batch support
  ├── parameters.py          # SimulationParameters + Parameter dataclasses with arithmetic operations
  ├── simulation_data.py     # BatchData & SimulationData results containers with export capabilities
  └── io_utils.py            # Shared pickle/path resolution helpers used by simulation, data, sensitivity
models/
  ├── kinetics.py            # KineticModel implementations (STOICHIOMETRIC/EXCESS) with inhibition models
  ├── stoichiometry.py       # Stoichiometric reaction models (AnabolicReaction/CatabolicReaction)
  ├── odesystems.py          # ODE systems for all operating modes (Batch/FedBatch/Chemostat/Perfusion/etc.)
  ├── events.py              # Terminal event functions for process control (12+ event types)
  ├── feed_strategies.py     # Feed rate models (Constant/Exponential/Linear/CSFR)
  ├── perfusion_strategies.py # Dilution rate strategies (Constant/Toxicity-Based/CSPR)
  ├── gas_controllers.py     # Dissolved gas control (DO/dCO2) with algebraic control + STR correlations
  └── pH_controller.py       # Algebraic pH control with base addition and CO2 stripping
analysis/
  ├── optimization.py        # OptimizationManager using differential evolution with multi-objective support
  ├── sensitivity.py         # SensitivityAnalyzer with objective mode + validity-only mode and Plotly outputs
  ├── batch_visualization.py # Plotly visualizations for batch-level results
  ├── kinetic_visualization.py # Visualizations for kinetic model outputs (rates, factors)
  ├── gas_visualization.py   # Visualizations for gas control (DO, dCO2, kLa, OTR/CTR)
  ├── process_visualization.py # Process-level metrics and comparisons
  └── _visualization_utils.py # Shared utilities for visualization modules
```

### Critical Enums & Options

**Kinetics Models** (`Kinetics` enum in kinetics.py):
- `STOICHIOMETRIC`: Stoichiometric yield-based kinetics using fixed anabolic and catabolic reactions
- `EXCESS`: Advanced kinetics with excess substrate effects (more mechanistic)

**Operating Modes** (`OperatingMode` enum in odesystems.py):
- `BATCH`: Closed system, no feed or harvest
- `FEDBATCH`: Continuous feeding, no harvest
- `CHEMOSTAT`: Continuous feed and harvest at constant volume
- `PERFUSION`: Cell retention with continuous medium exchange
- `CONCENTRATED_FEDBATCH`: Fed-batch with perfusion trigger at high toxicity
- `CONTINUOUS_PERFUSION`: Perfusion with continuous feed (not fully implemented)

**Feed Strategies** (`FeedStrategy` enum in feed_strategies.py):
- `CONSTANT`: Fixed flow rate
- `EXPONENTIAL`: F(t) = F₀ * exp(μ_set * t)
- `LINEAR`: Linear ramp
- `CSFR`: Cell-specific feed rate (proportional to biomass)
- `GLUCOSE_SETPOINT`: Proportional control on glucose setpoint with deadband and feed-rate clamping

**Perfusion Strategies** (`PerfusionStrategy` enum in perfusion_strategies.py):
- `CONSTANT`: Fixed dilution rate
- `TOXICITY_BASED_STEP`: Step increase based on lactate/ammonia thresholds
- `TOXICITY_BASED_PROPORTIONAL`: Proportional control based on metabolite levels
- `CSPR`: Cell-specific perfusion rate
- `GLUCOSE_SETPOINT`: Proportional control on glucose setpoint with deadband and dilution-rate clamping

**Gas Controllers** (`GasController` enum in gas_controllers.py):
- `CASCADE_N_FG_XO2`: Cascade control (agitation → gas flow → O2 fraction)
- `CASCADE_N_FG`: Two-stage cascade (agitation → gas flow)

**Inhibition Models** (passed as kwargs to kinetic model):
- Lactate: `"power_decay"` (default), `"linear"`, `"exponential"`
- Ammonia: `"power_decay"` (default), `"linear"`, `"exponential"`
- Osmolality: `"linear"` (default)

### Data Flow
1. **Parameters** → Load from YAML (`SimulationParameters.load_parameters("name")` or `.load_parameters()` for defaults)
2. **Model Initialization** → Create kinetic, gas, pH, feed, perfusion models via factory functions
3. **ODE System Setup** → `create_ode_systems()` builds mode-specific ODE solvers
4. **Simulation Execution** → `solve_ivp()` integrates ODEs with event-based termination
5. **Data Collection** → Results stored in `BatchData` → aggregated in `SimulationData`
6. **Analysis** → Optimization, sensitivity analysis, visualization via `analysis/` modules

## Key Conventions

### Parameter Management
- **Parameter Class**: Wrapper with `name`, `value`, `unit` supporting arithmetic operations (`+`, `-`, `*`, `/`, `**`)
- **Parameter Access**: Dict-style (`params['cell_properties.mu_max']`) or method (`params.get_parameter('cell_properties.mu_max')`)
- **Loading Standard & Custom Configurations**:
  - **Standard bundled parameters**: Names without directories/suffixes (e.g., `"base_parameters"`, `"cell_properties/cell_GFI"`) resolve under package `src/animakinetica/config/parameter_sets/`.
  - **Custom parameters**: Pass `Path` objects, relative paths (`"./my_params.yaml"`), or absolute paths (`"/path/to/my_params.yaml"`).
  - **Mixed loading**: Pass a list containing both standard bundled names and custom YAML paths; files are merged in order with later files taking precedence.
- **Saving**: `params.save_parameters("name", "description")` writes to `config/parameter_sets/name.yaml`
- **Immutability**: Parameters are frozen after simulation initialization (new instance required for changes)
- **Auto-Calculation**: Some parameters auto-calculated in `__post_init__` (e.g., `mu_max` from `td`, `F_L0` from `F_L0_specific`)

### Parameter Structure (Nested Dataclasses)
```
SimulationParameters
├── cell_properties (CellProperties): td, cell_diameter, density, wet/dry mass fractions, M_biomass, Xv_min
├── anabolic_reaction (AnabolicReaction): Stoichiometric coefficients for growth reaction (Glc, amino acids, DCM, CO2, H2O)
├── catabolic_reaction (CatabolicReaction): Stoichiometric coefficients for maintenance reaction (Glc, Gln, O2, Lac, NH3, CO2)
├── medium_properties (MediumProperties): density, osm_baseline, pH, C_O2, C_CT
├── acid_base_properties (AcidBaseProperties): pKa values, base/acid concentrations
├── kinetics (KineticsParameters): t_lag, K_Glc, K_Gln, K_O2, inhibition params (IC50, IC100, phi)
├── excess_kinetics (ExcessKineticsParameters): Y_Glc, Y_Gln, m_Glc, m_Gln, delta_q params (optional)
├── reactor (ReactorParameters): V_reactor, D_reactor, DO_setpoint, pH_setpoint, N_min/max, etc.
├── initial_conditions (InitialConditions): X_inoc, C_Glc_init, C_Gln_init, C_Lac_init, etc.
├── fed_batch (FedBatchParameters): C_Glc_feed, C_Gln_feed, C_AA_feed, F_L0_specific, r (exponential), etc. (optional)
├── chemostat (ChemostatParameters): D, Xv_chemostat_trig, feed concentrations (optional)
├── perfusion (PerfusionParameters): D_base, cell_density_setpoint, toxicity triggers (optional)
├── process_schedule (ProcessSchedule): yearly_uptime, max_culture_age
├── gas_transfer (GasTransferConstants): H_O2, H_CO2, D_L_O2, D_L_CO2 (Henry & diffusivity)
└── metadata (Dict): name, description, version, created timestamp
```

### Naming Conventions
**State Variables** (ODE system, indices in `y_points` array):
- `Xv` (index 0): Viable cell density (cells/L)
- `C_Glc` (index 1), `C_Gln` (index 2): Glucose/glutamine concentration (mmol/L)
- `C_Lac` (index 3), `C_Amm` (index 4): Lactate/ammonia concentration (mmol/L)
- `V` (index 5): Culture volume (L)
- `C_O2` (index 6): Dissolved oxygen concentration (mmol/L)
- `C_CT` (index 7): Total dissolved carbon (CO2 species, mmol/L)
- `C_H` (index 8): Hydrogen ion concentration (mol/L, for pH control)
- `C_Na` (index 9): Sodium ion concentration (mmol/L, from base addition)
- `C_AA` (index 10): Total amino acid concentration (mmol/L, excluding glutamine)

**Specific Rates** (mmol/cell/h or 1/h):
- `mu`: Specific growth rate (1/h)
- `q_Glc`, `q_Gln`, `q_O2`, `q_AA`: Substrate uptake rates (mmol/cell/h)
- `q_Lac`, `q_Amm`, `q_CO2`: Product formation rates (mmol/cell/h)

**Yields** (cells/mmol or mmol/mmol):
- `Y_Glc`, `Y_Gln`: Growth yields on glucose/glutamine (cells/mmol)
- `Y_Lac_Glc`, `Y_Amm_Gln`: Product yields (mmol product / mmol substrate)

**Control Variables**:
- `F_L`: Feed flow rate (L/h)
- `D`: Dilution rate (1/h) for perfusion/chemostat
- `N`: Agitation speed (rpm)
- `Fg`: Gas flow rate (vvm)
- `y_O2`, `y_CO2`: Mole fractions in inlet gas
- `kLa`: Volumetric mass transfer coefficient (1/h)
- `OTR`, `CTR`: Oxygen/CO2 transfer rate (mmol/L/h)

**Optimization Config Naming**: `{Mode}_{Kinetics}_{Target}_{FeedStrategy}_{PerfusionStrategy}_optimization.yaml`
- Mode: `B`=Batch, `F`=Fedbatch, `C`=Chemostat, `P`=Perfusion, `CF`=Concentrated Fedbatch, `CP`=Continuous Perfusion
- Kinetics: `S`=STOICHIOMETRIC, `E`=EXCESS
- Target: Target kinetics for optimization (if different from simulation)
- Feed Strategy: `C`=Constant, `E`=Exponential, `L`=Linear, `CSFR`=Cell-Specific, `GSP`=Glucose Setpoint
- Perfusion Strategy: `C`=Constant, `TBS`=Toxicity-Based Step, `TBP`=Toxicity-Based Proportional, `CSPR`=Cell-Specific, `GSP`=Glucose Setpoint
- Example: `F_E_S_E_TBS_optimization.yaml` = Fedbatch + EXCESS kinetics + STOICHIOMETRIC target + Exponential feed + Toxicity-Based Step perfusion

### Configuration Files (YAML)
- **Base parameters**: `src/animakinetica/config/parameter_sets/base_parameters.yaml` (mandatory parameters for all simulations)
- **Kinetics models**: `src/animakinetica/config/parameter_sets/kinetics_models/` (STOICHIOMETRIC and EXCESS kinetics parameters)
- **Stoichiometry**: `src/animakinetica/config/parameter_sets/stoichiometry/` (anabolic and catabolic reaction coefficients)
- **Cell properties**: `src/animakinetica/config/parameter_sets/cell_properties/` (cell-specific parameters)
- **Operating modes**: `src/animakinetica/config/parameter_sets/operating_modes/` (fed-batch, perfusion, chemostat parameters)
- **Parameter sets**: `src/animakinetica/config/parameter_sets/*.yaml` (custom/optimized complete parameter sets)
- **Optimization configs**: `src/animakinetica/config/optimization_config/*.yaml` (parameter bounds for optimization)
- **Schemas**: `src/animakinetica/config/schemas/parameter_schema.yaml` (validation rules)

### Using STOICHIOMETRIC Kinetic Model
**Overview**: STOICHIOMETRIC kinetics uses fixed stoichiometric yield coefficients derived from anabolic (growth) and catabolic (maintenance) reactions. Growth rate is calculated using Monod kinetics with nutrient limitation and inhibition effects, then substrate uptake and product formation rates are calculated stoichiometrically.

**Required Parameters**:
- `anabolic_reaction`: Stoichiometric coefficients for the growth reaction (conversion of substrates to biomass)
  - Substrates: Glc, amino acids (Arg, Cys, Gln, His, Ile, Lys, Met, Phe, Thr, Trp, Tyr, Val, Ala, Asn, Asp, Gly, Leu, Pro, Ser)
  - Products: DCM (dry cell mass), Glu, CO2, H2O
  - Thermodynamic properties: delta_Hf, delta_Gf
- `catabolic_reaction`: Stoichiometric coefficients for the maintenance reaction (energy metabolism)
  - Substrates: Glc, Gln, O2
  - Products: NH3, Lac, CO2, H2O
  - Thermodynamic properties: delta_Hf, delta_Gf

**Parameter Files Structure**:
```
config/parameter_sets/
├── stoichiometry/
│   ├── anab_Humbird_cell.yaml       # Anabolic reaction coefficients (Humbird source)
│   ├── anab_GFI_cell.yaml           # Alternative anabolic reaction
│   ├── cata_Humbird_wildtype.yaml   # Catabolic reaction coefficients (wildtype)
│   ├── cata_Humbird_enhanced.yaml   # Enhanced catabolic reaction
│   └── cata_GFI_cell.yaml           # Fitted catabolic reaction
├── cell_properties/
│   ├── cell_Humbird.yaml            # Cell properties (Humbird source)
│   └── cell_GFI.yaml                # Alternative cell properties
└── base_parameters.yaml             # Base parameters (required)
```

**Loading STOICHIOMETRIC Parameters** (modular approach):
```python
# Load base + stoichiometry + cell properties
params = SimulationParameters.load_parameters([
    "base_parameters",                      # Always required
  "stoichiometry/anab_GFI_cell",          # Anabolic reaction
  "stoichiometry/cata_GFI_cell",          # Catabolic reaction
  "cell_properties/cell_GFI"              # Cell properties
])

# For fed-batch mode, also load operating mode parameters
params = SimulationParameters.load_parameters([
    "base_parameters",
  "stoichiometry/anab_GFI_cell",
  "stoichiometry/cata_GFI_cell",
  "cell_properties/cell_GFI",
    "operating_modes/fed_batch"
])
```

**Calculation Flow**:
1. **Growth Rate**: μ = μ_max × NF × IF × lag_term
   - NF (Nutrient Factor): Monod terms for Glc, Gln, O2
   - IF (Inhibition Factor): Effects of O2, CO2, lactate, ammonia, osmolality
   - lag_term: Exponential lag phase (if t_lag > 0)

2. **Glucose Uptake**: q_Glc = (μ × Y_Glc/DCM) / cells_per_mmol
   - Y_Glc/DCM: Glucose consumed per dry cell mass produced (from anabolic + catabolic reactions)

3. **Other Rates** (stoichiometric from q_Glc):
   - q_Gln = q_Glc × Y_Gln/Glc
   - q_Lac = q_Glc × Y_Lac/Glc
   - q_O2 = q_Glc × Y_O2/Glc
   - q_CO2 = q_Glc × Y_CO2/Glc
   - q_AA = q_Glc × Y_AA/Glc
   - q_Amm = q_Gln × Y_Amm/Gln

**Yield Coefficients**: Automatically calculated from anabolic and catabolic reaction stoichiometry using the `StoichiometricModel` class in `models/stoichiometry.py`.

**Key Differences from EXCESS Model**:
- **STOICHIOMETRIC**: Fixed yields, simpler, requires anabolic/catabolic reactions
- **EXCESS**: Uses excess-rate kinetics for catabolic closure and then back-couples stoichiometry each ODE step to derive full yields

### Using EXCESS Kinetic Model (Current Flow)
**Overview**: EXCESS kinetics computes growth and catabolic closure from empirical excess-rate equations, then hands catabolic yield degrees of freedom to `StoichiometricModel` for atom-balanced stoichiometric back-calculation. Oxygen/CO2/amino-acid/ammonia rates are therefore still stoichiometrically consistent, but driven by excess-derived catabolic yields at each step.

**Required Parameters**:
- `excess_kinetics`: Pirt/excess terms for glucose, glutamine, and lactate rate construction
- `stoichiometric_model`: Required by EXCESS kinetics to back-calculate complete yields
- Recommended: `StoichiometryMode.EXCESS_COUPLED` so no catabolic coefficient file is required
- If not using `EXCESS_COUPLED`: catabolic section must provide yield-based fields (`Y_Lac_Glc`, `Y_Gln_Glc`) for runtime updates

**EXCESS Calculation Flow**:
1. **Growth Rate**: Compute `mu` with Monod + inhibition terms (same growth-rate framework as STOICHIOMETRIC)
2. **Catabolic Yield Closure from Excess Terms**:
  - Build catabolic glucose/glutamine/lactate rates from basal + overflow excess functions
  - Convert to apparent catabolic yields `Y_Lac_Glc_cat` and `Y_Gln_Glc_cat`
  - Enforce feasibility clamp for atom-balanced closure (prevents unphysical oxygen-producing catabolism)
3. **Stoichiometric Back-Calculation**:
  - Call `stoichiometric_model.calculate_yields(mu, (Y_Lac_Glc_cat, Y_Gln_Glc_cat))`
  - Stoichiometric model atom-balances catabolic coefficients, updates catabolic extent, and returns full 7-yield vector
4. **Rate Reconstruction**:
  - Compute `q_Glc` from `mu` and `Y_Glc_X`
  - Derive `q_Lac`, `q_Gln`, `q_O2`, `q_CO2`, `q_AA`, `q_Amm` from returned yields for a mass-consistent full rate set

## Development Workflows

### Running Simulations
```python
from animakinetica.core.simulation import CellCultureSimulation
from animakinetica.core.parameters import SimulationParameters
from animakinetica.models.kinetics import Kinetics
from animakinetica.models.odesystems import OperatingMode
from animakinetica.models.feed_strategies import FeedStrategy
from animakinetica.models.perfusion_strategies import PerfusionStrategy
from animakinetica.models.gas_controllers import GasController
from animakinetica.models.stoichiometry import StoichiometryMode

# Load parameters
params = SimulationParameters.load_parameters("base_parameters")  # or custom name

# Initialize simulation with full control
sim = CellCultureSimulation(
    parameters=params,
    kinetics=Kinetics.STOICHIOMETRIC,
    operating_mode=OperatingMode.CONCENTRATED_FEDBATCH,
    feed_strategy=FeedStrategy.EXPONENTIAL,
    perfusion_strategy=PerfusionStrategy.TOXICITY_BASED_STEP,
    gas_controller=GasController.CASCADE_N_FG_XO2,
  stoichiometry_mode=StoichiometryMode.CONSTANT,
    performance_mode=False,  # True for optimization (faster, less data)
  solver_method="LSODA",   # or "BDF" for persistently stiff systems
    inhibition_model_Lac="power_decay",  # Optional kwargs
    inhibition_model_Amm="power_decay",
    inhibition_model_Osm="linear"
)

# Run single batch or repeated batches
multi_result = sim.run_simulation(number_of_batches=10)
batch_result = multi_result.batches[0]  # First batch
# Access results
print(f"Final cell density: {batch_result.Xv_final:.2e} cells/L")
print(f"Batch duration: {batch_result.t_end:.1f} h")

# EXCESS kinetics always forces EXCESS_COUPLED stoichiometry internally
sim_excess = CellCultureSimulation(
  parameters=params,
  kinetics=Kinetics.EXCESS,
  operating_mode=OperatingMode.FEDBATCH,
  stoichiometry_mode=StoichiometryMode.DYNAMIC,  # ignored and overridden
)
```

### Parameter Optimization
```python
from animakinetica.analysis.optimization import OptimizationManager
from pathlib import Path

# Create optimization manager
opt = OptimizationManager(sim, opt_config_file="F_E_S_E_optimization")

# Single-objective optimization (maxiter/popsize auto-set based on performance_mode)
opt.optimize_parameters(
    objectives=["STY_kg_m3_day"],
    workers=-1,  # Use all CPU cores
    timeout=5.0,           # Max seconds per evaluation (default: 5.0)
    normalize_objectives=True  # Normalize by baseline values (default: True)
)

# Multi-objective optimization (weighted sum)
opt.optimize_parameters(
    objectives=["STY_kg_m3_day", "nutrient_use_efficiency_kg_kg"],
    weights=[0.7, 0.3],    # NOTE: parameter is 'weights', not 'objective_weights'
    normalize_objectives=True
)

# Compare performance (returns DataFrame with Baseline, Optimized, Improvement columns)
comparison = opt.compare_performance(verbose=True)

# Save only the optimised parameter values to a YAML file (grouped by section)
opt.save_opt_parameters(Path("config/parameter_sets/my_optimized_params.yaml"), overwrite=True)
```

### Sensitivity Analysis
```python
from animakinetica.analysis.sensitivity import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(
  simulation=sim,
  sensitivity_params=["cell_properties.td", "kinetics.K_Glc", "reactor.DO_setpoint"]
)

# 1) Validity-only mode (no objective list required)
analyzer.run_sensitivity_analysis(
  perturbation=0.2,  # ±20% variation
  n_points=21,
  create_plots=False
)

# 2) Objective mode (optional)
analyzer.run_sensitivity_analysis(
  objectives=["STY_kg_m3_day", "nutrient_use_efficiency_kg_kg"],
  obj_weights=[0.7, 0.3],
  perturbation=0.2,
  n_points=21,
  create_plots=False
)

# Plotly figures for one selected summary stat
fig_individual, fig_normalized, fig_tornado = analyzer.create_sensitivity_plots(
  summary_stat="STY_kg_m3_day"
)

# Plotly figures for all numeric summary stats
all_metric_figs = analyzer.create_sensitivity_plots_for_all_summary_stats()
```

### Visualization
```python
from animakinetica.analysis.batch_visualization import plot_batch_biomass_metabolites, plot_batch_volume_feed, plot_batch_pH_control
from animakinetica.analysis.kinetic_visualization import plot_batch_kinetics
from animakinetica.analysis.gas_visualization import plot_batch_gas_control

# Batch-level plots
fig1 = plot_batch_biomass_metabolites(sim, batch_number=1)
fig2 = plot_batch_volume_feed(sim.data, batch_number=1)
fig3 = plot_batch_pH_control(sim, batch_number=1)
fig4 = plot_batch_kinetics(sim.data, batch_number=1)
fig5 = plot_batch_gas_control(sim, batch_number=1)

fig1.show()  # Interactive Plotly figures
```

### Data Export & Serialization
```python
from animakinetica.core.simulation_data import get_script_export_dir

# Export to Excel (default: creates export/simulations/<filename>_<timestamp>/ subdirectory)
# include_pickle=True (default) also saves a .pkl alongside the .xlsx automatically
sim.data.save(
    filename="my_simulation",
    file_format="excel",
    add_timestamp=True,      # Appends YYYYMM_HHMM to folder/filename (default: True)
    include_metadata=True,
    include_summary_stats=True,
    include_cultivation_summary=True,
    include_batch_data=True,
    include_parameters=True, # Saves .parameters YAML alongside
    include_pickle=True,     # Saves .pkl alongside (default: True)
    overwrite=True,
    export_dir=None          # None = package default; or pass a custom Path
)
# Structure: export/simulations/my_simulation_YYYYMM_HHMM/
#              my_simulation_YYYYMM_HHMM.xlsx
#              my_simulation_YYYYMM_HHMM.parameters.yaml
#              my_simulation_YYYYMM_HHMM.pkl

# Export to script-relative directory (useful in campaign notebooks)
sim.data.save("my_simulation", export_dir=get_script_export_dir())
# Creates: <calling_script_dir>/simulation/my_simulation_<timestamp>/

# Export to CSV (creates directory with numbered CSV files)
sim.data.save(filename="my_simulation", file_format="csv")

# Standalone pickle save (separate from save())
sim.data.save_pickle("my_fedbatch_results", overwrite=True)
# Default: export/simulations/my_fedbatch_results.pkl
# Custom dir: sim.data.save_pickle("my_results", base_export_dir=get_script_export_dir())

# Load saved simulation data in new session
# Searches EXPORT_DIR and all subdirectories recursively; uses most recently modified if multiple
from animakinetica.core.simulation_data import SimulationData
loaded_data = SimulationData.load_pickle("my_fedbatch_results")
# Prints: simulation type, number of batches, timestamp
# All methods work: loaded_data.plot_biomass_metabolites(), etc.

# Manual dataframe access
metadata_df = sim.data.get_metadata_df()
summary_df = sim.data.get_summary_stats_df()
cultivation_df = sim.data.get_cultivation_summary_df()
batch_df = sim.data.get_batch_df(batch_number=1)
```

## Critical Implementation Details

### Event-Based Control
**Available Events** (in `events.py`):
1. `glucose_depletion_event`: Stops when glucose < threshold
2. `glutamine_depletion_event`: Stops when glutamine < threshold
3. `AA_depletion_event`: Stops when amino acids are depleted
4. `oxygen_depletion_event`: Stops when oxygen < threshold
5. `pH_deviation_event`: Stops if pH exceeds tolerance
6. `max_volume_event`: Stops at maximum reactor volume
7. `viscosity_limit_event`: Stops at maximum cell density (viscosity limit)
8. `cell_minimum_event`: Stops if viable cells < minimum (washout protection)
9. `celldensity_chemostat_trigger_event`: Triggers chemostat mode at target density
10. `celldensity_pf_trigger_event`: Triggers perfusion at cell density
11. `lactate_pf_trigger_event`: Triggers perfusion when lactate > threshold
12. `ammonia_pf_trigger_event`: Triggers perfusion when ammonia > threshold

**Usage**: Events are pre-built per ODE phase in `CellCultureSimulation._build_phase_events()` and passed to `solve_ivp(..., events=phase_events)`.

### ODE System Architecture
**Base Class**: `ODESystem` (abstract) defines interface:
- `compute_rates()`: Calculate all kinetic rates at current state
- `compute_derivatives()`: Calculate dy/dt for all state variables
- `__call__(t, y)`: Main integration function

**Concrete Implementations**:
- `BatchODESystem`: Closed system (dV/dt = 0)
- `FedBatchODESystem`: Feed without harvest (dV/dt = F_L)
- `ChemostatODESystem`: Continuous at constant volume (F_in = F_out)
- `PerfusionODESystem`: Cell retention with medium exchange

**Dynamic ODE Switching**: `CellCultureSimulation` can switch active ODE system mid-simulation (e.g., fedbatch → perfusion when trigger event fires)

### Algebraic Control Models
**Gas Control** (`gas_controllers.py`):
- Maintains DO and dCO2 at **exact setpoints** (not state variables)
- Calculates required OTR/CTR from cell rates: `OTR = q_O2 * Xv + dilution terms`
- Uses STR correlations to determine agitation speed, gas flow, O2 enrichment
- Updates `kLa` based on operating conditions (N, Fg, V, etc.)

**pH Control** (`pH_controller.py`):
- Maintains pH at **exact setpoint** using base addition
- Accounts for H+ sources: lactate production, CO2 dissolution, ammonia consumption
- Calculates base dilution rate `D_base` to neutralize acid production
- CO2 stripping via gas flow (adjusts y_CO2 within bounds)

### Performance Mode
- `performance_mode=True`: Relaxes ODE tolerances and suppresses detailed per-step logging in ODE systems for faster temporary/optimization runs
- `performance_mode=False`: Uses stricter tolerances and full logging for analysis/plotting
- Typical temporary-run behavior: use performance mode with `run_temporary_simulation(..., return_only_summary_stats=True)`
- Disable performance mode for final runs where detailed batch traces are needed

### Factory Pattern for Models
```python
# In simulation.__init__:
self.stoichiometric_model = StoichiometricModel(params, mode=stoichiometry_mode)
self.kinetic_model = create_kinetic_model(kinetics, params, **kwargs)
self.gas_control_model = create_gas_controller_model(gas_controller, params)
self.feed_model = create_feed_model(feed_strategy, params)
self.perfusion_model = create_perfusion_strategy(perfusion_strategy, params) if perfusion_strategy else None
self.ode_systems = create_ode_systems(params, operating_mode, self.stoichiometric_model, self.kinetic_model, self.gas_control_model, self.feed_model, self.perfusion_model)

# If kinetics == Kinetics.EXCESS:
# stoichiometry_mode is forced to StoichiometryMode.EXCESS_COUPLED internally.
```

### Temporary Simulation API
- `CellCultureSimulation.run_temporary_simulation(...)` creates a throwaway simulation for optimization/sensitivity sweeps
- Supports: `return_only_summary_stats`, `performance_mode`, `solver_method`, and `wall_timeout`
- `wall_timeout` can abort stalled integrations (for example threshold-chattering control scenarios)
- In performance mode, temporary runs are capped at one vessel lifetime (`max_culture_age`) for consistent evaluation horizon

### State Variables Tracked in BatchData
**Core ODE States**: `y_points` array with columns [Xv, C_Glc, C_Gln, C_Lac, C_Amm, V, C_O2, C_CT, C_H, C_Na, C_AA]

**Auxiliary Lists** (optional, not stored in performance mode):
- **Kinetic rates**: mu_list, q_Glc_list, q_Gln_list, q_Lac_list, q_Amm_list, q_O2_list, q_CO2_list
- **Nutrient factors**: NF_glucose_list, NF_glutamine_list, NF_O2_list, NF_list
- **Inhibition factors**: IF_O2_list, IF_CO2_list, IF_lactate_list, IF_ammonia_list, IF_osmolality_list, IF_list
- **Gas control**: N_list, Fg_list, y_O2_list, kLa_list, OTR_list, CTR_list, C_CO2_list, y_CO2_list
- **pH control**: D_base_list
- **Process variables**: D_list (dilution rate), osm_list (osmolality), lag_term_list
- **Material balances**: batch_medium_use, feed_medium_use, nutrient_use, harvest_volume, cells_harvested

### Pickle Serialization
**Purpose**: Save complete simulation state for later analysis without re-running

**Save/Load Methods**:
- `SimulationData.save_pickle(filename, overwrite=False, base_export_dir=None)`: Saves to `export/simulations/{filename}.pkl` or custom dir
- `SimulationData.load_pickle(filename)`: Loads from pickle with recursive search under `export/simulations/`
- `CellCultureSimulation.save_pickle(filename, overwrite=False, base_export_dir=None)`: Saves full simulation object
- `CellCultureSimulation.load_pickle(filename, base_export_dir=None)`: Loads full simulation object with recursive search under chosen root
- `SensitivityAnalyzer.save_pickle(filename, overwrite=False, export_dir=None)`: Saves sensitivity analysis object (default: `export/sensitivity/`)
- `SensitivityAnalyzer.load_pickle(filename, export_dir=None)`: Loads sensitivity analysis object with recursive search under chosen root
- `save(..., include_pickle=True)`: Automatically saves a `.pkl` alongside any Excel/CSV export (default behaviour)
- `get_script_export_dir(subdir='simulation')`: Helper returning a path relative to the **calling script's** location — use this in campaign notebooks to keep exports co-located with notebooks
- Shared implementation uses `core/io_utils.py` (`resolve_pickle_save_path`, `resolve_pickle_load_path`, `save_pickle_object`, `load_pickle_object`)
- Supports bare filename, relative path, and absolute path; appends `.pkl` automatically if missing
- If multiple matching filenames are found during recursive search, newest file is selected
- Uses Python's highest pickle protocol for best compression
- Preserves ALL data: numpy arrays, nested objects, parameters, timestamps

**Use Cases**:
1. **Long simulations**: Run expensive simulations once, analyze multiple times
2. **Sharing results**: Send pickle files to collaborators for analysis
3. **Resuming workflows**: Continue optimization/sensitivity analysis from saved state
4. **Archiving**: Store complete simulation state for reproducibility

**Compatibility**: Loaded `SimulationData` objects have full functionality - all methods (plotting, export, analysis) work identically to freshly run simulations

## Testing & Validation
- **Validation notebooks** in `validation/`:
  - `Ljunggren_Haggstrom_1992_fedbatch_excess.ipynb`: Fed-batch validation against literature
  - `Zhou_Hu_1994 - Batch excess.ipynb`: Batch mode validation
  - `GDP.ipynb`: General Data Protection validation (?)
- **Test notebooks** in `test/`: 
  - Visualization testing: batch, kinetic, gas, process modules
  - `pickle_serialization_test.ipynb`: Test save/load functionality with SimulationData
- **No automated unit tests** - validation is manual comparison to literature data

## Common Pitfalls & Debugging Tips

### Parameter-Related Issues
1. **Frozen Parameters**: Cannot modify `params.cell_properties.td = 48` after passing to simulation → Create new simulation instance
2. **Missing Parameter Sections**: Chemostat/perfusion parameters are `Optional` → Check `params.perfusion is not None` before access
3. **Auto-Calculated Parameters**: Don't set `mu_max` or `F_L0` directly → Set `td` or `F_L0_specific` instead
4. **Dict Access**: Use `params['cell_properties.mu_max']` or `params.get_parameter('cell_properties.mu_max')`, not `params.cell_properties['mu_max']`

### Simulation Setup Issues
5. **Operating Mode Mismatch**: FEDBATCH requires `FeedStrategy`, PERFUSION requires `PerfusionStrategy` → Pass correct strategy or defaults apply
6. **Event Conflicts**: Multiple events can trigger simultaneously → Check `BatchData` for termination reason
7. **Kinetics Model Selection**: STOICHIOMETRIC requires `anabolic_reaction` and `catabolic_reaction`; EXCESS requires `excess_kinetics` plus stoichiometric back-coupling (`stoichiometric_model`, typically `EXCESS_COUPLED`) → ensure the required sections/mode are loaded

### Optimization Issues
8. **Objective Function Names**: Must match `SimulationData` metric keys exactly (e.g., `"STY_kg_m3_day"` not `"STY"`)
9. **Optimization Config Bounds**: Ensure parameter paths in YAML match actual parameter structure
10. **Convergence Failures**: Increase `maxiter` or adjust bounds if optimization doesn't converge

### Numerical Issues
11. **ODE Integration Failures**: Check for negative concentrations (unphysical) → Adjust tolerances or check kinetics model
12. **Event Detection**: Events have `direction` attribute (0, 1, -1) controlling zero-crossing detection
13. **Unit Mismatches**: All internal units are cells/L, mmol/L, h, L → Convert external data accordingly

### Visualization Issues
14. **Empty Plots**: Ensure `performance_mode=False` to store detailed data lists
15. **Batch Number Indexing**: Batches are 1-indexed in user API, 0-indexed internally in lists

### Serialization Issues
16. **Pickle File Not Found**: `load_pickle()` methods search the configured root recursively and report attempted path + search root when failing.
17. **Ambiguous Relative Names**: If duplicate pickle filenames exist in subfolders, the most recently modified file is loaded.
18. **Sensitivity Legacy Pickles**: `SensitivityAnalyzer.load_pickle()` does **not** apply old-schema compatibility shims; use current-format files.
19. **Large File Sizes**: Pickle files can be large with many batches - consider saving only essential data
20. **Double Pickle on save()**: `save()` includes `include_pickle=True` by default - set `include_pickle=False` to avoid creating a redundant `.pkl` alongside the Excel export

### Optimization API Issues
20. **Wrong method name `save_optimized_parameters`**: Correct method is `save_opt_parameters(filepath: Path, overwrite=False)` — takes a full file path, not a name + description
21. **Wrong method name `compare_baseline_optimized`**: Correct method is `compare_performance(verbose=True)` — returns a DataFrame
22. **Wrong kwarg `objective_weights`**: The parameter is `weights`, not `objective_weights`
23. **Passing `maxiter`/`popsize` to `optimize_parameters`**: These are NOT arguments; they are auto-set internally based on `performance_mode` (10/25 for performance mode, 50/50 otherwise)
24. **Normalization off by default legacy**: `normalize_objectives=True` is the default — each objective is divided by its baseline value before weighting. Set `False` to use raw values.

## Package Installation & Dependencies
```bash
# Development installation (editable mode)
cd animakinetica
pip install -e .

# Required dependencies (see requirements.txt):
# - numpy: Array operations
# - scipy: ODE integration (solve_ivp), optimization (differential_evolution)
# - pandas: Data export and tabulation
# - matplotlib: Basic plotting (legacy, mostly replaced by Plotly)
# - plotly: Interactive visualization
# - pyyaml: Configuration file handling
```

## Key Performance Metrics
**Calculated in `SimulationData.summary_stats`** (used for optimization objectives):
- `total_DCM_kg`: Total dry cell mass produced (kg)
- `STY_kg_m3_day`: Space-Time Yield (kg biomass / m³ reactor / day)
- `medium_efficiency_kg_m3`: Biomass produced per medium volume (kg/m³)
- `protein_use_efficiency_kg_kg`: Biomass yield on protein/amino acid substrates (kg/kg)
- `nutrient_use_efficiency_kg_kg`: Biomass yield on all nutrients (kg/kg)
- `total_simulation_time_h`: Total cultivation time across all batches (h)
- `number_of_batches`: Number of batches completed

## File Organization
```
animakinetica/
├── export/simulations/                # Exported simulation results (Excel/CSV)
├── src/animakinetica/                 # Main package (see structure above)
│   └── config/                        # Bundled parameter & optimization YAML files
├── test/                              # Visualization test notebooks
├── validation/                        # Literature validation notebooks
├── README.md                          # User documentation
├── TODO                               # Development roadmap
└── setup.py                           # Package installation script

# Workspace root also contains:
scripts/                               # Simulation outputs organized by mode
CIP_SIP/                              # Clean-in-place/Sterilize-in-place analysis
SuperPro/                             # Commercial bioprocess software files
```

## TODO & Future Development
**From TODO file**:
- Research avian cell asparagine use (alternative to glutamine)
- Add cell death model with substrate starvation
- High waste concentration effects (death, not just inhibition)
- Clarify cells-to-gram dry/wet weight conversions
- Extend visualization to plot all data handles

**Not Implemented** (data limitations):
- Temperature effects on growth rate (no reliable correlations available)

## Quick Reference: Common Use Cases

### 1. Run Standard Fed-Batch Simulation
```python
# Load modular parameters (base + kinetics + operating mode)
params = SimulationParameters.load_parameters([
    "base_parameters",
  "stoichiometry/anab_GFI_cell",
  "stoichiometry/cata_GFI_cell",
  "cell_properties/cell_GFI",
    "operating_modes/fed_batch"
])
sim = CellCultureSimulation(params, Kinetics.STOICHIOMETRIC, OperatingMode.FEDBATCH)
results = sim.run_simulation()
```

### 2. Optimize Fed-Batch Parameters for STY
```python
from pathlib import Path

params = SimulationParameters.load_parameters([
    "base_parameters",
    "kinetics_models/excess_kinetics",
    "operating_modes/fed_batch"
])
sim = CellCultureSimulation(params, Kinetics.EXCESS, OperatingMode.FEDBATCH, performance_mode=True)
sim.run_simulation()  # First run to initialize baseline stats (required for normalization)
opt = OptimizationManager(sim, "F_E_S_E_optimization")
opt.optimize_parameters(objectives=["STY_kg_m3_day"], workers=-1)  # maxiter/popsize auto-set
# Compare baseline vs. optimized
opt.compare_performance()
# Save only the changed parameters as a YAML patch file
opt.save_opt_parameters(Path("config/parameter_sets/my_optimized_fedbatch.yaml"), overwrite=True)
```

### 3. Compare Operating Modes
```python
params = SimulationParameters.load_parameters([
    "base_parameters",
  "stoichiometry/anab_GFI_cell",
  "stoichiometry/cata_GFI_cell",
  "cell_properties/cell_GFI"
])
modes = [OperatingMode.BATCH, OperatingMode.FEDBATCH, OperatingMode.PERFUSION]
results = {}
for mode in modes:
    # Load mode-specific parameters if needed
    if mode == OperatingMode.FEDBATCH:
        params.update("operating_modes/fed_batch")
    elif mode == OperatingMode.PERFUSION:
        params.update("operating_modes/perfusion")
    
    sim = CellCultureSimulation(params, Kinetics.STOICHIOMETRIC, mode)
    results[mode] = sim.run_simulation(number_of_batches=5)
# Compare metrics across modes
```

### 4. Analyze pH and Gas Control
```python
params = SimulationParameters.load_parameters([
    "base_parameters",
  "stoichiometry/anab_GFI_cell",
  "stoichiometry/cata_GFI_cell",
  "cell_properties/cell_GFI",
    "operating_modes/fed_batch"
])
sim = CellCultureSimulation(params, Kinetics.STOICHIOMETRIC, OperatingMode.FEDBATCH, performance_mode=False)
sim.run_simulation()
from animakinetica.analysis.gas_visualization import plot_batch_gas_control
from animakinetica.analysis.batch_visualization import plot_batch_pH_control
fig1 = plot_batch_gas_control(sim, batch_number=1)
fig2 = plot_batch_pH_control(sim, batch_number=1)
```

### 5. Export Simulation Data
```python
from animakinetica.core.simulation_data import get_script_export_dir

# Excel export — also auto-saves .pkl and .parameters YAML by default
sim.data.save("my_simulation_results", file_format="excel", include_parameters=True)
# Creates: export/simulations/my_simulation_results_YYYYMM_HHMM/
#            my_simulation_results_YYYYMM_HHMM.xlsx
#            my_simulation_results_YYYYMM_HHMM.parameters.yaml
#            my_simulation_results_YYYYMM_HHMM.pkl

# Export to script-local directory (e.g. inside a campaign notebook folder)
sim.data.save("my_results", export_dir=get_script_export_dir())

# Standalone pickle save
sim.data.save_pickle("my_simulation_results", overwrite=True)
# Creates: export/simulations/my_simulation_results.pkl

# Later: Load pickle in new session for analysis (searches recursively)
from animakinetica.core.simulation_data import SimulationData
data = SimulationData.load_pickle("my_simulation_results")
# Prints simulation type, number of batches, and timestamp on load
```

### 6. Load Saved Simulations for Post-Processing
```python
from animakinetica.core.simulation_data import SimulationData

# Load previously saved simulation without re-running
# Works with filenames saved by save() (inside timestamped subdirs) or save_pickle()
data = SimulationData.load_pickle("my_fedbatch_results")
# Prints: simulation type, number of batches, timestamp

# Full access to all simulation results
print(f"Loaded {len(data.batches)} batches")
print(f"STY: {data.summary_stats['STY_kg_m3_day']:.4f} kg/m3/day")

# All visualization methods work on loaded data
from animakinetica.analysis.batch_visualization import plot_batch_biomass_metabolites
fig = plot_batch_biomass_metabolites(data, batch_number=1)

# Export loaded data to different format
data.save("reexported_results", file_format="excel")
```

### 7. Save/Load Sensitivity Analysis with Custom Directory
```python
from animakinetica.analysis.sensitivity import SensitivityAnalyzer
from animakinetica.core.simulation_data import get_script_export_dir

analyzer = SensitivityAnalyzer(sim, ["cell_properties.td", "reactor.DO_setpoint"])
analyzer.run_sensitivity_analysis(perturbation=0.2, n_points=11, create_plots=False)

# Save to script-relative directory
sens_dir = get_script_export_dir("sensitivity")
analyzer.save_pickle("fedbatch_sensitivity", export_dir=sens_dir, overwrite=True)

# Load from same custom directory
loaded = SensitivityAnalyzer.load_pickle("fedbatch_sensitivity", export_dir=sens_dir)

# Create Plotly figures from loaded analyzer
figs = loaded.create_sensitivity_plots(summary_stat="STY_kg_m3_day")
```
