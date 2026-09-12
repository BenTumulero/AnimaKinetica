# AnimaKinetica

AnimaKinetica is a Python framework for mechanistic simulation of animal-cell-culture processes in stirred-tank bioreactors. It models viable-cell growth, substrate uptake, lactate and ammonia formation, stoichiometric closure, gas transfer, pH control, feed and perfusion control, repeated cultivations, and process-level key performance indicators (KPIs).

The package is intended for bioprocess engineers and researchers who need reproducible, YAML-configured simulations of batch, fed-batch, chemostat, and perfusion operations.

> **Scope.** AnimaKinetica is a simulation and process-analysis tool. Model predictions depend on the selected kinetic, stoichiometric, control, and parameter assumptions; validate a configuration against relevant experimental data before using it for process decisions.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Simulation workflow](#simulation-workflow)
- [Process configurations](#process-configurations)
- [Kinetic and stoichiometric models](#kinetic-and-stoichiometric-models)
- [Parameters and YAML configuration](#parameters-and-yaml-configuration)
- [Results, KPIs, and exports](#results-kpis-and-exports)
- [Visualization](#visualization)
- [Optimization and sensitivity analysis](#optimization-and-sensitivity-analysis)
- [State variables and units](#state-variables-and-units)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Installation

AnimaKinetica requires Python 3.10 or later (supported up to Python 3.13). The package core dependencies are NumPy, pandas, SciPy, PyYAML, Plotly, xlsxwriter, and ipykernel.

### Standard / Editable Installation

```bash
git clone <repository-url>
cd animakinetica
python -m pip install -e .
```

The editable installation (`-e .`) makes `animakinetica` importable while retaining local source changes. Standard builds (`pip install .`) and wheel distributions are also fully supported with bundled default configuration and parameter sets.

To install dependencies using `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Quick start

The following example runs a detailed stoichiometric fed-batch simulation. It composes a complete configuration from the base parameters, GFI cell and reaction definitions, and fed-batch settings.

```python
from animakinetica import (
    CellCultureSimulation,
    FeedStrategy,
    Kinetics,
    OperatingMode,
    SimulationParameters,
    StoichiometryMode,
)

params = SimulationParameters.load_parameters([
    "base_parameters",
    "stoichiometry/anab_GFI",
    "stoichiometry/cata_GFI",
    "cell_properties/cell_GFI",
    "operating_modes/fed_batch",
])

sim = CellCultureSimulation(
    parameters=params,
    kinetics=Kinetics.STOICHIOMETRIC,
    operating_mode=OperatingMode.FEDBATCH,
    feed_strategy=FeedStrategy.EXPONENTIAL,
    stoichiometry_mode=StoichiometryMode.CONSTANT,
)
data = sim.run_simulation(number_of_batches=1)

print(data.summary_stats["STY_kg_m3_day"])
data.plot_biomass_metabolites(batch_number=1).show()
```

`number_of_batches=1` reports literal totals for one logical batch. Without an explicit batch count or time limit, `run_simulation()` simulates a maximum-culture-age window and projects additive totals to the configured yearly uptime; see [Simulation workflow](#simulation-workflow).

## Simulation workflow

A simulation has five stages:

1. **Compose parameters.** Load `base_parameters` plus cell, stoichiometry, kinetic, and operating-mode YAML sections in order.
2. **Configure models.** Select a kinetic formulation, operating mode, compatible feed/perfusion strategies, gas controller, and stoichiometry policy.
3. **Integrate phases.** The engine solves the applicable ODE phases with `scipy.integrate.solve_ivp`, applying terminal depletion, safety, and transition events.
4. **Accumulate results.** Each logical batch is stored as `BatchData`; the full run is represented by `SimulationData`.
5. **Analyze outputs.** Inspect KPIs, detailed trajectories, CHON closure diagnostics, Plotly figures, and Excel/CSV/pickle exports.

### Repeated batches and yearly projection

`CellCultureSimulation.run_simulation()` supports repeated logical batches and inoculum carryover. For the default call, it runs for `min(max_culture_age, yearly_uptime)` and scales additive totals to the configured yearly uptime. Ratio KPIs, including STY and resource efficiencies, are unchanged by this projection.

Use an explicit horizon when literal rather than projected totals are required:

```python
# One explicitly requested logical batch.
data = sim.run_simulation(number_of_batches=1)

# A literal 30-day window (time is in hours).
data = sim.run_simulation(t_max_simulation=30 * 24)

# Simulate the entire configured yearly uptime rather than projecting.
data = sim.run_simulation(project_yearly=False)
```

The result contains both `projected_totals_stats` and `simulated_window_totals_stats`. `summary_stats` contains the reported KPI set derived from projected totals when projection is active.

### Detailed and performance modes

Use the default `performance_mode=False` for final studies, exports, mass-balance diagnostics, and detailed plots. It retains kinetic, controller, gas-transfer, pH, and yield histories.

Use `performance_mode=True` for high-throughput optimization and sensitivity studies. It relaxes solver tolerances and omits most detailed traces; batch-level kinetic, gas, pH, and stoichiometric-yield figures are consequently unavailable. Detailed results also require `store_batches=True`, which is the default for `run_simulation()`.

## Process configurations

The `OperatingMode` enum selects the phase sequence and material balances. Some modes require a feed and/or perfusion strategy.

| Operating mode | Description | Required settings |
| --- | --- | --- |
| `OperatingMode.BATCH` | Closed culture with no culture-medium feed or harvest. Acid/base pH additions can change volume. | None |
| `OperatingMode.BATCH_WITH_STARTUP_FEDBATCH` | A fresh vessel lifetime starts with a fed-batch fill, followed by batch-only cycles. | `FeedStrategy` |
| `OperatingMode.FEDBATCH` | Feed without harvest; working volume increases with culture-medium and pH-control additions. | `FeedStrategy` |
| `OperatingMode.CHEMOSTAT` | Fed-batch startup followed by constant-volume continuous operation. | `FeedStrategy`; `chemostat` parameters |
| `OperatingMode.PERFUSION` | Constant-volume medium exchange with cell retention. | `PerfusionStrategy`; `perfusion` parameters |
| `OperatingMode.CONCENTRATED_FEDBATCH` | Fed-batch phase followed by event-triggered perfusion. | `FeedStrategy`, `PerfusionStrategy` |
| `OperatingMode.CONTINUOUS_PERFUSION` | Fed-batch startup followed by continuous perfusion with configured bleed. | `FeedStrategy`, `PerfusionStrategy` |

### Phase transitions and terminal events

The engine automatically attaches relevant terminal events to each ODE phase. Standard safety and depletion events include glucose, glutamine, amino-acid, and oxygen depletion; pH deviation; minimum viable-cell density; viscosity limit; lactate and ammonia IC100 ceilings; and, for fed-batch-derived phases, maximum reactor volume.

Optional process transitions are configured through YAML:

- `fed_batch.C_Glc_batch_trigger` prepends a batch phase and starts fed-batch when glucose falls to the trigger.
- `chemostat.Xv_chemostat_trig` switches the fed-batch startup to chemostat when viable-cell density rises to the trigger.
- `perfusion.C_Lac_pf_trig`, `perfusion.C_Amm_pf_trig`, or `perfusion.Xv_pf_trig` switch a concentrated fed-batch to perfusion.
- `perfusion.Xv_harvest_stop` can terminate concentrated-fed-batch perfusion after `perfusion.min_perfusion_runtime_h`.

### Feed strategies

`FeedStrategy` controls liquid feed for fed-batch-derived phases. All strategies cap the feed flow at `reactor.F_L_max`.

| Strategy | Control law | Key parameters |
| --- | --- | --- |
| `CONSTANT` | $F_L = F_{L0}$ | `fed_batch.F_L0_specific` |
| `EXPONENTIAL` | $F_L(t) = F_{L0}\exp(rt)$ | `fed_batch.F_L0_specific`, `fed_batch.r` |
| `LINEAR` | $F_L(t) = F_{L0} + \mathrm{feed\_slope}t$ | `fed_batch.F_L0_specific`, `fed_batch.feed_slope` |
| `CSFR` | $F_L = \mathrm{CSFR} X_v V$ | `fed_batch.CSFR` |
| `GLUCOSE_SETPOINT` | Algebraic glucose-balance control with optional proportional correction | `fed_batch.C_Glc_setpoint` and optional tuning fields |

For glucose-setpoint feed control, the nominal dilution rate is

$$
D_{\mathrm{alg}} = \frac{X_v q_{\mathrm{Glc}} + C_{\mathrm{Glc,set}}D_{\mathrm{base}}}
{C_{\mathrm{Glc,feed}} - C_{\mathrm{Glc,set}}}.
$$

The implementation applies a deadband, optional proportional correction, flow bounds, and optional slew-rate limit. The commanded feed is zero when the feed-to-setpoint denominator is too small.

### Perfusion strategies

`PerfusionStrategy` controls the culture-medium dilution rate for perfusion phases. Each strategy limits its command to `reactor.D_max`.

| Strategy | Description | Key parameters |
| --- | --- | --- |
| `CONSTANT` | Constant dilution rate. | `perfusion.D_baseline` |
| `TOXICITY_BASED_STEP` | One-way dilution-rate ratchet based on lactate/ammonia fractions of IC100. | `D_baseline`, `D_1_multiplier`, `D_2_multiplier` |
| `TOXICITY_BASED_PROPORTIONAL` | Baseline rate plus lactate and ammonia proportional terms. | `D_baseline`, `k_Lac`, `k_Amm` |
| `CSPR` | Cell-specific perfusion rate: $D = \mathrm{CSPR}X_v$. | `perfusion.CSPR` |
| `GLUCOSE_SETPOINT` | Algebraic glucose-balance dilution control with optional correction and slew limit. | `perfusion.C_Glc_setpoint` and optional tuning fields |

The toxicity step strategy only moves to higher steps during a phase. This ratchet behavior avoids rapid threshold switching near a toxicity limit.

### Gas transfer and pH control

The implemented gas controller is `GasController.CASCADE_N_FG_XO2`. It calculates the oxygen-transfer rate needed for the dissolved-oxygen setpoint, then applies the cascade:

1. increase agitation speed `N`;
2. increase gas flow `Fg`;
3. enrich the inlet oxygen fraction `y_O2`.

The controller uses the non-coalescing-medium van 't Riet correlation to estimate $k_La$. If maximum oxygen transfer is insufficient, all gas-control actuators are capped and the dissolved-oxygen state is allowed to decrease. `GasController.CASCADE_N_FG` is reserved but not implemented.

`pHControllerModel` regulates pH algebraically through gas-phase CO2, base addition, and acid addition within configured limits. Carbonate species are equilibrated algebraically; optional monoprotic non-carbonate buffers, such as phosphate or HEPES, are configured in `acid_base_properties.buffers` and tracked as additional ODE states.

## Kinetic and stoichiometric models

### Stoichiometric kinetics

`Kinetics.STOICHIOMETRIC` computes the specific growth rate from Monod limitation, inhibition, and optional lag, then derives substrate uptake and product formation from the yield vector supplied by `StoichiometricModel`.

The growth-rate framework uses the combined nutrient factor and inhibition factor:

$$
\mu = \max\left(\mu_{\max} \cdot NF \cdot IF - \mathrm{lag\_term}, 0\right).
$$

The nutrient factor combines glucose, glutamine, total amino acids excluding glutamine, and dissolved oxygen. The inhibition factor combines lactate, ammonia, oxygen, dissolved CO2, and osmolality. Lactate, ammonia, and osmolality inhibition selections can be supplied when constructing the simulation:

```python
sim = CellCultureSimulation(
    parameters=params,
    kinetics=Kinetics.STOICHIOMETRIC,
    inhibition_model_Lac="power_decay",
    inhibition_model_Amm="power_decay",
    inhibition_model_Osm="linear",
)
```

Supported forms are `"linear"`, `"exponential"`, `"hyperbolic"`, and `"power_decay"`. Ensure that the selected form's required IC50, IC100, and/or exponent parameters are present in the `kinetics` section.

### Stoichiometry modes

`StoichiometryMode` controls how yield coefficients are updated for stoichiometric kinetics.

| Mode | Behavior |
| --- | --- |
| `CONSTANT` | Calculates and caches yields at `cell_properties.mu_max`. |
| `DYNAMIC` | Recalculates yields at the current growth rate. |
| `EXCESS_COUPLED` | Updates catabolic stoichiometry from apparent catabolic yields provided by excess kinetics. |

The anabolic reaction contains biomass-growth stoichiometry. The catabolic reaction can be supplied as explicit coefficients or in yield-based form using `Y_Lac_Glc` and `Y_Gln_Glc`, from which a CHON-balanced catabolic reaction is derived.

### Excess kinetics

`Kinetics.EXCESS` uses empirical glucose and glutamine excess-rate terms to derive apparent catabolic glucose, glutamine, and lactate fluxes. These terms are feasibility-clamped and passed to the stoichiometric model, which reconstructs a full atom-balanced rate set including oxygen, carbon dioxide, amino acids, ammonia, metabolic water, and heat.

Excess kinetics requires an `excess_kinetics` parameter section and an anabolic reaction. The simulation automatically forces `StoichiometryMode.EXCESS_COUPLED`; a separate catabolic-reaction file is not required for this mode.

```python
params = SimulationParameters.load_parameters([
    "base_parameters",
    "stoichiometry/anab_GFI",
    "cell_properties/cell_GFI",
    "excess_kinetics/excess_kinetics_adjusted",
    "operating_modes/fed_batch",
])

sim = CellCultureSimulation(
    parameters=params,
    kinetics=Kinetics.EXCESS,
    operating_mode=OperatingMode.FEDBATCH,
    feed_strategy=FeedStrategy.GLUCOSE_SETPOINT,
)
```

## Parameters and YAML configuration

`SimulationParameters` is the configuration root. It stores each scalar as a `Parameter` object containing a display name, raw value, and unit. Numeric operations are supported, but use `float(parameter)` or `parameter.value` when an explicit scalar is preferred.

### Parameter-file composition

`SimulationParameters` provides modular loading using bundled standard configurations, user-defined custom YAML files, or a mix of both. Files in a list are merged in sequence, with later files overwriting matching sections or individual parameter fields. If no file in the list provides all core mandatory sections, the bundled `base_parameters` template is automatically prepended.

- **Standard bundled configurations**: Names without directory prefixes or `.yaml` extensions (e.g. `"base_parameters"`, `"cell_properties/cell_GFI"`, `"operating_modes/fed_batch"`) resolve automatically from the package's internal `config/parameter_sets/` directory.
- **Custom configurations**: User-defined YAML files can be supplied as `pathlib.Path` objects, relative paths (`"./my_params.yaml"`), or absolute paths (`"/path/to/my_params.yaml"`).
- **Mixed configurations**: You can compose a baseline using standard bundled modules and selectively patch or override specific sections with custom YAML files.

```python
from pathlib import Path
from animakinetica import SimulationParameters

# Mixing standard bundled templates with custom user parameter files:
params = SimulationParameters.load_parameters([
    "base_parameters",                        # Standard bundled base template
    "stoichiometry/anab_GFI",                 # Standard bundled anabolic stoichiometry
    "cell_properties/cell_GFI",               # Standard bundled cell properties
    Path("./custom_feed_strategy.yaml"),      # Custom user file (overwrites fed_batch section)
    "/absolute/path/to/custom_reactor.yaml",  # Custom user file (overwrites reactor section)
])
```

Common standard bundled sources include:

| Directory or file | Content |
| --- | --- |
| `base_parameters.yaml` | Required shared medium, acid/base, kinetics, reactor, initial-condition, schedule, and gas-transfer settings. |
| `stoichiometry/` | Anabolic and catabolic reaction parameter sets. |
| `cell_properties/` | Cell-line composition, mass, doubling time, and viability settings. |
| `excess_kinetics/` | EXCESS-model parameter sets. |
| `operating_modes/fed_batch.yaml` | Feed composition and feed-strategy settings. |
| `operating_modes/chemostat.yaml` | Chemostat dilution, feed composition, and transition setting. |
| `operating_modes/perfusion.yaml` | Separator, perfusion feed, strategy, bleed, and trigger settings. |
| `buffers/` | Optional parameter updates for non-carbonate buffers. |

### Accessing and modifying values

Use dotted paths to retrieve parameter objects and raw values:

```python
parameter = params["reactor.DO_setpoint"]
do_setpoint = params.get_parameter_value("reactor.DO_setpoint")

trial_params = params.create_modified_parameter_object({
    "reactor.DO_setpoint": 0.015,
    "fed_batch.C_Glc_setpoint": 6.0,
})
```

`create_modified_parameter_object()` creates an independent deep copy and recomputes dependent fields. Construct a new `CellCultureSimulation` from that copy because models cache parameter values during initialization. This is the recommended route for scenario analysis, optimization, and sensitivity studies. Direct assignment through `set_parameter_value()` does not recompute derived quantities; call `recompute_derived_parameters()` afterward if direct mutation is necessary.

Derived values include `cell_properties.mu_max`, cell dry mass, reactor working volumes, `reactor.F_L_max`, `reactor.D_max`, and `fed_batch.F_L0`.

Save a complete parameter configuration under the default parameter-set directory with:

```python
params.save_parameters(
    "my_parameter_set",
    description="Adjusted glucose setpoint scenario",
    overwrite=True,
)
```

## Results, KPIs, and exports

`run_simulation()` returns a `SimulationData` object and also assigns it to `sim.data`.

- `SimulationData.batches` contains one `BatchData` object per logical batch.
- `SimulationData.summary_stats` contains principal KPIs.
- `SimulationData.projected_totals_stats` contains additive totals at the reported projection horizon.
- `SimulationData.simulated_window_totals_stats` contains unprojected totals from the literal integration window.

Use one-based batch identifiers in public methods:

```python
batch = data.get_batch(1)
summary = data.get_summary_stats_df()
batch_table = data.get_batch_df(1)
cultivation_table = data.get_cultivation_summary_df()
```

### Key performance indicators

| Summary key | Meaning |
| --- | --- |
| `total_DCM_harvested_kg` | Productive dry cell mass harvested, excluding non-harvested cell losses. |
| `total_WCM_harvested_kg` | Productive wet cell mass harvested, excluding non-harvested cell losses. |
| `STY_kg_m3_day` | Harvested dry biomass per reactor volume and cultivation time. |
| `STY_WCM_kg_m3_day` | Harvested wet cell mass per reactor volume and cultivation time. |
| `medium_conversion_efficiency_m3_kg` | Total culture-medium input per harvested dry biomass (m³/kg DCM; lower is better). |
| `medium_conversion_efficiency_m3_kg_WCM` | Total culture-medium input per harvested wet cell mass (m³/kg WCM; lower is better). |
| `protein_conversion_efficiency_kg_kg` | Glutamine plus amino-acid input per harvested biomass protein (kg/kg; lower is better). |
| `feed_conversion_efficiency_kg_kg` | Glucose, glutamine, and amino-acid input per harvested dry biomass (kg/kg; lower is better). |
| `feed_conversion_efficiency_kg_kg_WCM` | Glucose, glutamine, and amino-acid input per harvested wet cell mass (kg/kg; lower is better). |
| `batch_time_range_h` | Difference between the longest and shortest batch duration. |

Additional totals include medium, base and acid additions; inlet, outlet, and net amounts for glucose, glutamine, amino acids, lactate, ammonium, oxygen, and total dissolved carbon; metabolic water and heat; cultivation time; batch count; and termination diagnostics.

WCM metrics supplement the DCM-based process metrics; they do not alter the
DCM basis used by kinetic, stoichiometric, and CHON-balance calculations. WCM
is calculated from `cell_properties.wet_cell_mass`. For a fixed
`dry_mass_fraction`, WCM mass and WCM STY are DCM values divided by that
fraction, while WCM conversion costs are DCM conversion costs multiplied by
it. No WCM protein-conversion KPI is reported because `protein_fraction` is
defined on a dry-mass basis; reconstructing protein through WCM would give the
same value as `protein_conversion_efficiency_kg_kg`.

### CHON closure diagnostics

`SimulationData` provides complementary CHON elemental-balance diagnostics:

- integrated stoichiometric closure from simulated reaction rates;
- external process-boundary closure from input and output streams;
- the difference between these two bases.

These reports require stored detailed traces and the parameter set used for the simulation. They are included in Excel export by default when available.

### Export data and parameter files

Export Excel or CSV data with optional parameter and pickle companions:

```python
from animakinetica.core.simulation_data import get_script_export_dir

output = sim.data.save(
    "fedbatch_results",
    file_format="excel",
    include_parameters=True,
    include_pickle=True,
    overwrite=True,
)

# Store a campaign result next to the calling script.
output = sim.data.save(
    "fedbatch_results",
    export_dir=get_script_export_dir(),
    overwrite=True,
)
```

With timestamps enabled (the default), exports are grouped in `export/simulations/<name>_<timestamp>/`. Excel exports can include metadata, summary statistics, projected and simulated-window totals, CHON closure reports, cultivation summaries, and time-series batch sheets. CSV export produces the corresponding tables as individual files.

`sim.save(...)` delegates the tabular export and, by default, serializes the complete `CellCultureSimulation` object. Use it when the saved object should later support optimization or sensitivity analysis without reinitialization.

### Pickle persistence

Pickles preserve nested objects and NumPy arrays. Load only files from trusted sources.

```python
from animakinetica.core.simulation import CellCultureSimulation
from animakinetica.core.simulation_data import SimulationData

# Complete simulation: models, configuration, and data.
sim.save_pickle("fedbatch_simulation", overwrite=True)
loaded_sim = CellCultureSimulation.load_pickle("fedbatch_simulation")

# Results only: batches, data tables, and parameter configuration.
sim.data.save_pickle("fedbatch_results", overwrite=True)
loaded_data = SimulationData.load_pickle("fedbatch_results")
```

Bare pickle names are searched recursively below the default export directory; if multiple names match, the newest file is selected.

## Visualization

Plot functions return Plotly `Figure` objects. Call `.show()` to display a figure or `.write_html(...)` to create an interactive HTML file.

```python
data.plot_biomass_metabolites(batch_number=1).show()
data.plot_volume_feed(batch_number=1).show()
data.plot_gas_transfer(batch_number=1).show()
data.plot_pH_control(batch_number=1).show()
data.plot_kinetics(batch_number=1).show()
data.plot_stoichiometric_yields(batch_number=1).show()

data.plot_cumulative_harvest(max_batches=5).show()
data.plot_cumulative_medium_use(max_batches=5).show()
data.plot_cumulative_base_use(max_batches=5).show()
data.plot_cumulative_gas_use(max_batches=5).show()
```

`SimulationData.plot_all(individual_batch=1, max_batches_to_plot=5)` returns the standard collection of available batch-level and campaign-level figures. `BatchData` also exposes direct batch plotting shortcuts, for example `batch.plot_kinetic_rates()`.

Detailed controller and kinetic figures require `performance_mode=False` and stored batches. Plots can also be imported directly from `animakinetica.analysis` when a functional rather than object-oriented interface is preferred.

## Optimization and sensitivity analysis

### Parameter optimization

`OptimizationManager` optimizes one or more `summary_stats` keys using `scipy.optimize.differential_evolution`. Productivity KPIs are maximized, while registered input-per-product conversion ratios are minimized. The search bounds are read from `config/optimization_config/`.

Run the baseline simulation before using baseline objective normalization:

```python
from pathlib import Path
from animakinetica.analysis import OptimizationManager

sim = CellCultureSimulation(
    parameters=params,
    kinetics=Kinetics.STOICHIOMETRIC,
    operating_mode=OperatingMode.FEDBATCH,
    feed_strategy=FeedStrategy.EXPONENTIAL,
    performance_mode=True,
)
sim.run_simulation()

optimizer = OptimizationManager(sim, opt_config_file="default_optimization")
optimizer.optimize_parameters(
    objectives=["STY_kg_m3_day", "feed_conversion_efficiency_kg_kg"],
    weights=[0.7, 0.3],
    workers=1,
    normalize_objectives=True,
)

comparison = optimizer.compare_performance()
optimizer.save_opt_parameters(
    Path("config/parameter_sets/optimized_fedbatch.yaml"),
    overwrite=True,
)
```

Optimization candidates are evaluated as temporary simulations. Failed integrations, invalid objective values, and hard operational validity violations receive penalties. The optimizer uses internally selected population and iteration settings according to `performance_mode`; `maxiter` and `popsize` are not arguments to `optimize_parameters()`.

On Windows, use the usual `if __name__ == "__main__":` guard when invoking parallel optimization from a script. `workers=-1` requests available CPUs; unsafe nested multiprocessing contexts fall back to serial evaluation.

### One-at-a-time sensitivity analysis

`SensitivityAnalyzer` perturbs each selected dotted parameter path independently around a completed baseline.

```python
from animakinetica.analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(
    simulation=sim,
    sensitivity_params=[
        "cell_properties.td",
        "kinetics.K_Glc",
        "reactor.DO_setpoint",
    ],
)

# Validity-only sweep.
analyzer.run_sensitivity_analysis(
    perturbation=0.2,
    n_points=11,
    create_plots=False,
)

# Optional objective aggregation.
analyzer.run_sensitivity_analysis(
    objectives=["STY_kg_m3_day", "feed_conversion_efficiency_kg_kg"],
    obj_weights=[0.7, 0.3],
    perturbation=0.2,
    n_points=11,
    create_plots=False,
)

individual, normalized, tornado = analyzer.create_sensitivity_plots(
    summary_stat="STY_kg_m3_day"
)
```

`n_points` must be odd and at least three so that the unmodified baseline factor is included. Failed temporary runs remain identifiable in stored sweep results and sensitivity plots.

## State variables and units

The core ODE state order is:

| Index | Symbol | Meaning | Unit |
| ---: | --- | --- | --- |
| 0 | `Xv` | Viable cell density | cells/L |
| 1 | `C_Glc` | Glucose concentration | mmol/L |
| 2 | `C_Gln` | Glutamine concentration | mmol/L |
| 3 | `C_Lac` | Lactate concentration | mmol/L |
| 4 | `C_Amm` | Ammonia concentration | mmol/L |
| 5 | `V` | Culture volume | L |
| 6 | `C_O2` | Dissolved oxygen concentration | mmol/L |
| 7 | `C_CT` | Total dissolved carbon concentration | mmol/L |
| 8 | `C_H` | Hydrogen-ion concentration | mol/L |
| 9 | `C_Na` | Sodium concentration from chemical addition | mmol/L |
| 10 | `C_AA` | Total amino acids excluding glutamine | mmol/L |
| 11 | `C_Osm_bg` | Background osmolality | mOsm |
| 12+ | `C_buffer_i` | Optional non-carbonate buffer total concentrations | mmol/L |

Specific rates use `1/h` for `mu` and `mmol/(cell h)` for substrate uptake and product formation. Gas transfer uses `1/h` for $k_La$, `mmol/(L h)` for OTR/CTR, rpm for agitation, and vvh for gas flow. Internal time is hours.

## Project layout

```text
animakinetica/
├── export/
│   └── simulations/               # Default simulation export location
├── src/animakinetica/
│   ├── config/                    # Bundled optimization & parameter sets
│   │   ├── optimization_config/   # Differential-evolution bounds
│   │   └── parameter_sets/        # Base and modular YAML parameter sets
│   ├── core/                      # Simulation orchestration, parameters, results, I/O
│   ├── models/                    # Kinetics, stoichiometry, ODEs, controllers, events
│   └── analysis/                  # Optimization, sensitivity, Plotly visualization
└── test/                          # Demonstration and validation notebooks
```

The notebooks in `test/` exercise stoichiometry, buffer behavior, each operating mode, visualizations, persistence, export, mass balance, and optimization. They are practical references for additional workflows.

## Limitations

- The implemented gas controller is the STR `CASCADE_N_FG_XO2` cascade. `CASCADE_N_FG` is not implemented.
- The non-carbonate buffer model supports monoprotic equilibria. Model multiprotic buffers as separate relevant equilibria if appropriate.
- Temperature effects and cell-death mechanisms beyond configured viability events are not represented by the current kinetic framework.
- Schema validation infrastructure exists, but detailed schema validation is not currently invoked by parameter loading.
- Automated tests are notebook-based; validate changes against the relevant test or literature-validation notebooks.

## Contributing

1. Create a feature branch.
2. Keep YAML parameter paths, units, and public APIs consistent with package conventions.
3. Run relevant notebooks in `test/` and verify affected exports, closures, or figures.
4. Submit a pull request that describes the scientific and software impact of the change.

## Authors and affiliation

**Ben Tumulero**  
Delft University of Technology, Bioprocess Engineering Section

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
