# AnimaKinetica Data Dashboard

A standalone [Streamlit](https://streamlit.io) app for exploring exported
cell-culture simulation results. The dashboard is a **pure consumer** of the
`animakinetica` package — it uses only the public API and contains no library
code. Pages live in `dashboard/views/` (rendered as in-app tabs by `app.py`).

## Features

- **Pickle backend**: loads `SimulationData` pickles produced by
  `sim.data.save(...)` / `sim.data.save_pickle(...)` (also accepts full
  `CellCultureSimulation` pickles) and delegates all plotting to the
  package's Plotly visualizations.
- **Excel backend**: loads `.xlsx` workbooks produced by
  `sim.data.save(file_format="excel")`. Provides KPI cards (metric labels
  are mapped back to canonical summary-stat keys), cultivation summary,
  per-batch state trajectories (Plotly Express), and the raw CHON closure
  report sheets. Controller/kinetic panels degrade gracefully with
  informative placeholders.
- **Graceful degradation**: pages check capability flags, so
  performance-mode pickles (no detailed traces) and Excel exports render
  placeholders instead of crashing.

### Pages

The app is organized into three top-level tabs that separate single-batch
(intra-batch) views from cumulative, process-level (multi-batch) views.

| Tab | Sub-tab | Pickle backend | Excel backend |
|---|---|---|---|
| Overview | — | KPI cards, summary statistics table | KPI cards (label→key mapped), summary table |
| Single Batch | Biomass & Metabolites | Biomass/metabolites figure | State-trajectory facets (Plotly Express) |
| | Volume and Feed | Volume/feed figure | Placeholder (traces not in export) |
| | Gas | Gas transfer figure | Placeholder (traces not in export) |
| | pH | pH control figure | Placeholder (traces not in export) |
| | Kinetics | Rates, NF/IF factors | Placeholder (traces not in export) |
| | Stoichiometry | Stoichiometric yields figure | Placeholder (traces not in export) |
| Full Cultivation | Cultivation Summary | Cultivation summary table | Cultivation summary table |
| | Harvest | Cumulative harvest figure | Placeholder (figure needs pickle) |
| | Medium use | Cumulative medium-use figure | Placeholder (figure needs pickle) |
| | Base use | Cumulative base-use figure | Placeholder (figure needs pickle) |
| | Gas use | Cumulative gas-use figure | Placeholder (figure needs pickle) |
| | CHON Closure | Interactive CHON closure computation | Raw closure report sheets |

## Setup

```bash
# 1. Using conda 
conda install streamlit

# 2. Using pip
pip install streamlit

```

## Usage

From the repository root:

```bash
streamlit run dashboard/app.py
```

1. In the sidebar, keep the **Pickle** backend.
2. Set the search directory (defaults to `export/simulations/`).
3. Pick a result file — pickles saved by `sim.data.save()` (inside
   timestamped export folders) or `save_pickle()` are found recursively.

## Notes

- Detailed controller/kinetic traces require a simulation run with
  `performance_mode=False`; performance-mode pickles show placeholders on
  the Single Batch → Gas, pH, Kinetics, and Stoichiometry sub-tabs and the
  CHON Closure sub-tab.
- CHON closure additionally requires the parameter set on the loaded data
  (`include_parameters=True` during export, the default).
