"""AnimaKinetica Data Dashboard — Streamlit entry point.

Run from the repository root:

    streamlit run dashboard/app.py

The dashboard is a standalone consumer of the `animakinetica` package
(public API only). It loads `SimulationData` pickles by default; an Excel
backend is wired into the source abstraction for future implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `sources` / `views` importable when launched as `streamlit run dashboard/app.py`.
sys.path.insert(0, str(Path(__file__).parent))

from sources import DashboardModel, ExcelSource, PickleSource  # noqa: E402

st.set_page_config(page_title="AnimaKinetica Dashboard", layout="wide")

DEFAULT_SEARCH_DIR_HINT = "export/simulations"


@st.cache_resource(show_spinner=False)
def load_model(source_kind: str, path_str: str) -> DashboardModel:
    """Load a result file through the selected backend (cached per path)."""
    source = PickleSource() if source_kind == "pickle" else ExcelSource()
    return source.load(Path(path_str))


def discover_files(source_kind: str, search_dir: Path) -> list[Path]:
    source = PickleSource() if source_kind == "pickle" else ExcelSource()
    return source.discover(search_dir)


def main() -> None:
    with st.sidebar:
        st.title("AnimaKinetica Data Dashboard")
        st.header("Data source")
        source_kind = st.radio("Backend", ["pickle", "excel"],
                               format_func=lambda s: {"pickle": "Pickle (full)",
                                                      "excel": "Excel (reduced)"}[s])
        search_dir_input = st.text_input(
            "Search directory", value=str((Path.cwd() / DEFAULT_SEARCH_DIR_HINT))
        )
        search_dir = Path(search_dir_input).expanduser()
        if not search_dir.exists():
            st.warning(f"Directory not found: {search_dir}")
            files: list[Path] = []
        else:
            files = discover_files(source_kind, search_dir)

        if not files:
            st.warning(
                f"No .{'pkl' if source_kind == 'pickle' else 'xlsx'} files found "
                f"under {search_dir}. Run a simulation and export with "
                "`sim.data.save(...)` first."
            )
            st.stop()

        selected_file = st.selectbox(
            "Result file",
            files,
            format_func=lambda p: f"{p.relative_to(search_dir)}",
        )

    try:
        model = load_model(source_kind, str(selected_file))
    except Exception as exc:  # noqa: BLE001 - surface load errors clearly
        st.error(f"Failed to load result file: {exc}")
        st.stop()

    # Loaded-file info and metadata, pinned to the bottom of the sidebar.
    with st.sidebar:
        st.markdown("---")
        st.caption(
            f"Loaded **{model.name}** ({model.n_batches} batches) from "
            f"`{model.source_path}`"
        )

        # Object type caption
        obj_type = model.metadata.get("pickle_object_type")
        st.caption(
            f"Pickle object type: **{obj_type or 'unknown'}**. "
        )

        # Metadata line: date, operating mode/strategy, and optimization
        # status (only knowable for full simulation pickles).
        meta = model.metadata
        timestamp = meta.get("timestamp")
        date_str = timestamp[:16].replace("T", " ") if timestamp else "unknown date"
        parts = [f"📅 {date_str}", f"Mode: {meta.get('simulation_type') or 'Unknown'}"]
        if meta.get("feed_strategy", "NA") != "NA":
            parts.append(f"Feed: {meta['feed_strategy']}")
        if meta.get("perfusion_strategy", "NA") != "NA":
            parts.append(f"Perfusion: {meta['perfusion_strategy']}")
        st.caption("  ·  ".join(parts))

        # Capability summary: which gated tabs have data for this result.
        caps = model.capabilities
        cap_parts = []
        cap_parts.append("✅ Parameters" if caps.has_parameters
                         else "🚫 Parameters")
        cap_parts.append("✅ Optimization" if caps.has_optimization
                         else "🚫 Optimization")
        st.caption(" · ".join(cap_parts))


    import views.overview as pg_overview
    import views.batch_profiles as pg_profiles
    import views.gas_ph as pg_gas_ph
    import views.kinetics as pg_kinetics
    import views.stoichiometry as pg_stoichiometry
    import views.process as pg_process
    import views.closure as pg_closure
    import views.parameters as pg_parameters
    import views.optimization as pg_optimization
    from views._ui import batch_selector

    # ------------------------------------------------------------------
    # Top-level tabs: Overview | Single Batch | Full Cultivation |
    #                 Parameters | Optimization
    # ------------------------------------------------------------------
    tabs = st.tabs(
        ["Overview", "Single Batch", "Full Cultivation", "Parameters",
         "Optimization"]
    )
    tab_overview, tab_single_batch, tab_cumulative_batch, tab_params, tab_opt = tabs

    with tab_overview:
        pg_overview.render(model)

    # ------------------------------------------------------------------
    # Single Batch: shared batch selector + per-topic sub-tabs
    # ------------------------------------------------------------------
    with tab_single_batch:
        st.subheader("Single Batch")
        batch_number = batch_selector(model, key_prefix="single_batch")

        subtabs = st.tabs(
            ["Biomass & Metabolites", "Volume and Feed", "Gas", "pH",
             "Kinetics", "Stoichiometry"]
        )
        (sub_biomass, sub_volume, sub_gas, sub_ph, sub_kinetics,
         sub_stoich) = subtabs

        with sub_biomass:
            pg_profiles.render_biomass_metabolites(model, batch_number)
        with sub_volume:
            pg_profiles.render_volume_feed(model, batch_number)
        with sub_gas:
            pg_gas_ph.render_gas(model, batch_number)
        with sub_ph:
            pg_gas_ph.render_ph(model, batch_number)
        with sub_kinetics:
            pg_kinetics.render(model, batch_number)
        with sub_stoich:
            pg_stoichiometry.render(model, batch_number)

    # ------------------------------------------------------------------
    # Full Cultivation: process-level sub-tabs
    # ------------------------------------------------------------------
    with tab_cumulative_batch:
        st.subheader("Full Cultivation")
        max_batches = pg_process.max_batches_selector(model)

        subtabs = st.tabs(
            ["Cultivation Summary", "Concatenated Profiles", "Harvest",
             "Medium use", "Base use", "Gas use", "CHON Closure"]
        )
        (sub_cult, sub_profiles, sub_harvest, sub_medium, sub_base,
         sub_gasuse, sub_closure) = subtabs

        with sub_cult:
            pg_process.render_cultivation_summary(model)
        with sub_profiles:
            pg_process.render_concatenated_profiles(model, max_batches)
        with sub_harvest:
            pg_process.render_harvest(model, max_batches)
        with sub_medium:
            pg_process.render_medium_use(model, max_batches)
        with sub_base:
            pg_process.render_base_use(model, max_batches)
        with sub_gasuse:
            pg_process.render_gas_use(model, max_batches)
        with sub_closure:
            pg_closure.render(model)

    # ------------------------------------------------------------------
    # Parameters & Optimization: capability-gated views that degrade to
    # an explanatory message when the loaded result cannot provide them.
    # ------------------------------------------------------------------
    with tab_params:
        pg_parameters.render(model)

    with tab_opt:
        pg_optimization.render(model)


if __name__ == "__main__":
    main()
