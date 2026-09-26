"""Overview page: KPI cards, metadata, and summary statistics."""

from __future__ import annotations

import streamlit as st

from sources.base import DashboardModel

# Primary KPI cards shown at the top of the Overview tab.
# (key, display label, unit)
KPI_KEYS = [
    ("STY_kg_m3_day", "STY", "kg/m³/day"),
    ("feed_conversion_efficiency_kg_kg", "FCE", "kg Glc+Gln+AA/kg DCM"),
    ("protein_conversion_efficiency_kg_kg", "PCE", "kg Gln+AA/kg protein"),
    ("medium_conversion_efficiency_m3_kg", "MCE", "m³/kg DCM"),
]

# Secondary KPI cards shown below the primary row.
SECONDARY_KPI_KEYS = [
    ("total_DCM_harvested_kg", "Harvested Biomass", "kg DCM"),
    ("total_simulation_time_h", "Total Cultivation Time", "h"),
    ("number_of_batches", "Number of Batches", ""),
]


def _format_value(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _render_kpi_row(
    stats: dict,
    keys: list[tuple[str, str, str]],
    n_cols: int | None = None,
) -> None:
    cols = st.columns(n_cols or len(keys))
    for col, (key, label, unit) in zip(cols, keys):
        col.metric(
            label,
            _format_value(stats.get(key)),
            help=f"{key} [{unit}]" if unit else key,
        )


def render(model: DashboardModel) -> None:
    st.subheader("Simulation Overview")

    meta = model.metadata
    st.caption(
        f"**Type:** {meta.get('simulation_type', '—')}  |  "
        f"**Feed:** {meta.get('feed_strategy', 'NA')}  |  "
        f"**Perfusion:** {meta.get('perfusion_strategy', 'NA')}  |  "
        f"**Timestamp:** {meta.get('timestamp', '—')}"
    )

    if not model.capabilities.has_summary_stats:
        st.info("Summary statistics are not available for this result.", icon="ℹ️")
        return

    stats = model.summary_stats

    st.markdown("##### Key performance indicators")
    _render_kpi_row(stats, KPI_KEYS)

    # Secondary row: batch count from the model itself when the summary key
    # is absent (older/reduced result sets).
    if stats.get("number_of_batches") is None:
        stats = {**stats, "number_of_batches": model.n_batches}
    secondary = [
        (key, label, unit) for key, label, unit in SECONDARY_KPI_KEYS
        if stats.get(key) is not None
    ]
    _render_kpi_row(
        stats,
        secondary,
        n_cols=len(SECONDARY_KPI_KEYS),
    )

    st.markdown("#### Summary statistics")
    data = model.simulation_data
    if data is not None:
        st.dataframe(data.get_summary_stats_df(), width='stretch', hide_index=True)
    else:
        st.dataframe(
            {"Metric": list(stats.keys()),
             "Value": list(stats.values())},
            width='stretch',
            hide_index=True,
        )
