"""Full Cultivation sub-tabs: cultivation summary, concatenated profiles,
and cumulative harvest/medium/base/gas plots."""

from __future__ import annotations

import streamlit as st

from views._ui import render_figure
from sources.base import DashboardModel


def max_batches_selector(model: DashboardModel, key: str = "max_batches") -> int:
    """Slider capped at the number of available batches (min. 1)."""
    n = max(model.n_batches, 1)
    return st.slider(
        "Max batches",
        min_value=1,
        max_value=n,
        value=n,
        step=1,
        key=key,
        help="Number of leading batches included in cumulative and "
             "concatenated profile figures.",
    )


def _require_process_totals(model: DashboardModel) -> bool:
    if not model.capabilities.has_process_totals:
        st.info("Process totals are not available for this result.", icon="ℹ️")
        return False
    return True


def render_cultivation_summary(model: DashboardModel) -> None:
    """Per-batch cultivation summary table."""
    st.subheader("Cultivation Summary")

    data = model.simulation_data
    if data is not None:
        try:
            st.dataframe(data.get_cultivation_summary_df(), width='stretch', hide_index=True)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            st.warning(f"Cultivation summary unavailable: {exc}")
        return

    cult = model.extra.get("cultivation_summary_df")
    if cult is not None:
        st.dataframe(cult, width='stretch', hide_index=True)
    else:
        st.info("Cultivation summary not found in this export.", icon="ℹ️")


def render_concatenated_profiles(model: DashboardModel, max_batches: int) -> None:
    """Concatenated campaign profiles for the selected batch range."""
    st.subheader("Concatenated Profiles")

    data = model.simulation_data
    if data is None:
        st.info(
            "Concatenated profiles require the pickle backend.",
            icon="ℹ️",
        )
        return

    try:
        figures = data.plot_profiles(max_batches=max_batches)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        st.warning(f"Concatenated profiles unavailable: {exc}")
        return

    if not figures:
        st.info("No profile figures available for this result.", icon="ℹ️")
        return

    for fig in figures:
        render_figure(fig)


def render_harvest(model: DashboardModel, max_batches: int) -> None:
    """Cumulative harvest over the batch sequence."""
    st.subheader("Cumulative Harvest")
    if not _require_process_totals(model):
        return
    data = model.simulation_data
    if data is None:
        st.info("Cumulative figures require the pickle backend.", icon="ℹ️")
        return
    render_figure(data.plot_cumulative_harvest(max_batches=max_batches))


def render_medium_use(model: DashboardModel, max_batches: int) -> None:
    """Cumulative medium use over the batch sequence."""
    st.subheader("Cumulative Medium Use")
    if not _require_process_totals(model):
        return
    data = model.simulation_data
    if data is None:
        st.info("Cumulative figures require the pickle backend.", icon="ℹ️")
        return
    render_figure(data.plot_cumulative_medium_use(max_batches=max_batches))


def render_base_use(model: DashboardModel, max_batches: int) -> None:
    """Cumulative base use over the batch sequence."""
    st.subheader("Cumulative Base Use")
    if not _require_process_totals(model):
        return
    data = model.simulation_data
    if data is None:
        st.info("Cumulative figures require the pickle backend.", icon="ℹ️")
        return
    render_figure(data.plot_cumulative_base_use(max_batches=max_batches))


def render_gas_use(model: DashboardModel, max_batches: int) -> None:
    """Cumulative gas use over the batch sequence."""
    st.subheader("Cumulative Gas Use")
    if not _require_process_totals(model):
        return
    data = model.simulation_data
    if data is None:
        st.info("Cumulative figures require the pickle backend.", icon="ℹ️")
        return
    render_figure(data.plot_cumulative_gas_use(max_batches=max_batches))
