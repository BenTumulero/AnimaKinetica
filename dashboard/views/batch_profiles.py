"""Single-batch sub-tabs: biomass/metabolites and volume/feed per batch."""

from __future__ import annotations

import streamlit as st

from views._ui import render_figure, render_state_trajectories
from sources.base import DashboardModel


def render_biomass_metabolites(model: DashboardModel, batch_number: int) -> None:
    """Biomass & metabolites trajectories for one batch."""
    st.subheader("Biomass & Metabolites")

    if not model.capabilities.has_batch_state_trajectories:
        st.info(
            "State trajectories are not available from this source.",
            icon="ℹ️",
        )
        return

    data = model.simulation_data
    if data is not None:
        # Pickle backend: delegate to the package's Plotly figures.
        render_figure(data.plot_biomass_metabolites(batch_number=batch_number))
        return

    # Excel backend: plot the batch time-series table directly.
    batch = model.get_batch(batch_number)
    if batch is None:
        st.info("Selected batch not found in this result.", icon="ℹ️")
        return
    st.caption(
        "The Excel export contains state trajectories only; detailed "
        "controller and kinetic panels require the pickle backend."
    )
    render_state_trajectories(batch)


def render_volume_feed(model: DashboardModel, batch_number: int) -> None:
    """Volume & feed trajectories for one batch."""
    st.subheader("Volume & Feed")

    if not model.capabilities.has_batch_state_trajectories:
        st.info(
            "State trajectories are not available from this source.",
            icon="ℹ️",
        )
        return

    data = model.simulation_data
    if data is not None:
        # Pickle backend: delegate to the package's Plotly figures.
        render_figure(data.plot_volume_feed(batch_number=batch_number))
        return

    # Excel backend: no dedicated volume/feed figure available.
    st.info(
        "Volume & feed trajectories require the pickle backend.",
        icon="ℹ️",
    )
