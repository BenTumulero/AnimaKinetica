"""Stoichiometry sub-tab: stoichiometric yield trajectories per batch."""

from __future__ import annotations

import streamlit as st

from views._ui import render_figure
from sources.base import DashboardModel


def render(model: DashboardModel, batch_number: int) -> None:
    """Stoichiometric yield trajectories for one batch."""
    st.subheader("Stoichiometry")

    if not model.capabilities.has_stoich_yields:
        st.info(
            "Stoichiometric yield trajectories (Y_*) are not available. This "
            "typically means the simulation was run in performance mode, or "
            "the data source is an Excel export.",
            icon="ℹ️",
        )
        return

    data = model.simulation_data
    if data is None:
        st.info("This sub-tab requires the pickle backend.", icon="ℹ️")
        return

    st.markdown("##### Stoichiometric yields")
    render_figure(data.plot_stoichiometric_yields(batch_number=batch_number))
