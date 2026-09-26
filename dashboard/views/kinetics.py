"""Kinetics sub-tab: rates and nutrient/inhibition factors per batch."""

from __future__ import annotations

import streamlit as st

from views._ui import render_figure
from sources.base import DashboardModel


def render(model: DashboardModel, batch_number: int) -> None:
    """Kinetic rates and factors for one batch."""
    st.subheader("Kinetics")

    if not model.capabilities.has_kinetic_traces:
        st.info(
            "Kinetic traces (mu, q-rates, NF/IF factors) are not available. "
            "This typically means the simulation was run in performance mode, "
            "or the data source is an Excel export.",
            icon="ℹ️",
        )
        return

    data = model.simulation_data
    if data is None:
        st.info("This sub-tab requires the pickle backend.", icon="ℹ️")
        return

    st.markdown("##### Comprehensive kinetics")
    render_figure(data.plot_kinetics(batch_number=batch_number))