"""Single-batch sub-tabs: gas-transfer and pH-control panels (pickle-only)."""

from __future__ import annotations

import streamlit as st

from views._ui import render_figure
from sources.base import DashboardModel


def render_gas(model: DashboardModel, batch_number: int) -> None:
    """Gas transfer & hydrodynamics for one batch."""
    st.subheader("Gas Transfer & Hydrodynamics")

    if not model.capabilities.has_controller_traces:
        st.info(
            "Controller traces (N, Fg, kLa, OTR/CTR) are not available. This "
            "typically means the simulation was run in performance mode, or "
            "the data source is an Excel export.",
            icon="ℹ️",
        )
        return

    data = model.simulation_data
    if data is None:
        st.info("This sub-tab requires the pickle backend.", icon="ℹ️")
        return

    render_figure(data.plot_gas_transfer(batch_number=batch_number))


def render_ph(model: DashboardModel, batch_number: int) -> None:
    """pH control panel for one batch."""
    st.subheader("pH Control")

    if not model.capabilities.has_controller_traces:
        st.info(
            "Controller traces (base addition, C_H, C_Na) are not available. "
            "This typically means the simulation was run in performance mode, "
            "or the data source is an Excel export.",
            icon="ℹ️",
        )
        return

    data = model.simulation_data
    if data is None:
        st.info("This sub-tab requires the pickle backend.", icon="ℹ️")
        return

    render_figure(data.plot_pH_control(batch_number=batch_number))
