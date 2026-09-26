"""Parameters page: browse the SimulationParameters carried by the result.

Available for both pickle flavours (`SimulationData` pickles carry the
parameters object set at simulation init). Renders per-section tables of
name / value / unit, with optimized entries highlighted when optimization
results are present.
"""

from __future__ import annotations

import streamlit as st

from animakinetica.core.parameters import ParameterIO

from sources.base import DashboardModel

__all__ = ["render"]


def _sections(params) -> dict[str, dict]:
    """Return the structured {section: {param: {name, value, unit}}} dict."""
    return {
        section: entries
        for section, entries in ParameterIO.convert_to_structured_format(params).items()
        if section != "metadata" and entries
    }


def _optimized_paths(model: DashboardModel) -> set[str]:
    """Dotted paths of optimized parameters, empty when not optimized."""
    sim = model.simulation_object
    manager = getattr(sim, "optimization_manager", None)
    optimal = getattr(manager, "optimal_parameters", None)
    return set(optimal.keys()) if optimal else set()


def render(model: DashboardModel) -> None:
    if not model.capabilities.has_parameters:
        st.info(
            "This result does not include simulation parameters. "
            "Parameters are available for pickle exports (both "
            "`SimulationData` and full `CellCultureSimulation` pickles); "
            "Excel exports do not embed the parameter set.",
            icon="ℹ️",
        )
        return

    params = model.simulation_data.parameters
    sections = _sections(params)
    optimized_paths = _optimized_paths(model)

    st.subheader("Simulation Parameters")

    if not sections:
        st.warning("No parameter sections found in this result.", icon="⚠️")
        return

    section_names = sorted(sections.keys())
    selected = st.selectbox(
        "Parameter section",
        section_names,
        format_func=lambda s: s.replace("_", " ").title(),
        key="params_section",
    )

    rows = []
    for param_name, entry in sections[selected].items():
        path = f"{selected}.{param_name}"
        if isinstance(entry, dict) and "value" in entry:
            value = entry["value"]
        else:
            # Non-scalar entries (e.g. buffer lists) shown as raw values.
            value = entry
        rows.append({
            "Parameter": (
                entry.get("name", param_name)
                if isinstance(entry, dict) and "name" in entry else param_name
            ),
            # Stringify uniformly so the Value column stays Arrow-compatible
            # even when a section mixes scalars with buffer lists.
            "Value": str(value),
            "Unit": (
                entry.get("unit", "") or ""
                if isinstance(entry, dict) else ""
            ),
            "Optimized": path in optimized_paths,
            "Path": path,
        })

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Value": st.column_config.Column(width="medium"),
            "Optimized": st.column_config.CheckboxColumn(
                help="Changed by parameter optimization",
            ),
        },
    )

    # Full parameter dump in an expander.
    with st.expander("All sections (structured YAML view)"):
        st.json(
            {
                section: entries
                for section, entries in ParameterIO.convert_to_structured_format(
                    params
                ).items()
            },
            expanded=False,
        )
