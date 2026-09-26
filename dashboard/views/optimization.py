"""Optimization page: results of a parameter optimization run.

Only available for full `CellCultureSimulation` pickles that were optimized
(`sim.optimized` with a linked `OptimizationManager`). All other result
types render an explanatory message instead.

Sections:
- Run summary: scoring mode, profile, optimization date, config file.
- Objectives: name, direction, weight, normalization/desirability.
- Differential evolution result: scipy OptimizeResult fields as a table.
- Optimized parameters, baseline-vs-optimized comparison, scoring config.
"""

from __future__ import annotations

import streamlit as st

from sources.base import DashboardModel

__all__ = ["render"]

_MODE_LABELS = {
    "raw": "Raw values",
    "baseline_normalized": "Baseline-normalized",
    "campaign_normalized": "Campaign-normalized",
    "engineering_desirability": "Engineering desirability",
}


def _unavailable_message(model: DashboardModel) -> str:
    """Explain why optimization data is unavailable for this result."""
    obj_type = model.metadata.get("pickle_object_type")
    if obj_type == "SimulationData":
        return (
            "This pickle contains results only (`SimulationData`). "
            "Optimization data is stored on the full simulation object — "
            "save it with `CellCultureSimulation.save_pickle()` after "
            "running `OptimizationManager.optimize_parameters()`."
        )
    if obj_type == "CellCultureSimulation":
        if not model.metadata.get("optimized"):
            return (
                "This full simulation pickle was **not optimized**. "
            )
        return (
            "This simulation was optimized, but its optimization manager "
            "carries no results (e.g. the pickle predates optimization or "
            "the manager was not serialized)."
        )
    return (
        "Optimization data is not available for this result type "
        "(Excel exports contain no optimization information)."
    )


def _fmt(value, fmt: str = "{:.4g}") -> str:
    """Format a scalar for display, with an em-dash fallback."""
    if value is None:
        return "—"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _render_run_summary(model: DashboardModel, manager, scoring: dict) -> None:
    """Header: scoring mode, profile, date, config file, baseline status."""
    meta = getattr(manager, "optimization_metadata", None) or {}

    mode = scoring.get("mode")
    mode_label = _MODE_LABELS.get(mode, mode or "—")
    profile = scoring.get("profile")

    badges = [f"**Mode:** {mode_label}"]
    if profile:
        badges.append(f"**Profile:** {profile}")

    # Optimization date: recorded at run time; fall back to the pickle's
    # file modification time for pickles saved before metadata capture.
    opt_date = meta.get("timestamp")
    if not opt_date:
        try:
            import datetime as _dt
            mtime = model.source_path.stat().st_mtime
            opt_date = _dt.datetime.fromtimestamp(mtime).strftime(
                "%Y-%m-%d %H:%M:%S (file mtime)"
            )
        except OSError:
            opt_date = None
    badges.append(f"**Optimized:** {opt_date or '—'}")

    config_file = meta.get("opt_config_file")
    if config_file:
        import os
        badges.append(f"**Config:** `{os.path.basename(config_file)}`")

    st.caption("  ·  ".join(badges))

    # Baseline evaluation status (when recorded).
    if meta.get("baseline_objective_value") is not None:
        feasible = meta.get("baseline_is_feasible")
        icon = "✅" if feasible else "⚠️"
        st.caption(
            f"{icon} Baseline objective: "
            f"{_fmt(meta.get('baseline_objective_value'))} "
            f"({'feasible' if feasible else 'rejected / infeasible'})"
        )


def _render_objectives(scoring: dict) -> None:
    """Table of objectives with directions, weights, and scaling config."""
    objectives = scoring.get("objectives", [])
    if not objectives:
        return

    directions = scoring.get("objective_directions", {})
    weights = scoring.get("weights", [])
    normalization = scoring.get("normalization_values", {})
    desirability = scoring.get("desirability", {})

    rows = []
    for i, obj in enumerate(objectives):
        weight = weights[i] if i < len(weights) else None
        row = {
            "Objective": obj,
            "Direction": directions.get(obj, "maximize"),
            "Weight": _fmt(weight),
        }
        if normalization:
            row["Normalization value"] = _fmt(normalization.get(obj))
        if desirability and obj in desirability:
            cfg = desirability[obj]
            row["Desirability"] = (
                f"min {_fmt(cfg.get('minimum'))} → target "
                f"{_fmt(cfg.get('target'))}"
            )
        rows.append(row)

    st.markdown("#### Objectives")
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_de_result(manager, result) -> None:
    """Differential evolution result fields as a table + population view."""
    st.markdown("#### Differential evolution result")

    meta = getattr(manager, "optimization_metadata", None) or {}
    rows = [
        ("Final score (fun)", _fmt(getattr(result, "fun", None))),
        ("Success", str(getattr(result, "success", "—"))),
        ("Message", str(getattr(result, "message", "—"))),
        ("Function evaluations (nfev)", _fmt(getattr(result, "nfev", None), "{:d}")),
        ("Generations (nit)", _fmt(getattr(result, "nit", None), "{:d}")),
    ]
    maxcv = getattr(result, "maxcv", None)
    if maxcv is not None:
        rows.append(("Max constraint violation (maxcv)", _fmt(maxcv)))

    # Solver settings from the persisted run metadata.
    for key, label in [
        ("maxiter", "Max iterations"),
        ("popsize", "Population size factor"),
        ("tol", "Tolerance"),
        ("workers", "Workers"),
        ("seed", "Seed"),
        ("timeout_s", "Per-eval timeout (s)"),
    ]:
        if key in meta:
            rows.append((label, _fmt(meta[key])))

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
    )

    # Population snapshot: best members mapped back to parameter names.
    population = getattr(result, "population", None)
    energies = getattr(result, "population_energies", None)
    opt_config = getattr(manager, "opt_config", None) or {}
    if population is not None and energies is not None and opt_config:
        param_names = list(opt_config.keys())
        with st.expander(f"Final population ({len(energies)} members, best first)"):
            pop_rows = []
            for i, (energy, vector) in enumerate(
                zip(energies, population), start=1
            ):
                row = {"Rank": i, "Score": _fmt(energy)}
                for name, value in zip(param_names, vector):
                    row[name] = float(value)
                pop_rows.append(row)
            st.dataframe(pop_rows, width="stretch", hide_index=True)


def render(model: DashboardModel) -> None:
    if not model.capabilities.has_optimization:
        st.info(_unavailable_message(model), icon="ℹ️")
        return

    manager = model.simulation_object.optimization_manager
    result = manager.optimization_result
    optimal = manager.optimal_parameters
    scoring = getattr(manager, "optimization_scoring", None) or {}

    st.subheader("Optimization Results")

    # ------------------------------------------------------------------
    # Run summary: mode, profile, date, config, baseline status
    # ------------------------------------------------------------------
    _render_run_summary(model, manager, scoring)
    _render_objectives(scoring)

    # ------------------------------------------------------------------
    # Differential evolution result table
    # ------------------------------------------------------------------
    if result is not None:
        _render_de_result(manager, result)

    # ------------------------------------------------------------------
    # Optimized parameters
    # ------------------------------------------------------------------
    st.markdown("#### Optimized parameters")
    param_rows = []
    for path, value in optimal.items():
        try:
            param = manager.optimized_simulation.parameters.get_parameter(path)
            name, unit = param.name, param.unit or ""
        except (ValueError, AttributeError):
            name, unit = path.split(".")[-1], ""
        param_rows.append({
            "Parameter": name,
            "Value": float(value),
            "Unit": unit,
            "Path": path,
        })
    st.dataframe(param_rows, width="stretch", hide_index=True)

    # ------------------------------------------------------------------
    # Baseline vs optimized comparison
    # ------------------------------------------------------------------
    st.markdown("#### Baseline vs. optimized performance")
    try:
        comparison = manager.compare_performance(verbose=False)
        st.dataframe(comparison, width="stretch", hide_index=True)

        # Improvement bar chart for the largest numeric improvements.
        numeric = comparison.dropna(subset=["Improvement"])
        if not numeric.empty:
            top = numeric.reindex(
                numeric["Improvement"].abs().sort_values(ascending=False).index
            ).head(15)
            import plotly.express as px

            fig = px.bar(
                top,
                x="Improvement",
                y="Metric",
                orientation="h",
                title="Top improvements (direction-aware)",
                height=max(300, 28 * len(top)),
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
    except Exception as exc:  # noqa: BLE001 - comparison is best-effort
        st.warning(f"Performance comparison unavailable: {exc}", icon="⚠️")

    # ------------------------------------------------------------------
    # Scoring configuration
    # ------------------------------------------------------------------
    if scoring:
        with st.expander("Scoring configuration"):
            st.json(scoring, expanded=False)
