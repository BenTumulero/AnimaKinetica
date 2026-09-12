"""One-at-a-time parameter sensitivity analysis for cell-culture simulations.

The :class:`SensitivityAnalyzer` perturbs selected
:class:`~animakinetica.core.parameters.SimulationParameters` around a
previously simulated baseline. Each perturbation is evaluated through
:meth:`~animakinetica.core.simulation.CellCultureSimulation.run_temporary_simulation`.
The analyzer records valid summary-statistic results, optionally aggregates
objectives with user-supplied weights, and produces Plotly sensitivity,
tornado, and cross-KPI heatmap figures.

Parameters are identified using the dotted paths accepted by
:meth:`~animakinetica.core.parameters.SimulationParameters.get_parameter`, for
example ``"kinetics.K_Glc"``.
"""

from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from animakinetica.core.io_utils import (
    load_pickle_object,
    resolve_pickle_load_path,
    resolve_pickle_save_path,
    save_pickle_object,
)
from animakinetica.core.simulation import CellCultureSimulation
from animakinetica.core.simulation_data import SUMMARY_METRIC_DIRECTIONS

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = ["SensitivityAnalyzer"]

# Export directory for sensitivity analysis pickle files
SENSITIVITY_EXPORT_DIR = Path(__file__).parent.parent.parent.parent / "export" / "sensitivity"
DEFAULT_OBJECTIVE = "STY_kg_m3_day"
DEFAULT_PLOTTABLE_SENSITIVITY_STATS = {
    "total_DCM_kg",
    "total_WCM_harvested_kg",
    "STY_kg_m3_day",
    "STY_WCM_kg_m3_day",
    "medium_conversion_efficiency_m3_kg",
    "medium_conversion_efficiency_m3_kg_WCM",
    "protein_conversion_efficiency_kg_kg",
    "feed_conversion_efficiency_kg_kg",
    "feed_conversion_efficiency_kg_kg_WCM",
    "total_simulation_time_h",
}

# Human-readable labels for summary-stat keys used in plot titles/axes.
KPI_LABELS = {
    "weighted_objective": "Weighted Objective",
    "total_DCM_kg": "Total Biomass (DCM) [kg]",
    "total_WCM_harvested_kg": "Harvested Biomass (WCM) [kg]",
    "STY_kg_m3_day": "Space-Time Yield (STY) [kg/m3/day]",
    "STY_WCM_kg_m3_day": "Wet-Cell-Mass STY [kg WCM/m3/day]",
    "medium_conversion_efficiency_m3_kg": "Medium Conversion Efficiency [m3/kg DCM]",
    "medium_conversion_efficiency_m3_kg_WCM": "WCM Medium Conversion Efficiency [m3/kg WCM]",
    "protein_conversion_efficiency_kg_kg": "Protein Conversion Efficiency [kg Gln+AA/kg protein]",
    "feed_conversion_efficiency_kg_kg": "Feed Conversion Efficiency [kg Glc+Gln+AA/kg DCM]",
    "feed_conversion_efficiency_kg_kg_WCM": "WCM Feed Conversion Efficiency [kg Glc+Gln+AA/kg WCM]",
    "total_simulation_time_h": "Total Cultivation Time [h]",
    "number_of_batches": "Number of Batches",
    "total_base_use_L": "Total Base Addition [L]",
}

KPI_LABELS_NO_UNITS = {
    "weighted_objective": "Weighted Objective",
    "total_DCM_kg": "Total Biomass (DCM)",
    "total_WCM_harvested_kg": "Harvested Biomass (WCM)",
    "STY_kg_m3_day": "Space-Time Yield (STY)",
    "STY_WCM_kg_m3_day": "Wet-Cell-Mass STY",
    "medium_conversion_efficiency_m3_kg": "Medium Conversion Efficiency",
    "medium_conversion_efficiency_m3_kg_WCM": "WCM Medium Conversion Efficiency",
    "protein_conversion_efficiency_kg_kg": "Protein Conversion Efficiency",
    "feed_conversion_efficiency_kg_kg": "Feed Conversion Efficiency",
    "feed_conversion_efficiency_kg_kg_WCM": "WCM Feed Conversion Efficiency",
    "total_simulation_time_h": "Total Cultivation Time",
    "number_of_batches": "Number of Batches",
    "total_base_use_L": "Total Base Addition",
}


class SensitivityAnalyzer:
    """Run and visualize one-at-a-time parameter sensitivity sweeps.

    A completed :class:`~animakinetica.core.simulation.CellCultureSimulation`
    supplies the baseline parameters and summary statistics. In objective mode,
    :meth:`run_sensitivity_analysis` forms a weighted sum of selected summary
    statistics. Without objectives, the analyzer records successful temporary
    simulations and their individual metrics without objective aggregation.

    Parameters
    ----------
    simulation : CellCultureSimulation
        Completed baseline simulation. Its ``data.summary_stats`` dictionary is
        used as the reference for validation, normalization, and plots.
    sensitivity_params : list of str
        Non-empty list of parameter paths accepted by
        :meth:`~animakinetica.core.parameters.SimulationParameters.get_parameter`.

    Raises
    ------
    ValueError
        If the simulation has not been run, no parameter paths are supplied, or
        a path does not resolve to a parameter.
    """

    def __init__(self, simulation: CellCultureSimulation, sensitivity_params: List[str]):
        """Initialize the analyzer from a completed baseline simulation.

        Parameters
        ----------
        simulation : CellCultureSimulation
            Simulation with populated ``data.summary_stats``.
        sensitivity_params : list of str
            Dotted parameter paths, for example ``"kinetics.K_Glc"``.
        """
        self.baseline_simulation = simulation
        if self.baseline_simulation.data is None:
            raise ValueError("Simulation must be run before sensitivity analysis.")

        self.baseline_parameters = self.baseline_simulation.parameters
        self.sensitivity_params = sensitivity_params
        self.baseline_summary_stats = dict(self.baseline_simulation.data.summary_stats)

        self._validate_sensitivity_parameters()

        # Runtime configuration
        self.objective_mode: bool = False
        self.objectives: List[str] = []
        self.weights: List[float] = []
        self.perturbation: Optional[float] = None
        self.n_points: Optional[int] = None
        self.factors: Optional[np.ndarray] = None
        self.timestamp: Optional[datetime] = None

        # Summary-stat metadata for plotting
        self.available_summary_stats: List[str] = sorted(self.baseline_summary_stats.keys())
        self.all_numeric_summary_stats: List[str] = self._discover_numeric_summary_stats()
        self.numeric_summary_stats: List[str] = self._filter_default_plottable_summary_stats(
            self.all_numeric_summary_stats
        )

        # Results
        self.detailed_sensitivity: Dict[str, Any] = {}

    @staticmethod
    def _is_numeric_finite(value: Any) -> bool:
        """Return whether a value is finite numeric data rather than a boolean."""
        if isinstance(value, bool) or not isinstance(value, Number):
            return False
        return np.isfinite(float(value))

    @staticmethod
    def _is_valid_summary_stats(summary_stats: Any) -> bool:
        """Return whether a summary-statistic payload contains finite numeric data."""
        if not isinstance(summary_stats, dict) or not summary_stats:
            return False
        return any(
            isinstance(v, Number) and not isinstance(v, bool) and np.isfinite(float(v))
            for v in summary_stats.values()
        )

    def _discover_numeric_summary_stats(self) -> List[str]:
        """Return finite numeric keys from the baseline summary statistics."""
        keys: List[str] = []
        for key, value in self.baseline_summary_stats.items():
            if self._is_numeric_finite(value):
                keys.append(key)
        return keys

    def _ensure_summary_stat_metadata(self):
        """Restore and refresh plot metadata, including for legacy pickles."""
        if not hasattr(self, "baseline_summary_stats") or not isinstance(self.baseline_summary_stats, dict):
            sim_data = getattr(getattr(self, "baseline_simulation", None), "data", None)
            if sim_data is not None and isinstance(getattr(sim_data, "summary_stats", None), dict):
                self.baseline_summary_stats = dict(sim_data.summary_stats)
            else:
                self.baseline_summary_stats = {}

        if not hasattr(self, "available_summary_stats") or not isinstance(self.available_summary_stats, list):
            self.available_summary_stats = sorted(self.baseline_summary_stats.keys())

        if not hasattr(self, "all_numeric_summary_stats") or not isinstance(self.all_numeric_summary_stats, list):
            self.all_numeric_summary_stats = self._discover_numeric_summary_stats()

        # Always refresh curated defaults so legacy pickles with broad numeric lists
        # do not reintroduce low-information metrics into default all-metric plots.
        self.numeric_summary_stats = self._filter_default_plottable_summary_stats(
            self.all_numeric_summary_stats
        )

    @staticmethod
    def _filter_default_plottable_summary_stats(numeric_keys: List[str]) -> List[str]:
        """Select default KPI keys, falling back to all supplied numeric keys."""
        default_stats = [k for k in numeric_keys if k in DEFAULT_PLOTTABLE_SENSITIVITY_STATS]
        return default_stats if default_stats else list(numeric_keys)

    # Input validation methods
    def _validate_sensitivity_parameters(self):
        """Validate parameter paths against the baseline configuration."""
        if not self.sensitivity_params:
            raise ValueError("At least one parameter name must be provided")
        if not isinstance(self.sensitivity_params, list):
            raise ValueError("sensitivity_params must be a list")

        for param_name in self.sensitivity_params:
            try:
                self.baseline_parameters.get_parameter(param_name)
            except AttributeError as exc:
                raise ValueError(
                    f"Parameter '{param_name}' not found in SimulationParameters."
                ) from exc

    def _validate_objectives(self):
        """Validate objective keys and their corresponding finite weights."""
        if not self.objectives:
            raise ValueError(
                "At least one objective must be provided in objective mode. "
                f"Available objectives: {self.available_summary_stats}"
            )
        if not isinstance(self.objectives, list):
            raise ValueError("objectives must be a list.")

        for obj in self.objectives:
            if obj not in self.baseline_summary_stats:
                raise KeyError(
                    f"Objective '{obj}' not found in summary statistics. "
                    f"Available stats: {self.available_summary_stats}"
                )

        if not isinstance(self.weights, list):
            raise ValueError("obj_weights must be a list when provided.")
        if len(self.weights) != len(self.objectives):
            raise ValueError(
                f"Length mismatch: got {len(self.objectives)} objectives and {len(self.weights)} weights."
            )
        if any(not self._is_numeric_finite(w) for w in self.weights):
            raise ValueError("All objective weights must be finite numeric values.")

    def _validate_sensitivity_inputs(self):
        """Validate configured objectives, perturbation range, and point count."""
        if self.objective_mode:
            self._validate_objectives()
        if not 0 < float(self.perturbation) < 1:
            raise ValueError(f"Perturbation must be between 0 and 1, got: {self.perturbation}")
        if int(self.n_points) < 3:
            raise ValueError(f"At least 3 points are needed for meaningful analysis, got: {self.n_points}")
        if int(self.n_points) % 2 == 0:
            raise ValueError(
                f"Number of points should be odd to include the current value, got: {self.n_points}"
            )

    def run_sensitivity_analysis(
        self,
        objectives: Optional[List[str]] = None,
        obj_weights: Optional[List[float]] = None,
        perturbation: float = 0.5,
        n_points: int = 11,
        create_plots: bool = True,
        verbose: bool = True,
        summary_stat: Optional[str] = None,
    ):
        """Run a one-at-a-time relative perturbation sweep.

        Temporary simulations are run for every selected parameter and factor.
        Failed runs remain in the recorded data with a false success flag. If
        ``create_plots`` is true, this method creates figures through
        :meth:`create_sensitivity_plots` but does not display them.

        Parameters
        ----------
        objectives : list of str, optional
            Summary-statistic keys to combine. Omit or pass an empty list for
            validity-only mode.
        obj_weights : list of float, optional
            Weights corresponding to ``objectives``. Defaults to ``1.0`` for
            every objective in objective mode.
        perturbation : float, default=0.5
            Relative variation about each baseline value. Must be strictly
            between 0 and 1; ``0.5`` evaluates factors from 0.5 to 1.5.
        n_points : int, default=11
            Odd number of factors per parameter. An odd count includes the
            unmodified baseline factor of 1.0.
        create_plots : bool, default=True
            Whether to create the three figures for ``summary_stat`` after the
            sweep.
        verbose : bool, default=True
            Whether to print progress and temporary-simulation failure details.
        summary_stat : str, optional
            Metric passed to :meth:`create_sensitivity_plots` when
            ``create_plots`` is true. The analyzer selects a mode-appropriate
            default when omitted.

        Raises
        ------
        ValueError
            If the perturbation or point count is invalid, or objective weights
            are invalid.
        KeyError
            If an objective is not present in the baseline summary statistics.
        """
        self.objectives = list(objectives) if objectives else []
        self.objective_mode = len(self.objectives) > 0

        if self.objective_mode:
            self.weights = [1.0] * len(self.objectives) if obj_weights is None else list(obj_weights)
        else:
            self.weights = []

        self.perturbation = perturbation
        self.n_points = n_points
        self.factors = np.linspace(1 - self.perturbation, 1 + self.perturbation, self.n_points)
        self.timestamp = datetime.now()

        # Reset previous results
        self.detailed_sensitivity = {}

        self._validate_sensitivity_inputs()
        self._perform_sensitivity_analysis(verbose=verbose)

        if create_plots:
            self.create_sensitivity_plots(summary_stat=summary_stat)

    def _perform_sensitivity_analysis(self, verbose: bool = True):
        """Evaluate configured factors for every parameter and store the results."""
        baseline_obj_value = None
        if self.objective_mode:
            baseline_obj_value = self._calc_objective_value(self.baseline_summary_stats)

        if verbose:
            if self.objective_mode:
                print(
                    f"\n Starting sensitivity analysis with objective(s): "
                    f"{', '.join(self.objectives)}, with weights {self.weights}."
                )
                print(f"✓ Baseline objective value: {baseline_obj_value:.4g}")
            else:
                print("\n Starting sensitivity analysis in validity-only mode (no objective aggregation).")

            print(
                f"✓ Perturbation: ±{self.perturbation * 100:.0f}% "
                f"(range: {1 - self.perturbation:.2f}x to {1 + self.perturbation:.2f}x)"
            )
            print("Starting parameter sweep...\n")

        no_params = len(self.sensitivity_params)
        total_sims = no_params * self.n_points
        completed_sims = 0

        for i, param_name in enumerate(self.sensitivity_params, start=1):
            if verbose:
                print(
                    f"[{i}/{no_params}] Analyzing {param_name}... "
                    f"({completed_sims}/{total_sims} simulations completed)"
                )

            param_results = self._analyze_single_parameter(param_name, debug_failures=verbose)
            self._store_parameter_results(param_name, param_results)
            completed_sims += self.n_points

            if verbose:
                n_success = int(np.sum(param_results["success_flags"]))
                success_rate = (n_success / len(param_results["success_flags"]) * 100)
                print(f"    ✓ Success rate: {success_rate:.1f}%")
                if n_success == 0:
                    print(
                        "    ! All temporary runs failed for this parameter. "
                        "See temporary-simulation debug lines above for root cause."
                    )

        if verbose:
            print("\n✅ Sensitivity analysis complete!")
            print(f"Total simulations run: {completed_sims}")

    def _analyze_single_parameter(self, param_name: str, debug_failures: bool = False) -> Dict[str, Any]:
        """Evaluate one parameter over all configured perturbation factors.

        Returns a result dictionary containing parameter values, validity flags,
        per-run summary statistics, and objective values when objective mode is
        active. Exceptions from temporary simulations are represented as failed
        points rather than propagated.
        """
        param_values = np.zeros(self.n_points)
        obj_values = np.full(self.n_points, np.nan)
        success_flags = np.zeros(self.n_points, dtype=bool)
        summary_stats_runs: List[Optional[Dict[str, Any]]] = [None] * self.n_points

        for i, factor in enumerate(self.factors):
            param_values[i] = self.baseline_parameters[param_name] * factor
            test_param = {param_name: param_values[i]}
            modified_param_obj = self.baseline_simulation.parameters.create_modified_parameter_object(test_param)

            try:
                sim_payload = CellCultureSimulation.run_temporary_simulation(
                    params=modified_param_obj,
                    kinetics=self.baseline_simulation.kinetics,
                    gas_controller=self.baseline_simulation.gas_controller,
                    feed_strategy=self.baseline_simulation.feed_strategy,
                    perfusion_strategy=self.baseline_simulation.perfusion_strategy,
                    operating_mode=self.baseline_simulation.operating_mode,
                    stoichiometry_mode=self.baseline_simulation.stoichiometry_mode,
                    performance_mode=self.baseline_simulation.performance_mode,
                    solver_method=self.baseline_simulation.solver_method,
                    return_only_summary_stats=True,
                    debug_failures=debug_failures,
                    **self.baseline_simulation._kwargs,
                )
            except Exception:
                continue

            sim_summary_stats = (
                sim_payload.get("summary_stats", {})
                if isinstance(sim_payload, dict)
                else {}
            )

            if not self._is_valid_summary_stats(sim_summary_stats):
                continue

            summary_stats_runs[i] = dict(sim_summary_stats)
            success_flags[i] = True

            if self.objective_mode:
                try:
                    weighted_sum = self._calc_objective_value(sim_summary_stats)
                except KeyError:
                    success_flags[i] = False
                    continue

                if self._is_numeric_finite(weighted_sum):
                    obj_values[i] = float(weighted_sum)
                else:
                    success_flags[i] = False

        if self.objective_mode:
            objective_valid_mask = np.isfinite(obj_values)
            success_flags = success_flags & objective_valid_mask
            obj_values[~success_flags] = np.nan

        return {
            "param_values": param_values,
            "obj_values": obj_values,
            "success_flags": success_flags,
            "summary_stats_runs": summary_stats_runs,
            "baseline_value": self.baseline_parameters[param_name],
        }

    def _calc_objective_value(self, summary_stats: Dict[str, float]) -> float:
        """Return a direction-aware weighted objective score from summary statistics."""
        weighted_sum = 0.0
        for obj, weight in zip(self.objectives, self.weights):
            value = float(summary_stats[obj])
            direction = SUMMARY_METRIC_DIRECTIONS.get(obj, "maximize")
            score_value = value if direction == "maximize" else -value
            weighted_sum += float(weight) * score_value
        return weighted_sum

    def _store_parameter_results(self, param_name: str, param_results: Dict[str, Any]):
        """Store one parameter's results and normalized objective when applicable."""
        self.detailed_sensitivity[param_name] = {
            "param_values": param_results["param_values"],
            "obj_values": param_results["obj_values"],
            "success_flags": param_results["success_flags"],
            "summary_stats_runs": param_results["summary_stats_runs"],
            "baseline_value": param_results["baseline_value"],
        }

        if self.objective_mode:
            baseline_obj = self._calc_objective_value(self.baseline_summary_stats)
            if self._is_numeric_finite(baseline_obj) and baseline_obj != 0:
                self.detailed_sensitivity[param_name]["obj_normalized"] = (
                    param_results["obj_values"] / baseline_obj
                )
            else:
                self.detailed_sensitivity[param_name]["obj_normalized"] = np.full(self.n_points, np.nan)


    # Pickle serialization methods
    def save_pickle(
        self,
        filename: Union[str, Path],
        overwrite: bool = False,
        export_dir: Optional[Path] = None,
    ) -> Path:
        """Serialize this analyzer and its results to a pickle file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Bare filename, path relative to ``export_dir``, or absolute output
            path. The ``.pkl`` suffix is added when absent.
        overwrite : bool, default=False
            Whether to replace an existing target file.
        export_dir : pathlib.Path, optional
            Root directory used for bare and relative filenames. Defaults to
            :data:`SENSITIVITY_EXPORT_DIR`.

        Returns
        -------
        pathlib.Path
            Resolved path of the saved pickle file.

        Raises
        ------
        ValueError
            If no sensitivity sweep has been completed.
        FileExistsError
            If the target exists and ``overwrite`` is false.
        """
        if not self.detailed_sensitivity:
            raise ValueError("No sensitivity data to save. Run run_sensitivity_analysis() first.")

        export_root = Path(export_dir) if export_dir is not None else SENSITIVITY_EXPORT_DIR
        filepath = resolve_pickle_save_path(filename, export_root)
        filepath = save_pickle_object(self, filepath, overwrite=overwrite)

        print(f"SensitivityAnalyzer saved to: {filepath}")
        return filepath

    @classmethod
    def load_pickle(
        cls,
        filename: Union[str, Path],
        export_dir: Optional[Path] = None,
    ) -> "SensitivityAnalyzer":
        """Load a serialized analyzer and restore its plotting metadata.

        Parameters
        ----------
        filename : str or pathlib.Path
            Bare filename, path relative to ``export_dir``, or absolute pickle
            path. Bare names are searched recursively below ``export_dir``.
        export_dir : pathlib.Path, optional
            Search root for bare and relative filenames. Defaults to
            :data:`SENSITIVITY_EXPORT_DIR`.

        Returns
        -------
        SensitivityAnalyzer
            Loaded analyzer with summary-statistic plotting metadata refreshed
            for compatibility with older pickle files.
        """
        search_root = Path(export_dir) if export_dir is not None else SENSITIVITY_EXPORT_DIR
        resolved_path = resolve_pickle_load_path(filename, search_root)
        analyzer = load_pickle_object(resolved_path, expected_type=cls)
        analyzer._ensure_summary_stat_metadata()

        print(f"SensitivityAnalyzer loaded from: {resolved_path}")
        print(f"  - Parameters analysed : {len(analyzer.sensitivity_params)}")
        print(f"  - Objective mode      : {analyzer.objective_mode}")
        print(f"  - Objectives          : {analyzer.objectives}")
        print(f"  - Perturbation        : ±{analyzer.perturbation * 100:.0f}%")
        print(f"  - Timestamp           : {analyzer.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return analyzer


    # Metric extraction helpers for plotting
    def _resolve_plot_metric(self, summary_stat: Optional[str] = None) -> str:
        """Return the metric selected explicitly or by the current analysis mode."""
        self._ensure_summary_stat_metadata()

        if summary_stat is not None:
            if summary_stat == "weighted_objective":
                if not self.objective_mode:
                    raise ValueError("weighted_objective requires objective mode.")
                return "weighted_objective"

            if summary_stat not in self.available_summary_stats:
                raise KeyError(
                    f"summary_stat '{summary_stat}' not found. "
                    f"Available summary stats: {self.available_summary_stats}"
                )
            if summary_stat not in self.all_numeric_summary_stats:
                raise ValueError(f"summary_stat '{summary_stat}' is not numeric and cannot be plotted.")
            return summary_stat

        if self.objective_mode:
            return self.objectives[0] if len(self.objectives) == 1 else "weighted_objective"

        if DEFAULT_OBJECTIVE in self.numeric_summary_stats:
            return DEFAULT_OBJECTIVE
        if self.numeric_summary_stats:
            return self.numeric_summary_stats[0]
        if self.all_numeric_summary_stats:
            return self.all_numeric_summary_stats[0]
        raise ValueError("No numeric summary stats available for plotting.")

    def _get_metric_baseline_value(self, metric: str) -> float:
        """Return the baseline value for a summary statistic or weighted objective."""
        if metric == "weighted_objective":
            return float(self._calc_objective_value(self.baseline_summary_stats))
        return float(self.baseline_summary_stats[metric])

    def _extract_metric_series(self, param_name: str, metric: str) -> np.ndarray:
        """Return one metric across a parameter's perturbation factors.

        Failed evaluations and invalid metric values are represented by
        ``numpy.nan``.
        """
        data = self.detailed_sensitivity[param_name]
        values = np.full(self.n_points, np.nan)

        if metric == "weighted_objective":
            return np.array(data["obj_values"], dtype=float)

        summary_stats_runs = data["summary_stats_runs"]
        for i, stats in enumerate(summary_stats_runs):
            if not data["success_flags"][i] or stats is None:
                continue
            val = stats.get(metric)
            if self._is_numeric_finite(val):
                values[i] = float(val)

        return values

    def _calculate_metric_tornado_data(
        self,
        metric: str,
        param_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate low/high percent changes used by a tornado plot.

        The low and high values use the first and last valid perturbation
        factors, respectively. Parameters with fewer than two valid values are
        excluded.
        """
        baseline = self._get_metric_baseline_value(metric)
        if not self._is_numeric_finite(baseline) or baseline == 0:
            raise ValueError(f"Cannot build tornado for '{metric}': invalid or zero baseline.")

        selected_params = self._resolve_plot_parameters(plot_params=param_names)
        result: Dict[str, Dict[str, float]] = {}
        for param_name in selected_params:
            data = self.detailed_sensitivity[param_name]
            series = self._extract_metric_series(param_name, metric)
            valid = data["success_flags"] & np.isfinite(series)
            valid_idx = np.where(valid)[0]
            if len(valid_idx) < 2:
                continue

            min_idx = int(valid_idx[0])
            max_idx = int(valid_idx[-1])
            low_val = float(series[min_idx])
            high_val = float(series[max_idx])

            low_change = (low_val - baseline) / baseline * 100
            high_change = (high_val - baseline) / baseline * 100

            result[param_name] = {
                "low_change": float(low_change),
                "high_change": float(high_change),
                "sensitivity": float(abs(high_change - low_change)),
                "actual_low_factor": float(self.factors[min_idx]),
                "actual_high_factor": float(self.factors[max_idx]),
                "valid_range_fraction": float(len(valid_idx) / len(self.factors)),
            }

        return result

    @staticmethod
    def _metric_label(metric: str, include_units: bool = True) -> str:
        """Return a human-readable label for a summary-statistic key.

        Parameters
        ----------
        metric : str
            Summary-statistic key.
        include_units : bool, default=True
            Whether to retain units in labels with predefined unit-aware names.

        Returns
        -------
        str
            Predefined or title-cased fallback label.
        """
        label_map = KPI_LABELS if include_units else KPI_LABELS_NO_UNITS
        mapped = label_map.get(metric)
        if mapped is not None:
            return mapped
        return metric.replace("_", " ").title()

    def _plot_mode_title_context(self) -> str:
        """Return operating-mode and applicable control-strategy title context."""
        mode = self.baseline_simulation.operating_mode
        mode_label = f"{mode} mode"
        mode_value = str(getattr(mode, "value", mode)).lower()

        if mode_value in {"fed-batch"}:
            feed_strategy = getattr(self.baseline_simulation, "feed_strategy", None)
            if feed_strategy is not None:
                return f"{mode_label} | Feed strategy: {feed_strategy}"

        if mode_value in {"perfusion", "concentrated-fed-batch", "continuous-perfusion"}:
            perfusion_strategy = getattr(self.baseline_simulation, "perfusion_strategy", None)
            if perfusion_strategy is not None:
                return f"{mode_label} | Perfusion strategy: {perfusion_strategy}"

        return mode_label

    def _resolve_plot_parameters(self, plot_params: Optional[List[str]] = None) -> List[str]:
        """Return validated analyzed parameter paths selected for a plot."""
        if plot_params is None:
            return list(self.sensitivity_params)

        if not isinstance(plot_params, list):
            raise ValueError("plot_params must be a list of dot-notation parameter names.")
        if not plot_params:
            raise ValueError("plot_params cannot be empty. Pass None to include all parameters.")

        selected_params: List[str] = []
        for param_name in plot_params:
            if not isinstance(param_name, str):
                raise ValueError("All entries in plot_params must be strings.")
            if param_name not in self.detailed_sensitivity:
                raise KeyError(
                    f"Parameter '{param_name}' has no sensitivity results. "
                    f"Available parameters: {self.sensitivity_params}"
                )
            selected_params.append(param_name)

        return selected_params

    def _resolve_plot_summary_stats(
        self,
        summary_stats: Optional[List[str]] = None,
    ) -> List[str]:
        """Return validated numeric metrics selected for cross-KPI plotting."""
        self._ensure_summary_stat_metadata()

        if summary_stats is None:
            if self.numeric_summary_stats:
                return list(self.numeric_summary_stats)
            if self.all_numeric_summary_stats:
                return list(self.all_numeric_summary_stats)
            raise ValueError("No numeric summary statistics available for plotting.")

        if not isinstance(summary_stats, list):
            raise ValueError("summary_stats must be a list of metric names.")
        if not summary_stats:
            raise ValueError("summary_stats cannot be empty. Pass None to include defaults.")

        selected_metrics: List[str] = []
        for metric in summary_stats:
            if not isinstance(metric, str):
                raise ValueError("All entries in summary_stats must be strings.")
            if metric == "weighted_objective":
                if not self.objective_mode:
                    raise ValueError("weighted_objective requires objective mode.")
            elif metric not in self.all_numeric_summary_stats:
                raise KeyError(
                    f"Metric '{metric}' is not a numeric summary statistic. "
                    f"Available numeric stats: {self.all_numeric_summary_stats}"
                )

            if metric not in selected_metrics:
                selected_metrics.append(metric)

        return selected_metrics


    # Plotting methods
    def create_sensitivity_plots(
        self,
        summary_stat: Optional[str] = None,
        plot_params: Optional[List[str]] = None,
    ) -> Tuple[go.Figure, go.Figure, go.Figure]:
        """Create individual, normalized, and tornado plots for one metric.

        Parameters
        ----------
        summary_stat : str, optional
            Numeric summary-statistic key to plot. In objective mode, omitting
            this selects the only objective or ``"weighted_objective"`` for
            multiple objectives. In validity-only mode, the preferred default
            is ``"STY_kg_m3_day"`` when available.
        plot_params : list of str, optional
            Analyzed dotted parameter paths to include. Defaults to every
            parameter supplied during initialization.

        Returns
        -------
        tuple of plotly.graph_objects.Figure
            Individual parameter subplots, normalized curves, and a tornado
            ranking, in that order.

        Raises
        ------
        ValueError
            If no sweep results exist, the metric cannot be normalized, or too
            few valid points exist for a tornado plot.
        KeyError
            If a requested metric or parameter path is unavailable.
        """
        if not self.detailed_sensitivity:
            raise ValueError("No sensitivity data available. Run run_sensitivity_analysis() first.")

        metric = self._resolve_plot_metric(summary_stat=summary_stat)
        selected_params = self._resolve_plot_parameters(plot_params=plot_params)
        fig1 = self._create_individual_sensitivity_plots(summary_stat=metric, plot_params=selected_params)
        fig2 = self._create_normalized_sensitivity_plot(summary_stat=metric, plot_params=selected_params)
        fig3 = self._create_tornado_plot(summary_stat=metric, plot_params=selected_params)
        return fig1, fig2, fig3

    def create_sensitivity_plots_for_all_summary_stats(
        self,
        plot_params: Optional[List[str]] = None,
        include_optional: bool = False,
        include_stats: Optional[List[str]] = None,
        exclude_stats: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[go.Figure, go.Figure, go.Figure]]:
        """Create a figure triplet for each selected numeric summary statistic.

        Parameters
        ----------
        plot_params : list of str, optional
            Analyzed dotted parameter paths to include. Defaults to all analyzed
            parameters.
        include_optional : bool, default=False
            Whether to add numeric metrics outside the curated default KPI set.
            Ignored when ``include_stats`` is provided.
        include_stats : list of str, optional
            Explicit numeric summary-statistic keys to plot. Overrides the
            default KPI selection and ``include_optional``.
        exclude_stats : list of str, optional
            Metrics to remove after the selection step.

        Returns
        -------
        dict of str to tuple of plotly.graph_objects.Figure
            Mapping from each selected metric to the tuple returned by
            :meth:`create_sensitivity_plots`.

        Raises
        ------
        ValueError
            If no metrics remain after filtering or any selected metric cannot
            produce its complete plot triplet.
        KeyError
            If ``include_stats`` contains a non-numeric or unavailable metric.

        Notes
        -----
        Metric generation is fail-fast: a zero or invalid baseline for any
        selected metric prevents the complete set from being returned.
        """
        self._ensure_summary_stat_metadata()

        if not self.all_numeric_summary_stats:
            raise ValueError("No numeric summary statistics available for plotting.")

        selected_params = self._resolve_plot_parameters(plot_params=plot_params)

        if include_stats is not None:
            unknown = [m for m in include_stats if m not in self.all_numeric_summary_stats]
            if unknown:
                raise KeyError(
                    f"include_stats contains non-numeric or unknown metrics: {unknown}. "
                    f"Available numeric stats: {self.all_numeric_summary_stats}"
                )
            metrics: List[str] = []
            for metric in include_stats:
                if metric not in metrics:
                    metrics.append(metric)
        else:
            metrics = list(self.numeric_summary_stats)
            if include_optional:
                for metric in self.all_numeric_summary_stats:
                    if metric not in metrics:
                        metrics.append(metric)

        if exclude_stats is not None:
            metrics = [m for m in metrics if m not in set(exclude_stats)]

        if not metrics:
            raise ValueError("No metrics selected for plotting after include/exclude filters.")

        figures: Dict[str, Tuple[go.Figure, go.Figure, go.Figure]] = {}
        for summary_stat in metrics:
            figures[summary_stat] = self.create_sensitivity_plots(
                summary_stat=summary_stat,
                plot_params=selected_params,
            )
        return figures

    def _create_individual_sensitivity_plots(
        self,
        summary_stat: str,
        plot_params: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create per-parameter subplots of metric value against parameter value.

        Failed evaluations are shown as red crosses at the baseline metric value.
        The horizontal and vertical dashed lines identify the baseline metric and
        parameter value, respectively.
        """
        param_names = self._resolve_plot_parameters(plot_params=plot_params)
        baseline_metric_value = self._get_metric_baseline_value(summary_stat)
        metric_label = self._metric_label(summary_stat)
        mode_context = self._plot_mode_title_context()

        n_params = len(param_names)
        n_cols = 2
        if n_params >= 24:
            n_cols = 2
        if n_params >= 40:
            n_cols = 1
        if n_params == 1:
            n_cols = 1
        n_rows = int(np.ceil(n_params / n_cols))

        subplot_titles = [
            self.baseline_parameters.get_parameter(name).full_label
            for name in param_names
        ]

        subplot_kwargs: Dict[str, Any] = {
            "rows": n_rows,
            "cols": n_cols,
            #"subplot_titles": subplot_titles,
            "horizontal_spacing": 0.08,
        }
        if n_rows > 1:
            max_vertical_spacing = 1.0 / (n_rows - 1)
            # Keep row gaps compact, especially for 2-row layouts used often in notebooks.
            if n_rows <= 2:
                target_vertical_spacing = 0.05
            elif n_rows <= 4:
                target_vertical_spacing = 0.05
            else:
                target_vertical_spacing = 0.008
            subplot_kwargs["vertical_spacing"] = min(target_vertical_spacing, 0.85 * max_vertical_spacing)

        try:
            fig = make_subplots(**subplot_kwargs)
        except ValueError as exc:
            # Keep plotting robust for high-row grids where Plotly enforces strict spacing bounds.
            if "Vertical spacing cannot be greater" not in str(exc) or n_rows <= 1:
                raise
            max_vertical_spacing = 1.0 / (n_rows - 1)
            subplot_kwargs["vertical_spacing"] = 0.1 * max_vertical_spacing
            fig = make_subplots(**subplot_kwargs)

        for i, param_name in enumerate(param_names):
            row = i // n_cols + 1
            col = i % n_cols + 1

            param_obj = self.baseline_parameters.get_parameter(param_name)
            data = self.detailed_sensitivity[param_name]
            x_values = data["param_values"]
            y_values = self._extract_metric_series(param_name, summary_stat)
            valid_mask = data["success_flags"] & np.isfinite(y_values)
            failed_mask = ~valid_mask

            if np.any(valid_mask):
                fig.add_trace(
                    go.Scatter(
                        x=x_values[valid_mask],
                        y=y_values[valid_mask],
                        mode="lines+markers",
                        showlegend=False,
                        hovertemplate="Param: %{x:.4g}<br>Metric: %{y:.4g}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

            if np.any(failed_mask):
                fig.add_trace(
                    go.Scatter(
                        x=x_values[failed_mask],
                        y=np.full(np.sum(failed_mask), baseline_metric_value),
                        mode="markers",
                        marker=dict(color="crimson", symbol="x", size=8),
                        showlegend=False,
                        hovertemplate="Param: %{x:.4g}<br>Failed run<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

            x_min = float(np.nanmin(x_values))
            x_max = float(np.nanmax(x_values))

            fig.add_shape(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=baseline_metric_value,
                y1=baseline_metric_value,
                line=dict(color="darkgreen", dash="dash"),
                row=row,
                col=col,
            )
            fig.add_vline(
                x=float(data["baseline_value"]),
                line_dash="dash",
                line_color="firebrick",
                row=row,
                col=col,
            )

            fig.update_xaxes(title_text=param_obj.full_label, row=row, col=col)
            fig.update_yaxes(title_text=metric_label, row=row, col=col)

        fig.update_layout(
            title=f"Parameter Sensitivity ({metric_label}) - {mode_context}",
            title_x=0.5,
            template="plotly_white",
            hovermode="closest",
            # Keep a balanced aspect ratio with compact whitespace in notebook rendering.
            height=max(560, 290 * n_rows + 100),
            width=max(820, 440 * n_cols),
            margin=dict(l=70, r=30, t=70, b=50),
            showlegend=False,
        )
        return fig

    def _create_normalized_sensitivity_plot(
        self,
        summary_stat: str,
        plot_params: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create normalized metric curves against relative parameter factors.

        Each metric value is divided by its baseline value. The horizontal and
        vertical reference lines therefore intersect at the baseline point
        ``(1.0, 1.0)``.
        """
        baseline_metric_value = self._get_metric_baseline_value(summary_stat)
        if not self._is_numeric_finite(baseline_metric_value) or baseline_metric_value == 0:
            raise ValueError(
                f"Cannot normalize summary stat '{summary_stat}': baseline value is invalid or zero."
            )

        metric_label = self._metric_label(summary_stat)
        mode_context = self._plot_mode_title_context()
        selected_params = self._resolve_plot_parameters(plot_params=plot_params)
        fig = go.Figure()
        metric_label_no_units = self._metric_label(summary_stat, include_units=False)

        for param_name in selected_params:
            param_obj = self.baseline_parameters.get_parameter(param_name)
            metric_series = self._extract_metric_series(param_name, summary_stat)
            normalized = metric_series / baseline_metric_value
            valid = np.isfinite(normalized)
            if not np.any(valid):
                continue

            fig.add_trace(
                go.Scatter(
                    x=self.factors[valid],
                    y=normalized[valid],
                    mode="lines+markers",
                    name=param_obj.name,
                    hovertemplate="Factor: %{x:.3f}x<br>Normalized: %{y:.4f}<extra></extra>",
                )
            )

        fig.add_hline(y=1.0, line_dash="dash", line_color="black")
        fig.add_vline(x=1.0, line_dash="dash", line_color="black")

        fig.update_layout(
            title=f"Normalized Sensitivity ({metric_label}) - {mode_context}",
            title_x=0.5,
            xaxis_title="Parameter factor",
            yaxis_title=f"Normalized {metric_label_no_units}",
            template="plotly_white",
            hovermode="x unified",
            height=650,
            width=1200,
            legend=dict(x=1.02, y=1.0),
        )
        return fig

    def _create_tornado_plot(
        self,
        summary_stat: str,
        plot_params: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create a tornado ranking from low and high metric changes.

        Parameters are ordered by the absolute difference between their low and
        high percent changes from baseline. Only parameters with at least two
        valid metric values are included.
        """
        metric_label = self._metric_label(summary_stat)
        mode_context = self._plot_mode_title_context()
        selected_params = self._resolve_plot_parameters(plot_params=plot_params)
        tornado = self._calculate_metric_tornado_data(summary_stat, param_names=selected_params)
        if not tornado:
            raise ValueError(
                f"Not enough valid sensitivity points to build tornado plot for '{summary_stat}'."
            )

        sorted_items = sorted(
            tornado.items(),
            key=lambda item: item[1]["sensitivity"],
            reverse=True,
        )

        labels = [self.baseline_parameters.get_parameter(name).name for name, _ in sorted_items]
        low_values = [vals["low_change"] for _, vals in sorted_items]
        high_values = [vals["high_change"] for _, vals in sorted_items]
        low_text = [f"{vals['actual_low_factor']:.2f}x" for _, vals in sorted_items]
        high_text = [f"{vals['actual_high_factor']:.2f}x" for _, vals in sorted_items]

        perturbation_pct = None
        if self.perturbation is not None and self._is_numeric_finite(self.perturbation):
            perturbation_pct = abs(float(self.perturbation) * 100.0)

        if perturbation_pct is not None:
            low_label = f"{-perturbation_pct:g}%"
            high_label = f"+{perturbation_pct:g}%"
        else:
            low_label = "Low"
            high_label = "High"

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=labels,
                x=low_values,
                orientation="h",
                name=low_label,
                marker_color="lightcoral",
                text=low_text,
                textposition="inside",
                hovertemplate=f"%{{y}}<br>{low_label} change: %{{x:.2f}}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                y=labels,
                x=high_values,
                orientation="h",
                name=high_label,
                marker_color="lightblue",
                text=high_text,
                textposition="inside",
                hovertemplate=f"%{{y}}<br>{high_label} change: %{{x:.2f}}%<extra></extra>",
            )
        )

        fig.add_vline(x=0.0, line_width=1, line_color="black")
        fig.update_layout(
            title=f"Tornado Sensitivity ({metric_label}) - {mode_context}",
            title_x=0.5,
            xaxis_title="Metric change (%)",
            yaxis_title="Parameter",
            template="plotly_white",
            barmode="overlay",
            height=max(500, 45 * len(labels) + 180),
            width=1200,
            legend=dict(x=1.01, y=1.0),
        )
        return fig

    def create_sensitivity_heatmap(
        self,
        summary_stats: Optional[List[str]] = None,
        plot_params: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create paired KPI-by-parameter heatmaps for low and high perturbations.

        Each cell reports the percentage change from that metric's baseline.
        Low and high values use the first and last valid parameter factors, as
        in :meth:`_calculate_metric_tornado_data`.

        Parameters
        ----------
        summary_stats : list of str, optional
            Numeric summary-statistic keys to include. Defaults to the curated
            KPI subset, or all numeric metrics when no curated key is available.
        plot_params : list of str, optional
            Analyzed dotted parameter paths to include. Defaults to all analyzed
            parameters.

        Returns
        -------
        plotly.graph_objects.Figure
            Two-panel heatmap containing low- and high-perturbation changes.

        Raises
        ------
        ValueError
            If no sweep results or valid percentage changes are available.
        KeyError
            If a requested metric or parameter path is unavailable.
        """
        if not self.detailed_sensitivity:
            raise ValueError("No sensitivity data available. Run run_sensitivity_analysis() first.")

        selected_metrics = self._resolve_plot_summary_stats(summary_stats=summary_stats)
        selected_params = self._resolve_plot_parameters(plot_params=plot_params)
        mode_context = self._plot_mode_title_context()

        low_matrix = np.full((len(selected_params), len(selected_metrics)), np.nan)
        high_matrix = np.full((len(selected_params), len(selected_metrics)), np.nan)

        for j, metric in enumerate(selected_metrics):
            tornado_data = self._calculate_metric_tornado_data(metric, param_names=selected_params)
            for i, param_name in enumerate(selected_params):
                metric_values = tornado_data.get(param_name)
                if metric_values is None:
                    continue
                low_matrix[i, j] = float(metric_values["low_change"])
                high_matrix[i, j] = float(metric_values["high_change"])

        combined = np.concatenate([low_matrix[np.isfinite(low_matrix)], high_matrix[np.isfinite(high_matrix)]])
        if combined.size == 0:
            raise ValueError("No valid sensitivity values found to build heatmap.")

        abs_limit = float(np.nanmax(np.abs(combined)))
        if abs_limit == 0:
            abs_limit = 1.0

        # Heatmap values are relative changes (%), so labels are shown without units.
        y_labels = [self.baseline_parameters.get_parameter(name).name for name in selected_params]
        metric_labels = [self._metric_label(metric, include_units=False) for metric in selected_metrics]

        perturbation_pct = None
        if self.perturbation is not None and self._is_numeric_finite(self.perturbation):
            perturbation_pct = abs(float(self.perturbation) * 100.0)

        if perturbation_pct is not None:
            low_subplot_title = f"Perturbation -{perturbation_pct:g}% (change %)"
            high_subplot_title = f"Perturbation +{perturbation_pct:g}% (change %)"
        else:
            low_subplot_title = "Low perturbation change (%)"
            high_subplot_title = "High perturbation change (%)"

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(low_subplot_title, high_subplot_title),
            horizontal_spacing=0.1,
        )

        heatmap_common = {
            "x": metric_labels,
            "y": y_labels,
            "colorscale": "viridis",
            "zmin": -abs_limit,
            "zmax": abs_limit,
            "zmid": 0,
            "colorbar": {"title": "Change (%)"},
            "hovertemplate": "Parameter: %{y}<br>KPI: %{x}<br>Change: %{z:.2f}%<extra></extra>",
            "texttemplate": "%{z:.1f}%",
            "textfont": {"size": 12},
        }

        fig.add_trace(
            go.Heatmap(
                z=low_matrix,
                showscale=False,
                **heatmap_common,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=high_matrix,
                showscale=True,
                **heatmap_common,
            ),
            row=1,
            col=2,
        )

        # fig.update_xaxes(title_text="KPI", tickangle=30, row=1, col=1)
        # fig.update_xaxes(title_text="KPI", tickangle=30, row=1, col=2)
        fig.update_yaxes(title_text="", autorange="reversed", row=1, col=1)
        fig.update_yaxes(
            title_text="",
            autorange="reversed",
            showticklabels=False,
            row=1,
            col=2,
        )

        fig.update_layout(
            title=(
                "Cross-KPI Sensitivity Heatmap "
                f"- {mode_context}"
            ),
            title_x=0.5,
            template="plotly_white",
            height=max(500, 34 * len(selected_params) + 220),
            width=max(1200, 110 * len(selected_metrics) + 560),
        )

        return fig