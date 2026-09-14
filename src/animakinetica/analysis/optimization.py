"""Optimize cell-culture simulation parameters within YAML-defined bounds.

``OptimizationManager`` evaluates candidate parameter sets with
``CellCultureSimulation.run_temporary_simulation`` and maximizes one or more
``SimulationData.summary_stats`` metrics through
``scipy.optimize.differential_evolution``. Because SciPy minimizes, the
objective callable returns the negative weighted score.

Candidates that fail simulation, violate hard validity criteria, or contain
invalid objective values receive a penalty of ``1e6``. Finite, measurable
violations of selected operational constraints receive graded penalties.
Objectives can be scored as raw values, normalized against the baseline
simulation, normalized with supplied campaign-wide values, or transformed
through configured engineering desirability functions.
"""

# Imports
import multiprocessing
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from scipy.optimize import differential_evolution
from animakinetica.core.simulation import CellCultureSimulation, CONTINUOUS_MODES
from animakinetica.core.parameters import ParameterIO
from animakinetica.core.simulation_data import SimulationData, SUMMARY_METRIC_DIRECTIONS
from animakinetica.models.odesystems import OperatingMode
from time import time
import gc
import pandas as pd
import numpy as np

__version__ = "1.0.0"

__all__ = [
    "OptimizationManager",
    "ObjectiveFunction",
    "list_opt_configs",
]


# Add global constants for configuration paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
OPT_CONFIG_DIR = CONFIG_DIR / "optimization_config"
DEFAULT_OPT_CONFIG = OPT_CONFIG_DIR / "default_optimization.yaml"

def list_opt_configs() -> List[str]:
    """List available optimization configuration names.

    Returns
    -------
    list of str
        Stems of ``*.yaml`` files in ``OPT_CONFIG_DIR``. Returns an empty list
        when the directory is missing or contains no YAML files.
    """
    if not OPT_CONFIG_DIR.exists():
        print(f"No parameter sets directory found at {OPT_CONFIG_DIR}")
        return []
    
    yaml_files = list(OPT_CONFIG_DIR.glob("*.yaml"))
    if not yaml_files:
        print("No parameter set files found.\n")
        return []
    
    return [f.stem for f in yaml_files]

def timing_decorator(func):
    """Optionally print the elapsed execution time of a callable.

    Set ``ANIMAKINETICA_TIMING_OUTPUT=0`` to suppress timing diagnostics for
    a process, such as when a campaign runner executes many child scripts.
    Direct script execution keeps timing output enabled by default.

    Parameters
    ----------
    func : callable
        Callable to wrap.

    Returns
    -------
    callable
        Wrapper that returns the result of ``func`` and prints elapsed time in
        milliseconds.
    """
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        if os.environ.get("ANIMAKINETICA_TIMING_OUTPUT", "1") != "0":
            print(f"{func.__name__} execution time: {(end_time - start_time) * 1000:.3f} ms")
        return result
    return wrapper


class OptimizationManager:
    """Manage bounded differential-evolution optimization for a simulation.

    Parameters
    ----------
    simulation : CellCultureSimulation
        Baseline simulation supplying fixed model choices, parameters, and
        summary statistics. Run it before using baseline normalization.
    opt_config_file : str or pathlib.Path, optional
        YAML configuration name or path. A name without an extension is
        resolved under ``OPT_CONFIG_DIR``. If omitted,
        ``DEFAULT_OPT_CONFIG`` is used.

    Attributes
    ----------
    baseline_simulation : CellCultureSimulation
        Reference simulation used to construct candidates.
    opt_config : dict or None
        Parsed parameter paths, bounds, display metadata, and vector indices.
    optimization_result : scipy.optimize.OptimizeResult or None
        Result returned by ``scipy.optimize.differential_evolution``.
    optimal_parameters : dict or None
        Mapping from dotted parameter paths to optimized values.
    optimized_simulation : CellCultureSimulation or None
        Full simulation rerun with the selected optimized parameters.
    """
    
    def __init__(self, simulation: CellCultureSimulation, opt_config_file: Optional[Union[str, Path]] = None):
        """Initialize the manager and load its optimization configuration.

        Raises
        ------
        ValueError
            If ``simulation`` has no parameter object.
        FileNotFoundError
            If the selected configuration cannot be found or loaded.
        """
        self.baseline_simulation = simulation
        self.opt_config = None
        self.optimal_parameters = None
        self.optimization_result = None
        self.optimized_simulation = None
        self.objectives = None
        self.optimization_scoring = None

        # Ensure simulation has parameters
        if not self.baseline_simulation.parameters:
            raise ValueError("Simulation must have parameters defined for optimization.")
        
        # Store reference to base parameters (immutable)
        self.base_parameters = self.baseline_simulation.parameters
        
        # Load optimization configuration
        self.opt_config_file = opt_config_file
        self.load_optimization_config(self.opt_config_file)
  
    ### Optimization Configuration Methods ###
    ## Public Methods ##
    def load_optimization_config(self, opt_config: Optional[Union[str, Path]] = None):
        """Load and parse an optimization configuration YAML file.

        Parameters
        ----------
        opt_config : str or pathlib.Path, optional
            Configuration name or path. Names without a suffix are resolved as
            ``OPT_CONFIG_DIR / "<name>.yaml"``. If omitted, uses
            ``DEFAULT_OPT_CONFIG``.

        Raises
        ------
        FileNotFoundError
            If the requested file or default configuration does not exist, or
            if YAML loading returns ``None``.
        """
        # Resolve file path
        if opt_config is None:
            # Use default config
            print(f"No optimization config specified. Using default...")
            if not DEFAULT_OPT_CONFIG.exists():
                raise FileNotFoundError(f"Default optimization config not found: {DEFAULT_OPT_CONFIG}")
            filepath = DEFAULT_OPT_CONFIG
            self.opt_config_file = filepath
        else:
            print(f"Searching specified optimization config: {opt_config}")
            filepath = Path(opt_config)
            
            # If only name given (no extension), look in optimization_config directory
            if not filepath.suffix:
                filepath = OPT_CONFIG_DIR / f"{opt_config}.yaml"
            
            # Check if file exists
            if not filepath.exists():
                raise FileNotFoundError(f"Optimization config not found: {filepath}")
        
        print(f"Loading optimization config from: {filepath}")
        
        # Load the config
        config_data = ParameterIO.load_yaml(filepath)
        self.raw_config = config_data
        if self.raw_config is None:
            raise FileNotFoundError(f"Failed to load optimization config: {filepath}")
        
        print(f"Parsing...")
        self.opt_config = self._create_optimization_config()
        print(f"Parsed and loaded configuration containing {len(self.opt_config)} optimization parameters.")

    ## Private Methods ##
    def _create_optimization_config(self):
        """Create the internal optimization-parameter mapping from raw YAML.

        Returns
        -------
        dict
            Mapping from dotted parameter paths to bounds, vector indices, and
            optional display metadata.
        """
        config = {}
        
        # Extract section names directly from the configuration file
        sections_to_process = []
        for key in self.raw_config.keys():
            if key != 'metadata':  # Skip metadata section
                sections_to_process.append(key)
        
        index = 0
        for section_name in sections_to_process:
            if section_name in self.raw_config:
                for param_name, param_config in self.raw_config[section_name].items():
                    opt_key = f"{section_name}.{param_name}"
                    
                    config[opt_key] = {
                        'bounds': self._process_bounds(param_config, section_name, param_name),
                        'index': index,
                        'label': param_config.get('label', param_name),
                        'unit': param_config.get('unit', ''),
                        'full_label': param_config.get('full_label', f"{param_config.get('label', param_name)} ({param_config.get('unit', '')})"),
                        'section': section_name,
                        'parameter': param_name
                    }
                    index += 1

        return config
    
    def _process_bounds(self, param_config: Dict[str, Any], section_name: str, param_name: str) -> tuple:
        """Resolve direct or baseline-relative bounds for one parameter.

        Parameters
        ----------
        param_config : dict
            YAML entry containing ``bounds`` or ``bounds_type: relative`` and
            ``bounds_factor``.
        section_name : str
            Top-level parameter section in the YAML file.
        param_name : str
            Parameter name within ``section_name``.

        Returns
        -------
        tuple
            Lower and upper optimization bounds.

        Raises
        ------
        ValueError
            If neither a direct nor valid relative-bounds definition is given.
        """
        
        # Direct bounds
        if 'bounds' in param_config:
            return tuple(param_config['bounds'])
        
        # Relative bounds
        if param_config.get('bounds_type') == 'relative':
            bounds_factor = param_config['bounds_factor']
            
            # Get reference value
            ref_path = param_config.get('bounds_reference', f"{section_name}.{param_name}")
            ref_value = self.base_parameters.get_parameter_value(ref_path)
            
            return (ref_value * bounds_factor[0], ref_value * bounds_factor[1])
        
        raise ValueError(f"Invalid bounds configuration for {section_name}.{param_name}")

    def _resolve_effective_workers(self, workers: int, verbose: bool) -> int:
        """Resolve a safe worker count for ``differential_evolution``.

        Parameters
        ----------
        workers : int
            Requested SciPy worker count. ``-1`` requests all available CPUs;
            positive values request that many workers.
        verbose : bool
            Whether to print the requested and effective worker counts.

        Returns
        -------
        int
            Requested worker count unless a daemonized or non-main Windows
            process requires serial execution.

        Raises
        ------
        ValueError
            If ``workers`` cannot be converted to an integer or is zero or
            less than ``-1``.
        """
        try:
            requested_workers = int(workers)
        except (TypeError, ValueError):
            raise ValueError(f"workers must be an integer, got: {workers}")

        if requested_workers == 0 or requested_workers < -1:
            raise ValueError(
                f"workers must be 1, -1, or any positive integer. Received: {requested_workers}"
            )

        process = multiprocessing.current_process()
        unsafe_context_reasons: List[str] = []

        # Windows uses spawn and is prone to recursive import/worker startup issues.
        if os.name == "nt" and process.name != "MainProcess":
            unsafe_context_reasons.append(f"process name is {process.name}")

        if process.daemon:
            unsafe_context_reasons.append("process is daemonized")

        effective_workers = requested_workers
        if requested_workers != 1 and unsafe_context_reasons:
            effective_workers = 1
            if verbose:
                reasons = "; ".join(unsafe_context_reasons)
                print(
                    "Warning: forcing workers=1 due to unsafe multiprocessing context "
                    f"({reasons})."
                )

        if verbose:
            print(
                f"Worker configuration: requested={requested_workers}, effective={effective_workers}"
            )

        return effective_workers


    ### Optimization Handling ###
    ## Public Methods ##
    @timing_decorator
    def optimize_parameters(self, objectives: Union[str, List[str]] = ["STY_kg_m3_day"], 
                          weights: Optional[Union[float, List[float]]] = [1], workers: int = 1, seed: Optional[int] = None,
                          verbose: bool = True, timeout: float = 5.0,
                          normalize_objectives: bool = True,
                          normalization_values: Optional[Dict[str, float]] = None,
                          desirability_config: Optional[Dict[str, Dict[str, float]]] = None,
                          scoring_profile: Optional[str] = None,
                          debug_rejections: bool = False,
                          evaluate_baseline: bool = True,
                          seed_baseline: bool = False,
                          ):
        """Optimize configured parameters with differential evolution.

        Candidate simulations are evaluated by :class:`ObjectiveFunction`.
        Valid candidates maximize the weighted objective score; SciPy receives
        its negative. The final selected values are rerun through
        :meth:`_create_optimized_simulation` and stored on this manager.

        Parameters
        ----------
        objectives : str or list of str, default ["STY_kg_m3_day"]
            Keys in ``SimulationData.summary_stats``. Objectives registered as
            conversion/use ratios are minimized; all other objectives default
            to maximization.
        weights : float or list of float or None, default [1]
            Objective weights. ``None`` assigns equal weights. A scalar is
            accepted for one objective; otherwise the count must match
            ``objectives``.
        workers : int, default 1
            Worker count passed to ``scipy.optimize.differential_evolution``.
            ``-1`` requests all CPUs. Unsafe nested multiprocessing contexts
            fall back to serial evaluation.
        seed : int, optional
            Random seed passed to SciPy.
        verbose : bool, default True
            Print configuration, normalization, and optimizer progress.
        timeout : float, default 5.0
            Maximum wall-clock time in seconds for one temporary simulation.
            Timed-out candidates receive the hard rejection penalty.
        normalize_objectives : bool, default True
            Divide each objective by its baseline summary-statistic value.
            Ignored when ``normalization_values`` is provided.
        normalization_values : dict of str to float, optional
            Explicit divisors for each objective, typically shared across a
            campaign. Values must be finite and non-zero.
        desirability_config : dict of str to dict, optional
            Engineering desirability settings keyed by objective. Each setting
            must contain finite ``minimum`` and ``target`` values. Maximize
            KPIs require ``target > minimum``; minimize KPIs require
            ``target < minimum``, where ``minimum`` is the upper unacceptable
            value. An optional per-objective ``safety_cap`` must be finite and
            greater than 1; omitted values default to
            :attr:`ObjectiveFunction.DESIRABILITY_SAFETY_CAP`. This mode is
            mutually exclusive with ``normalization_values`` and maps each
            objective to a smooth, asymptotically capped desirability.
        scoring_profile : str, optional
            Human-readable identifier persisted with exported optimization
            metadata. It does not affect objective calculation.
        debug_rejections : bool, default False
            Print diagnostics for hard and soft candidate rejections.
        evaluate_baseline : bool, default True
            Evaluate the unmodified parameter vector through
            :class:`ObjectiveFunction` before optimization. Stores
            ``baseline_objective_value`` and ``baseline_is_feasible``.
        seed_baseline : bool, default False
            Supply the baseline vector as SciPy's ``x0``. It must lie within
            all configured bounds.

        Returns
        -------
        None
            Stores ``optimization_result``, ``optimal_parameters``, and
            ``optimized_simulation``.

        Raises
        ------
        ValueError
            If the configuration is unavailable, objectives and weights are
            inconsistent, normalization divisors are invalid, or baseline
            seeding is out of bounds.
        KeyError
            If an objective is absent from baseline statistics or explicit
            normalization values.
        RuntimeError
            If SciPy returns a non-finite objective value.
        """
        # Ensure optimization configuration is loaded
        if self.opt_config is None:
            raise ValueError("Optimization configuration not loaded. Please load or create it before optimization.")

        # Normalize objectives and weights to lists
        if isinstance(objectives, str):
            objectives = [objectives]
        
        if weights is None:
            weights = [1.0] * len(objectives)  # Equal weights by default
        elif isinstance(weights, (int, float)):
            weights = [weights]
        
        # Validate inputs
        if len(objectives) != len(weights):
            raise ValueError(f"Number of objectives ({len(objectives)}) must match number of weights ({len(weights)})")
        
        if not objectives:
            raise ValueError("At least one objective must be specified")

        if verbose:
            obj_str = ', '.join(
                f"{obj} ({SUMMARY_METRIC_DIRECTIONS.get(obj, 'maximize')}, weight: {weight})"
                for obj, weight in zip(objectives, weights)
            )
            print(f"Starting optimization with objectives: {obj_str}")

        effective_workers = self._resolve_effective_workers(workers=workers, verbose=verbose)
        
        bounds = [config['bounds'] for config in self.opt_config.values()]
        param_names = list(self.opt_config.keys())
        
        if verbose:
            print(f"Optimizing {len(param_names)} parameters: {param_names}")

        if desirability_config is not None and normalization_values is not None:
            raise ValueError(
                "desirability_config and normalization_values are mutually exclusive. "
                "Choose either engineering desirability scoring or legacy campaign normalization."
            )

        validated_desirability_config = None
        if desirability_config is not None:
            validated_desirability_config = ObjectiveFunction.validate_desirability_config(
                desirability_config,
                objectives,
            )

        # Compute legacy normalization scalars. Engineering desirability is
        # already dimensionless and intentionally bypasses this path.
        # Priority without desirability: explicit campaign values >
        # per-simulation baseline (normalize_objectives=True) > raw values.
        baseline_values: Optional[Dict[str, float]] = None
        if normalization_values is not None:
            # Validate that all objectives are covered
            missing = [obj for obj in objectives if obj not in normalization_values]
            if missing:
                raise KeyError(
                    f"normalization_values is missing entries for: {missing}. "
                    f"Available keys: {list(normalization_values.keys())}"
                )
            non_finite = [obj for obj in objectives if not np.isfinite(normalization_values[obj])]
            if non_finite:
                raise ValueError(
                    f"normalization_values contains non-finite entries (nan/inf) for: {non_finite}."
                )
            near_zero = [obj for obj in objectives if abs(normalization_values[obj]) < 1e-12]
            if near_zero:
                raise ValueError(
                    f"normalization_values contains zero/near-zero entries for: {near_zero}. "
                    f"Cannot normalize."
                )
            baseline_values = {obj: normalization_values[obj] for obj in objectives}
            if verbose:
                print("Using campaign-level normalization scalars:")
                for obj, val in baseline_values.items():
                    print(f"  {obj}: {val:.4g}")
        elif normalize_objectives and validated_desirability_config is None:
            baseline_summary = self.baseline_simulation.data.summary_stats
            baseline_values = {}
            for obj in objectives:
                val = baseline_summary.get(obj)
                if val is None:
                    raise KeyError(
                        f"Objective '{obj}' not found in baseline simulation summary stats. "
                        f"Available keys: {list(baseline_summary.keys())}"
                    )
                if not np.isfinite(val):
                    raise ValueError(
                        f"Baseline value for objective '{obj}' is non-finite ({val}). "
                        f"Cannot normalize. Check baseline simulation results."
                    )
                if abs(val) < 1e-12:
                    raise ValueError(
                        f"Baseline value for objective '{obj}' is zero or near-zero ({val:.4g}). "
                        f"Cannot normalize. Check baseline simulation results."
                    )
                baseline_values[obj] = val
            if verbose:
                print("Using per-simulation baseline normalization scalars:")
                for obj, val in baseline_values.items():
                    print(f"  {obj}: {val:.4g}")

        if verbose and validated_desirability_config is not None:
            print("Using engineering desirability scoring (per-KPI smooth asymptotic caps):")
            for obj in objectives:
                config = validated_desirability_config[obj]
                direction = SUMMARY_METRIC_DIRECTIONS.get(obj, 'maximize')
                print(
                    f"  {obj} ({direction}): minimum={config['minimum']:.4g}, "
                    f"target={config['target']:.4g}, safety_cap={config['safety_cap']:.4g}"
                )

        # Create objective function instance (pickleable for parallel execution)
        fun_objective = ObjectiveFunction(
            baseline_simulation=self.baseline_simulation,
            opt_config=self.opt_config,
            objectives=objectives,
            weights=weights,
            timeout=timeout,
            baseline_values=baseline_values,
            desirability_config=validated_desirability_config,
            debug_rejections=debug_rejections,
        )

        # Evaluate the exact unmodified parameter vector through the same
        # temporary-simulation and rejection path as DE candidates. This is
        # especially useful after changing validity filters: a saved simulation
        # may look valid in a full run while the lightweight optimization path
        # exposes a timeout, accounting, or filter discrepancy.
        baseline_vector = np.asarray([
            self.base_parameters.get_parameter_value(param_name)
            for param_name in param_names
        ], dtype=float)
        baseline_in_bounds = all(
            lower <= value <= upper
            for value, (lower, upper) in zip(baseline_vector, bounds)
        )
        self.baseline_objective_value = None
        self.baseline_is_feasible = None
        if evaluate_baseline:
            self.baseline_objective_value = float(fun_objective(baseline_vector))
            self.baseline_is_feasible = self.baseline_objective_value != 1e6
            if verbose:
                status = "feasible" if self.baseline_is_feasible else "rejected"
                print(
                    f"Baseline objective evaluation: {status} "
                    f"(value={self.baseline_objective_value:.6g}; "
                    f"within_bounds={baseline_in_bounds})"
                )

        if seed_baseline and not baseline_in_bounds:
            raise ValueError(
                "Cannot seed differential evolution with the baseline parameter vector "
                "because one or more values lie outside the configured bounds."
            )
            
        if verbose:
            print("Running differential evolution...")
        
        if self.baseline_simulation.performance_mode == True:
            maxiter = 10
            popsize = np.minimum(3 * len(self.opt_config), 15)  # Rule of thumb is to scale the population size with number of parameters
            atol = 0
            tol = 0.1
        else:
            maxiter = 35
            popsize = np.minimum(5 * len(self.opt_config), 50)  # Larger population for non-performance mode
            atol = 0
            tol = 0.01

        result = differential_evolution(
            fun_objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            disp=verbose,
            polish=True,
            mutation=(0.5, 1.5),
            # atol=atol,
            tol=tol,
            workers=effective_workers,
            updating='deferred' if effective_workers != 1 else 'immediate',  # Automatically changes to deferred when using muliprocessing
            x0=baseline_vector if seed_baseline else None,
        )

        if not np.isfinite(result.fun):
            raise RuntimeError(
                "Optimization ended with a non-finite objective value (nan/inf). "
                "This usually indicates invalid candidate metrics or normalization scalars. "
                "Re-run with debug_rejections=True to diagnose rejected candidates."
            )
        
        # Store and process results
        self.objectives = objectives  # Store for later reference (e.g. opt YAML export)
        if validated_desirability_config is not None:
            scoring_mode = 'engineering_desirability'
        elif normalization_values is not None:
            scoring_mode = 'campaign_normalized'
        elif normalize_objectives:
            scoring_mode = 'baseline_normalized'
        else:
            scoring_mode = 'raw'
        self.optimization_scoring = {
            'profile': scoring_profile,
            'mode': scoring_mode,
            'objectives': list(objectives),
            'objective_directions': {
                objective: SUMMARY_METRIC_DIRECTIONS.get(objective, 'maximize')
                for objective in objectives
            },
            'weights': [float(weight) for weight in weights],
        }
        if baseline_values is not None:
            self.optimization_scoring['normalization_values'] = {
                objective: float(value) for objective, value in baseline_values.items()
            }
        if validated_desirability_config is not None:
            self.optimization_scoring['desirability'] = {
                objective: {
                    'minimum': float(config['minimum']),
                    'target': float(config['target']),
                    'safety_cap': float(config['safety_cap']),
                }
                for objective, config in validated_desirability_config.items()
            }
        self._store_optimization_results(result)
        
        if verbose:
            print(f"\nOptimization completed!")
            for obj in objectives:
                if obj in self.optimized_simulation.data.summary_stats:
                    value = self.optimized_simulation.data.summary_stats[obj]
                    print(f"  {obj}: {value:.4f}")
        
    def compare_performance(self, verbose: bool = True) -> pd.DataFrame:
        """Compare baseline and optimized summary statistics.

        Parameters
        ----------
        verbose : bool, default True
            Print the comparison table.

        Returns
        -------
        pandas.DataFrame
            Outer-joined metric table with ``Baseline``, ``Optimized``, and
            direction-aware numeric ``Improvement`` columns. Positive
            improvement means better performance for registered minimize as
            well as maximize KPIs. Non-numeric metrics have ``NaN`` values.

        Raises
        ------
        ValueError
            If optimization has not produced ``optimal_parameters``.
        """
        if self.optimal_parameters is None:
            raise ValueError("No optimal parameters available. Run optimize_parameters() first.")

        # Get summary stats DataFrames
        baseline_df = self.baseline_simulation.data.get_summary_stats_df(verbose=False)
        optimized_df = self.optimized_simulation.data.get_summary_stats_df(verbose=False)
        
        # Align on metric metadata and only subtract numeric values.
        # Some summary statistics, such as projection_mode, are descriptive strings.
        comparison_df = baseline_df.rename(columns={'Value': 'Baseline'}).merge(
            optimized_df.rename(columns={'Value': 'Optimized'}),
            on=['Metric', 'Description', 'Unit'],
            how='outer',
        )

        baseline_numeric = pd.to_numeric(comparison_df['Baseline'], errors='coerce')
        optimized_numeric = pd.to_numeric(comparison_df['Optimized'], errors='coerce')
        raw_change = optimized_numeric - baseline_numeric
        directions = comparison_df['Metric'].map(SUMMARY_METRIC_DIRECTIONS).fillna('maximize')
        comparison_df['Improvement'] = raw_change.where(directions == 'maximize', -raw_change)
        
        if verbose:
            print("\n=== PERFORMANCE COMPARISON ===")
            print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    ## Public Methods ##
    def save_opt_parameters(self, filepath: Path, overwrite: bool = False) -> None:
        """Save optimized parameter values as a sectioned YAML patch file.

        Each saved entry contains its display name, floating-point value, and
        unit. The leading ``metadata`` section records the generation time,
        finite final score, and scoring configuration when available. Objective
        names are kept only inside ``optimization_scoring`` to avoid duplicate
        YAML metadata.

        Parameters
        ----------
        filepath : pathlib.Path
            Destination YAML path, including filename.
        overwrite : bool, default False
            Whether :meth:`ParameterIO.save_yaml` may replace an existing
            file.

        Raises
        ------
        ValueError
            If :meth:`optimize_parameters` has not been run.
        """
        from datetime import datetime
        from animakinetica.core.parameters import ParameterIO

        if self.optimal_parameters is None:
            raise ValueError("No optimal parameters available. Run optimize_parameters() first.")

        # Build nested YAML dict grouped by parameter section
        yaml_dict: dict = {}
        for param_path, value in self.optimal_parameters.items():
            section, param_name = param_path.split('.', 1)
            # Look up name and unit from the optimized simulation's parameter set
            try:
                param = self.optimized_simulation.parameters.get_parameter(param_path)
                param_display_name = param.name
                unit = param.unit
            except (ValueError, AttributeError):
                param_display_name = param_name
                unit = ""
            yaml_dict.setdefault(section, {})[param_name] = {
                'name': param_display_name,
                'value': float(value),
                'unit': unit,
            }

        # Build metadata block. The scalar and its components are calculated
        # from the exported optimized simulation so each component sums exactly
        # to the recorded final score.
        objective_value, objective_contributions = self._get_final_objective_metadata()
        metadata_block: dict = {
            'generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if objective_value is not None:
            metadata_block['objective_value'] = objective_value
        if objective_contributions:
            metadata_block['objective_contributions'] = objective_contributions
        if self.optimization_scoring is not None:
            metadata_block['optimization_scoring'] = self.optimization_scoring

        # Insert metadata first so it appears at the top of the file
        ordered: dict = {'metadata': metadata_block}
        ordered.update(yaml_dict)

        ParameterIO.save_yaml(ordered, filepath, overwrite=overwrite)

    def _get_final_objective_metadata(self) -> tuple[Optional[float], Dict[str, Dict[str, Any]]]:
        """Return the selected simulation's scalar score and KPI breakdown.

        The breakdown records raw KPI values, their direction-aware score,
        weights, and weighted contributions. Consequently, the sum of
        ``weighted_contribution`` values equals ``objective_value``.
        """
        if (
            self.optimized_simulation is None
            or self.objectives is None
            or self.optimization_scoring is None
        ):
            return None, {}

        summary_stats = self.optimized_simulation.data.summary_stats
        scoring_mode = self.optimization_scoring['mode']
        weights = self.optimization_scoring['weights']
        desirability = self.optimization_scoring.get('desirability')
        normalization_values = self.optimization_scoring.get('normalization_values')
        objective_directions = self.optimization_scoring.get('objective_directions', {})
        contributions: Dict[str, Dict[str, Any]] = {}
        objective_value = 0.0

        for objective, weight in zip(self.objectives, weights):
            raw_value = summary_stats.get(objective)
            if not isinstance(raw_value, (int, float, np.integer, np.floating)):
                return None, {}
            raw_value = float(raw_value)
            if not np.isfinite(raw_value):
                return None, {}

            if scoring_mode == 'engineering_desirability':
                config = desirability[objective]
                scaled_value = ObjectiveFunction.calculate_desirability(
                    raw_value,
                    config['minimum'],
                    config['target'],
                    config['safety_cap'],
                    direction=objective_directions.get(objective, 'maximize'),
                )
            elif normalization_values is not None:
                scaled_value = raw_value / float(normalization_values[objective])
            else:
                scaled_value = raw_value

            direction = objective_directions.get(objective, 'maximize')
            # Desirability is already larger-is-better for both objective
            # directions. Legacy raw and normalized values still need a sign
            # change for minimize KPIs before maximization.
            score_value = (
                scaled_value
                if scoring_mode == 'engineering_desirability' or direction == 'maximize'
                else -scaled_value
            )
            weighted_contribution = float(weight) * score_value
            if not np.isfinite(weighted_contribution):
                return None, {}
            contributions[objective] = {
                'raw_value': raw_value,
                'direction': direction,
                'score_value': score_value,
                'weight': float(weight),
                'weighted_contribution': weighted_contribution,
            }
            objective_value += weighted_contribution

        return float(objective_value), contributions

    ## Private Methods ##
    def _store_optimization_results(self, differential_evolution_result):
        """Store a SciPy result and rerun the selected parameter vector.

        Parameters
        ----------
        differential_evolution_result : scipy.optimize.OptimizeResult
            Result whose ``x`` vector follows the index order in
            ``self.opt_config``.
        """
        # Store results
        self.optimization_result = differential_evolution_result
        
        # Convert result to named parameters
        self.optimal_parameters = {}
        for param_name, config in self.opt_config.items():
            self.optimal_parameters[param_name] = differential_evolution_result.x[config['index']]

        # Store optimized simulation object
        self.optimized_simulation = self._create_optimized_simulation()

    def _create_optimized_simulation(self):
        """Create and run a full simulation using the selected parameters.

        The new :class:`CellCultureSimulation` preserves the baseline model,
        control, operating-mode, solver, and keyword settings. It runs with
        ``store_batches=True`` and is marked as optimized.

        Returns
        -------
        CellCultureSimulation
            Completed optimized simulation linked to this manager.

        Raises
        ------
        ValueError
            If no optimized parameter vector is available.
        """
        if self.optimal_parameters is None:
            raise ValueError("No optimal parameters available. Run optimize_parameters() first.")
        
        # Create optimized parameters
        optimized_param_obj = self.baseline_simulation.parameters.create_modified_parameter_object(self.optimal_parameters)
        
        # Create new simulation with optimized parameters
        optimized_sim = CellCultureSimulation(
            parameters=optimized_param_obj,
            kinetics=self.baseline_simulation.kinetics,
            gas_controller=self.baseline_simulation.gas_controller,
            feed_strategy=self.baseline_simulation.feed_strategy,
            perfusion_strategy=self.baseline_simulation.perfusion_strategy,
            operating_mode=self.baseline_simulation.operating_mode,
            stoichiometry_mode=self.baseline_simulation.stoichiometry_mode,
            performance_mode=self.baseline_simulation.performance_mode,
            solver_method=self.baseline_simulation.solver_method,
            **self.baseline_simulation._kwargs
        )
        
        # Run the simulation
        optimized_sim.run_simulation(store_batches=True)
        optimized_sim.optimized = True  # Mark as optimized for reference
        optimized_sim.optimization_manager = self  # Link back to optimization manager for reference

        return optimized_sim

class ObjectiveFunction:
    """Evaluate one candidate parameter vector for differential evolution.

    Instances are pickleable so SciPy can distribute evaluations across worker
    processes. Each call creates modified parameters and invokes
    :meth:`CellCultureSimulation.run_temporary_simulation`. Hard failures
    return ``1e6``; finite operational constraint violations return a graded
    positive penalty; valid candidates return the negative weighted score.

    Parameters
    ----------
    baseline_simulation : CellCultureSimulation
        Reference simulation supplying the parameter object, model settings,
        runtime horizon, and baseline totals.
    opt_config : dict
        Mapping of dotted parameter paths to configurations containing an
        ``index`` used to read ``opt_vars``.
    objectives : list of str
        Summary-statistic keys. Their orientation is resolved from
        ``SUMMARY_METRIC_DIRECTIONS``; unknown keys default to maximize.
    weights : list of float
        Weights applied to objectives in the same order.
    timeout : float, default 5
        Maximum wall-clock time in seconds for a temporary simulation.
    baseline_values : dict of str to float, optional
        Objective normalization divisors. If omitted, raw values are scored.
    debug_rejections : bool, default False
        Whether to print candidate rejection diagnostics.
    """

    _UNWANTED_TERMINATION_REASONS = frozenset({
        # 'glucose depletion',
        # 'glutamine depletion',
        # 'aa depletion',
        # 'oxygen depletion',
        # 'ph deviation',
        'invalid initial ph',
        'cell minimum',
        'unknown termination',
    })
    _CONTINUOUS_ALLOWED_TERMINATION_REASONS = frozenset({
        'maximum time reached',
    })
    _CONTINUOUS_MIN_RUNTIME_FRACTION = 0.995
    _SIMULATION_WINDOW_MIN_RUNTIME_FRACTION = 0.995
    _MIN_ACTIVE_PHASE_RUNTIME_H = 1e-6

    _MIN_BATCH_COUNT_ABSOLUTE = 2
    _MIN_BATCH_COUNT_FRACTION = 0.25
    _MIN_BATCH_DCM_FRACTION = 0.20
    _MIN_FINAL_VOLUME_FRACTION = 0.98
    _SHORT_SINGLE_BATCH_RUNTIME_FRACTION = 0.70
    _MAX_ACID_BASE_FRACTION = 0.05  # Reject when acid and base additions exceed 5% of total medium volume
    _MAX_CLOSURE_PCT = 5.0

    # Soft validity constraints are ranked below every valid objective value but
    # above hard failures. This gives differential evolution a directional
    # signal towards the feasible region rather than making every near-miss
    # indistinguishable from a failed integration.
    _SOFT_PENALTY_BASE = 1e3
    _SOFT_PENALTY_SCALE = 1e3
    DESIRABILITY_SAFETY_CAP = 1.5
    
    def __init__(self, baseline_simulation: CellCultureSimulation, 
                 opt_config: Dict, objectives: List[str], weights: List[float],
                 timeout: float = 5, baseline_values: Optional[Dict[str, float]] = None,
                 desirability_config: Optional[Dict[str, Dict[str, float]]] = None,
                 debug_rejections: bool = False):
        """Initialize a candidate evaluator.

        Parameters are described in :class:`ObjectiveFunction`. The baseline
        ``summary_stats``, projected totals, and simulated-window totals are
        captured for validity comparisons.
        """
        self.sim = baseline_simulation
        self.baseline_params = baseline_simulation.parameters
        self.kwargs = baseline_simulation._kwargs
        self.opt_config = opt_config
        self.objectives = objectives
        self.weights = weights
        self.objective_directions = {
            objective: SUMMARY_METRIC_DIRECTIONS.get(objective, 'maximize')
            for objective in objectives
        }
        self.timeout = timeout
        self.baseline_values = baseline_values  # None → no normalization
        if desirability_config is not None and baseline_values is not None:
            raise ValueError("desirability_config and baseline_values are mutually exclusive.")
        self.desirability_config = (
            self.validate_desirability_config(desirability_config, objectives)
            if desirability_config is not None else None
        )
        self.debug_rejections = bool(debug_rejections)
        self.baseline_summary_stats = baseline_simulation.data.summary_stats
        self.baseline_projected_totals = baseline_simulation.data.projected_totals_stats
        self.baseline_simulated_totals = baseline_simulation.data.simulated_window_totals_stats

    @classmethod
    def validate_desirability_config(
        cls,
        desirability_config: Dict[str, Dict[str, float]],
        objectives: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Validate engineering desirability settings for the active objectives."""
        if not isinstance(desirability_config, dict):
            raise ValueError("desirability_config must be a mapping keyed by objective name.")

        configured_objectives = set(desirability_config)
        active_objectives = set(objectives)
        missing = [objective for objective in objectives if objective not in configured_objectives]
        extra = sorted(configured_objectives - active_objectives)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing settings for {missing}")
            if extra:
                details.append(f"unexpected settings for {extra}")
            raise ValueError("Invalid desirability_config: " + "; ".join(details) + ".")

        validated: Dict[str, Dict[str, float]] = {}
        for objective in objectives:
            config = desirability_config[objective]
            if not isinstance(config, dict):
                raise ValueError(f"Desirability settings for '{objective}' must be a mapping.")
            required = {'minimum', 'target'}
            optional = {'safety_cap'}
            missing_keys = sorted(required - set(config))
            extra_keys = sorted(set(config) - required - optional)
            if missing_keys or extra_keys:
                details = []
                if missing_keys:
                    details.append(f"missing keys {missing_keys}")
                if extra_keys:
                    details.append(f"unexpected keys {extra_keys}")
                raise ValueError(
                    f"Invalid desirability settings for '{objective}': "
                    + "; ".join(details) + "."
                )
            try:
                minimum = float(config['minimum'])
                target = float(config['target'])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Desirability settings for '{objective}' require numeric minimum and target values."
                ) from exc
            if not np.isfinite(minimum) or not np.isfinite(target):
                raise ValueError(
                    f"Desirability settings for '{objective}' must be finite; "
                    f"got minimum={minimum}, target={target}."
                )
            direction = SUMMARY_METRIC_DIRECTIONS.get(objective, 'maximize')
            if direction == 'maximize' and target <= minimum:
                raise ValueError(
                    f"Maximize desirability target must exceed minimum for '{objective}'; "
                    f"got minimum={minimum}, target={target}."
                )
            if direction == 'minimize' and target >= minimum:
                raise ValueError(
                    f"Minimize desirability target must be below the upper unacceptable "
                    f"minimum for '{objective}'; got minimum={minimum}, target={target}."
                )
            try:
                safety_cap = float(config.get('safety_cap', cls.DESIRABILITY_SAFETY_CAP))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Desirability safety_cap for '{objective}' must be numeric."
                ) from exc
            if not np.isfinite(safety_cap) or safety_cap <= 1.0:
                raise ValueError(
                    f"Desirability safety_cap for '{objective}' must be finite and greater than 1; "
                    f"got safety_cap={safety_cap}."
                )
            validated[objective] = {
                'minimum': minimum,
                'target': target,
                'safety_cap': safety_cap,
            }
        return validated

    @classmethod
    def calculate_desirability(
        cls,
        value: float,
        minimum: float,
        target: float,
        safety_cap: float = DESIRABILITY_SAFETY_CAP,
        direction: str = 'maximize',
    ) -> float:
        """Map a KPI to a configurable capped desirability.

        For maximize KPIs, ``minimum`` maps to 0 and ``target`` maps to 1.
        For minimize KPIs, ``target`` is the desirable low value (maps to 1)
        and ``minimum`` is the upper unacceptable value (maps to 0). Values
        beyond the target remain beneficial but asymptotically approach the
        fixed ``safety_cap``.

        For more information about this optimization approach, please see:
        https://www.itl.nist.gov/div898/handbook/pri/section5/pri5322.htm
        """
        if direction == 'maximize':
            normalized_value = max(0.0, (value - minimum) / (target - minimum))
        elif direction == 'minimize':
            normalized_value = max(0.0, (minimum - value) / (minimum - target))
        else:
            raise ValueError(f"Unknown objective direction '{direction}'.")
        target_rate = -np.log(1.0 - 1.0 / safety_cap)
        desirability = safety_cap * (
            1.0 - np.exp(-target_rate * normalized_value)
        )
        return float(min(safety_cap, desirability))

    def _scale_objective_value(self, objective: str, value: float) -> float:
        """Apply engineering desirability or the selected legacy scaling."""
        if self.desirability_config is not None:
            config = self.desirability_config[objective]
            return self.calculate_desirability(
                value,
                config['minimum'],
                config['target'],
                config['safety_cap'],
                direction=self.objective_directions[objective],
            )

        if self.baseline_values is not None:
            baseline_value = float(self.baseline_values[objective])
            if not np.isfinite(baseline_value) or abs(baseline_value) < 1e-12:
                raise ValueError(
                    f"Invalid baseline normalization for '{objective}': {baseline_value}."
                )
            return value / baseline_value

        return value

    def _reject(self, reason: str, details: Optional[str] = None) -> float:
        """Return the fixed hard-rejection penalty.

        Parameters
        ----------
        reason : str
            Machine-readable rejection category.
        details : str, optional
            Context appended to diagnostic output when enabled.

        Returns
        -------
        float
            Hard-rejection penalty, ``1e6``.
        """
        if self.debug_rejections:
            mode = getattr(self.sim.operating_mode, 'name', str(self.sim.operating_mode))
            if details:
                print(f"[ObjectiveFunction][reject][mode={mode}] {reason}: {details}")
            else:
                print(f"[ObjectiveFunction][reject][mode={mode}] {reason}")
        return 1e6

    def _soft_reject(self, violations: Dict[str, float]) -> float:
        """Return a graded penalty for finite, measurable constraint violations.

        Each value is a non-negative normalized excess above its constraint.
        A valid candidate always returns its negative objective, while even
        the smallest soft violation remains worse than every valid candidate.

        Parameters
        ----------
        violations : dict of str to float
            Named normalized constraint excesses.

        Returns
        -------
        float
            Positive penalty based on the total normalized excess.
        """
        total_violation = float(sum(violations.values()))
        penalty = self._SOFT_PENALTY_BASE + self._SOFT_PENALTY_SCALE * total_violation
        if self.debug_rejections:
            mode = getattr(self.sim.operating_mode, 'name', str(self.sim.operating_mode))
            details = "; ".join(f"{name}={value:.6g}" for name, value in violations.items())
            print(
                f"[ObjectiveFunction][soft_reject][mode={mode}] "
                f"penalty={penalty:.6g}; total_violation={total_violation:.6g}; {details}"
            )
        return penalty

    def _has_unwanted_termination(self, simulated_totals: Dict[str, Any]) -> bool:
        """Check for globally disallowed batch termination reasons.

        Parameters
        ----------
        simulated_totals : dict
            Raw simulated-window totals from ``SimulationData``.

        Returns
        -------
        bool
            ``True`` if the final or any unique reason is in
            ``_UNWANTED_TERMINATION_REASONS``.
        """
        termination_reason = str(simulated_totals.get('last_batch_termination_reason', '')).strip().lower()
        termination_reasons = {
            str(reason).strip().lower()
            for reason in simulated_totals.get('unique_batch_termination_reasons', [])
            if str(reason).strip()
        }

        # Always reject globally disallowed reasons, independent of operating mode.
        # This prevents trajectories ending in e.g. "cell minimum" from slipping
        # through in modes that are not explicitly listed below (such as PERFUSION
        # and CONCENTRATED_FEDBATCH).
        if termination_reasons & self._UNWANTED_TERMINATION_REASONS:
            return True

        if termination_reason in self._UNWANTED_TERMINATION_REASONS:
            return True

        if self.sim.operating_mode in CONTINUOUS_MODES:
            return False

        return False

    def _has_incomplete_batch_volume(self, simulated_totals: Dict[str, Any]) -> bool:
        """Check whether any logical batch ends below the volume limit.

        Parameters
        ----------
        simulated_totals : dict
            Raw simulated-window totals from ``SimulationData``.

        Returns
        -------
        bool
            ``True`` for malformed volume data or a minimum final-volume
            fraction below ``_MIN_FINAL_VOLUME_FRACTION``.
        """
        min_fraction = simulated_totals.get('batch_final_volume_min_fraction', None)
        if min_fraction is None:
            # Backward compatibility with older totals payloads.
            min_fraction = simulated_totals.get('last_batch_final_volume_fraction', 0.0)
        try:
            min_fraction = float(min_fraction)
        except (TypeError, ValueError):
            return True
        return min_fraction < self._MIN_FINAL_VOLUME_FRACTION

    def _has_short_single_batch_runtime(self, simulated_totals: Dict[str, Any]) -> bool:
        """Return True when a repeated-batch candidate collapses to one short batch.

        Applies only to batch-style campaign modes where multiple batches should
        fit inside the configured vessel lifetime window.

        Parameters
        ----------
        simulated_totals : dict
            Raw simulated-window totals from ``SimulationData``.

        Returns
        -------
        bool
            ``True`` when a batch-style candidate contains at most one batch
            and runs for less than the configured runtime fraction.
        """
        startup_fedbatch_mode = getattr(OperatingMode, 'BATCH_WITH_STARTUP_FEDBATCH', None)
        repeated_batch_modes = {OperatingMode.BATCH, startup_fedbatch_mode}
        if self.sim.operating_mode not in repeated_batch_modes:
            return False

        candidate_batches = int(simulated_totals.get('number_of_batches', 0) or 0)
        if candidate_batches > 1:
            return False

        simulated_time_h = float(simulated_totals.get('total_simulation_time_h', 0.0) or 0.0)
        max_culture_age_h = float(getattr(self.sim, 'max_culture_age', 0.0) or 0.0)
        if max_culture_age_h <= 0:
            return False

        return simulated_time_h < (self._SHORT_SINGLE_BATCH_RUNTIME_FRACTION * max_culture_age_h)

    def _get_expected_simulation_window_target_h(self) -> float:
        """Return the expected simulated-window horizon used by temporary runs.

        ``run_temporary_simulation(..., project_yearly=True)`` executes one
        simulated window with ``t_max_simulation = min(max_culture_age,
        yearly_uptime)`` and then scales additive totals for yearly reporting.
        Runtime-completeness checks must therefore use this raw simulated-window
        target rather than projected totals.

        Returns
        -------
        float
            Expected temporary-simulation horizon in hours, or ``0.0`` if no
            positive horizon is configured.
        """
        max_culture_age_h = float(getattr(self.sim, 'max_culture_age', 0.0) or 0.0)
        yearly_uptime_h = float(getattr(self.sim, 'yearly_uptime', 0.0) or 0.0)

        if max_culture_age_h > 0.0 and yearly_uptime_h > 0.0:
            return min(max_culture_age_h, yearly_uptime_h)
        if max_culture_age_h > 0.0:
            return max_culture_age_h
        if yearly_uptime_h > 0.0:
            return yearly_uptime_h
        return 0.0

    def _has_incomplete_simulation_window(self, simulated_totals: Dict[str, Any]) -> bool:
        """Return True when simulated runtime is shorter than the expected window.

        This guard is mode-agnostic and catches early-stop trajectories (e.g.,
        dilution/washout) across both batch-like and continuous campaigns.

        Parameters
        ----------
        simulated_totals : dict
            Raw simulated-window totals from ``SimulationData``.

        Returns
        -------
        bool
            ``True`` when runtime is below the required fraction of the
            expected temporary-simulation horizon.
        """
        target_time_h = self._get_expected_simulation_window_target_h()
        if target_time_h <= 0.0:
            return False

        simulated_time_h = float(simulated_totals.get('total_simulation_time_h', 0.0) or 0.0)
        return simulated_time_h < (self._SIMULATION_WINDOW_MIN_RUNTIME_FRACTION * target_time_h)

    def _has_incomplete_continuous_runtime(self, simulated_totals: Dict[str, Any]) -> bool:
        """Return True when a continuous-mode run does not complete its full runtime window.

        Continuous modes are expected to run until the vessel-lifetime cap
        (``max_culture_age``). Any event-based early stop should be rejected.

        Parameters
        ----------
        simulated_totals : dict
            Raw simulated-window totals from ``SimulationData``.

        Returns
        -------
        bool
            ``True`` for a continuous-mode candidate that terminates for an
            unallowed reason or before the required runtime fraction.
        """
        if self.sim.operating_mode not in CONTINUOUS_MODES:
            return False

        termination_reason = str(simulated_totals.get('last_batch_termination_reason', '')).strip().lower()
        termination_reasons = {
            str(reason).strip().lower()
            for reason in simulated_totals.get('unique_batch_termination_reasons', [])
            if str(reason).strip()
        }

        allowed = self._CONTINUOUS_ALLOWED_TERMINATION_REASONS
        if termination_reason not in allowed:
            return True
        if any(reason not in allowed for reason in termination_reasons):
            return True

        simulated_time_h = float(simulated_totals.get('total_simulation_time_h', 0.0) or 0.0)
        target_time_h = float(getattr(self.sim, 'max_culture_age', 0.0) or 0.0)
        if target_time_h > 0.0 and simulated_time_h < self._CONTINUOUS_MIN_RUNTIME_FRACTION * target_time_h:
            return True

        return False

    def _get_inactive_required_phases(self, simulated_totals: Dict[str, Any]) -> List[str]:
        """Return scheduled multi-phase operations with no active runtime.

        A candidate is infeasible when a configured downstream operation (for
        example PERFUSION in a concentrated fed-batch) never receives a
        positive integration interval. This prevents optimization from scoring
        a simpler upstream-only trajectory as the requested operating mode.
        """
        required_phases = tuple(self.sim.ode_systems)
        if len(required_phases) < 2:
            return []

        phase_durations = simulated_totals.get('phase_active_runtime_h')
        if not isinstance(phase_durations, dict):
            return list(required_phases)

        inactive_phases: List[str] = []
        for phase_name in required_phases:
            try:
                duration_h = float(phase_durations.get(phase_name, 0.0))
            except (TypeError, ValueError):
                inactive_phases.append(phase_name)
                continue
            if not np.isfinite(duration_h) or duration_h <= self._MIN_ACTIVE_PHASE_RUNTIME_H:
                inactive_phases.append(phase_name)
        return inactive_phases

    def _get_inactive_batch_phases(self, simulated_totals: Dict[str, Any]) -> Dict[int, List[str]]:
        """Return logical-batch phases below the active-runtime threshold.

        Run-level ``phase_active_runtime_h`` proves only that a phase ran at
        least once. A final logical batch can still exhaust the vessel lifetime
        in an upstream phase, leaving a downstream phase unrun or effectively
        zero-duration. Per-batch phase durations emitted by
        :meth:`CellCultureSimulation.run_simulation` close that validity gap.

        Every phase scheduled for a logical batch must run for at least
        :attr:`_MIN_ACTIVE_PHASE_RUNTIME_H`. ``BATCH_WITH_STARTUP_FEDBATCH``
        is an exception: its FEDBATCH phase is a one-time startup operation,
        so only its first logical batch is required to include FEDBATCH.

        Parameters
        ----------
        simulated_totals : dict
            Raw simulated-window totals from ``SimulationData``.

        Returns
        -------
        dict of int to list of str
            Logical batch numbers and their inactive phase names. A malformed
            duration payload returns ``{-1: ["invalid phase-duration data"]}``
            so the candidate is rejected.
        """
        required_phases = tuple(self.sim.ode_systems)
        if len(required_phases) < 2:
            return {}

        batch_durations = simulated_totals.get('batch_phase_active_runtime_h')
        if not isinstance(batch_durations, list) or not batch_durations:
            return {-1: ['missing phase-duration data']}

        inactive_batch_phases: Dict[int, List[str]] = {}
        for fallback_batch_number, durations in enumerate(batch_durations, start=1):
            if not isinstance(durations, dict):
                return {-1: ['invalid phase-duration data']}
            batch_number = durations.get('batch_number', fallback_batch_number)
            try:
                batch_number = int(batch_number)
            except (TypeError, ValueError):
                return {-1: ['invalid batch number']}

            phases_required_for_batch = required_phases
            startup_fedbatch_mode = getattr(OperatingMode, 'BATCH_WITH_STARTUP_FEDBATCH', None)
            if (
                self.sim.operating_mode == startup_fedbatch_mode
                and batch_number != 1
            ):
                phases_required_for_batch = tuple(
                    phase_name for phase_name in required_phases
                    if phase_name != 'FEDBATCH'
                )

            inactive_phases: List[str] = []
            for phase_name in phases_required_for_batch:
                try:
                    duration_h = float(durations.get(phase_name, 0.0))
                except (TypeError, ValueError):
                    return {-1: ['invalid phase-duration data']}
                if not np.isfinite(duration_h):
                    return {-1: ['invalid phase-duration data']}
                if duration_h < self._MIN_ACTIVE_PHASE_RUNTIME_H:
                    inactive_phases.append(phase_name)
            if inactive_phases:
                inactive_batch_phases[batch_number] = inactive_phases

        return inactive_batch_phases

    def _has_unrealistic_batch_productivity(
        self,
        sim_summary_stats: Dict[str, Any],
        simulated_totals: Dict[str, Any],
    ) -> bool:
        """Check whether a batch candidate collapses repeated-batch output.

        Parameters
        ----------
        sim_summary_stats : dict
            Candidate summary statistics.
        simulated_totals : dict
            Candidate raw simulated-window totals.

        Returns
        -------
        bool
            ``True`` when batch count or total dry-cell-mass production falls
            below the configured baseline-relative limit.
        """
        startup_fedbatch_mode = getattr(OperatingMode, 'BATCH_WITH_STARTUP_FEDBATCH', None)
        repeated_batch_modes = {OperatingMode.BATCH, startup_fedbatch_mode}
        if self.sim.operating_mode not in repeated_batch_modes:
            return False

        # Use raw simulated-window batch counts so the check is not inflated by
        # yearly projection scaling.
        candidate_batches = int(simulated_totals.get('number_of_batches', 0) or 0)
        baseline_batches  = int(self.baseline_simulated_totals.get('number_of_batches', 0) or 0)
        if baseline_batches > 0:
            min_batches = max(
                self._MIN_BATCH_COUNT_ABSOLUTE,
                int(np.ceil(baseline_batches * self._MIN_BATCH_COUNT_FRACTION)),
            )
        else:
            min_batches = self._MIN_BATCH_COUNT_ABSOLUTE

        if candidate_batches < min_batches:
            return True

        baseline_total_dcm = float(
            self.baseline_simulated_totals.get(
                'total_DCM_kg',
                self.baseline_summary_stats.get('total_DCM_kg', 0.0),
            ) or 0.0
        )
        if baseline_total_dcm <= 0:
            return False

        candidate_total_dcm = float(
            simulated_totals.get(
                'total_DCM_kg',
                sim_summary_stats.get('total_DCM_kg', 0.0),
            ) or 0.0
        )
        min_total_dcm = baseline_total_dcm * self._MIN_BATCH_DCM_FRACTION
        return candidate_total_dcm < min_total_dcm

    def _has_unrealistic_acid_base_addition(self, simulated_totals: Dict[str, Any]) -> bool:
        """Check whether acid and base additions exceed the medium-use limit.

        High pH-control additions can create unrealistic dilution that reduces
        inhibitory metabolites without corresponding process feed.

        Parameters
        ----------
        simulated_totals : dict
            Raw simulated-window totals from ``SimulationData``.

        Returns
        -------
        bool
            ``True`` when acid plus base volume exceeds
            ``_MAX_ACID_BASE_FRACTION`` of total medium volume.
        """
        total_base = float(simulated_totals.get('total_base_in_L', 0.0) or 0.0)
        total_acid = float(simulated_totals.get('total_acid_in_L', 0.0) or 0.0)
        total_medium = float(simulated_totals.get('total_medium_in_L', 0.0) or 0.0)
        if total_medium <= 0:
            return False
        return ((total_base + total_acid) / total_medium) > self._MAX_ACID_BASE_FRACTION

    def _has_unacceptable_mass_balance_closure(
        self,
        sim_data: SimulationData,
        max_closure_pct: float = 5.0,
    ) -> bool:
        """Check whether external-boundary CHON closure exceeds tolerance.

        Delegates to :meth:`SimulationData.compute_boundary_closure`
        to compute the whole-process boundary element balance and checks each
        CHON element's relative residual individually.

        Parameters
        ----------
        sim_data : SimulationData
            Full candidate result object.
        max_closure_pct : float, default 5.0
            Maximum absolute relative residual for each element, in percent.

        Returns
        -------
        bool
            ``True`` if an available element residual exceeds the limit.
            Closure data that cannot be calculated does not reject a candidate.
        """
        try:
            closure = sim_data.compute_boundary_closure()
        except (ValueError, KeyError):
            return False  # Cannot evaluate — do not reject

        for _element, values in closure['elements'].items():
            if abs(values['relative_residual_pct']) > max_closure_pct:
                return True
        return False

    @staticmethod
    def _normalized_deficit(actual: float, minimum: float) -> float:
        """Return the normalized shortfall below a threshold.

        Parameters
        ----------
        actual : float
            Measured value.
        minimum : float
            Required positive minimum.

        Returns
        -------
        float
            ``(minimum - actual) / minimum`` when below the threshold;
            otherwise ``0.0``.
        """
        return max(0.0, (minimum - actual) / minimum)

    @staticmethod
    def _normalized_excess(actual: float, maximum: float) -> float:
        """Return the normalized excess above a threshold.

        Parameters
        ----------
        actual : float
            Measured value.
        maximum : float
            Allowed positive maximum.

        Returns
        -------
        float
            ``(actual - maximum) / maximum`` when above the threshold;
            otherwise ``0.0``.
        """
        return max(0.0, (actual - maximum) / maximum)

    def _get_soft_constraint_violations(
        self,
        sim_data: SimulationData,
        simulated_totals: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """Measure soft optimization-constraint violations.

        The returned values are normalized relative excesses. ``None`` marks a
        malformed or non-finite required measurement, which remains a hard
        rejection because its distance to feasibility cannot be determined.

        Parameters
        ----------
        sim_data : SimulationData
            Full candidate result object used for boundary-closure evaluation.
        simulated_totals : dict
            Candidate raw simulated-window totals.

        Returns
        -------
        dict of str to float or None
            Positive normalized violations for final volume, runtime,
            acid/base fraction, and available boundary closure; ``None`` when
            a required measurement is invalid.
        """
        violations: Dict[str, float] = {}

        min_volume_fraction = simulated_totals.get('batch_final_volume_min_fraction')
        if min_volume_fraction is None:
            min_volume_fraction = simulated_totals.get('last_batch_final_volume_fraction')
        try:
            min_volume_fraction = float(min_volume_fraction)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(min_volume_fraction):
            return None
        violations['final_volume'] = self._normalized_deficit(
            min_volume_fraction,
            self._MIN_FINAL_VOLUME_FRACTION,
        )

        target_time_h = self._get_expected_simulation_window_target_h()
        if target_time_h > 0.0:
            try:
                simulated_time_h = float(simulated_totals.get('total_simulation_time_h', 0.0) or 0.0)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(simulated_time_h):
                return None
            required_time_h = self._SIMULATION_WINDOW_MIN_RUNTIME_FRACTION * target_time_h
            violations['simulation_window'] = self._normalized_deficit(
                simulated_time_h,
                required_time_h,
            )

        try:
            total_base = float(simulated_totals.get('total_base_in_L', 0.0) or 0.0)
            total_acid = float(simulated_totals.get('total_acid_in_L', 0.0) or 0.0)
            total_medium = float(simulated_totals.get('total_medium_in_L', 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(value) for value in (total_base, total_acid, total_medium)):
            return None
        if total_medium > 0.0:
            acid_base_fraction = (total_base + total_acid) / total_medium
            violations['acid_base'] = self._normalized_excess(
                acid_base_fraction,
                self._MAX_ACID_BASE_FRACTION,
            )

        try:
            closure = sim_data.compute_boundary_closure()
        except (ValueError, KeyError):
            # Preserve the existing behavior: unavailable closure is not a
            # rejection because no meaningful distance can be calculated.
            closure = None
        if closure is not None:
            try:
                max_closure_pct = max(
                    abs(float(values['relative_residual_pct']))
                    for values in closure['elements'].values()
                )
            except (KeyError, TypeError, ValueError):
                return None
            if not np.isfinite(max_closure_pct):
                return None
            violations['boundary_closure'] = self._normalized_excess(
                max_closure_pct,
                self._MAX_CLOSURE_PCT,
            )

        return {name: value for name, value in violations.items() if value > 0.0}
    

    def __call__(self, opt_vars):
        """Evaluate an optimization vector.

        The vector is converted to a dotted parameter mapping according to
        ``opt_config``. A temporary simulation is then screened for runtime,
        termination, productivity, operational, and closure constraints before
        its summary metrics are scored.

        Parameters
        ----------
        opt_vars : array-like of float
            Candidate values ordered by the ``index`` entries in ``opt_config``.

        Returns
        -------
        float
            Negative weighted objective for a valid candidate; a positive soft
            penalty for measurable constraint violations; or ``1e6`` for hard
            rejections.

        Notes
        -----
        Calls :meth:`CellCultureSimulation.run_temporary_simulation` with
        ``return_only_summary_stats=False`` because
        :meth:`SimulationData.compute_boundary_closure` requires the result
        object rather than summary statistics alone.
        """
        try:
            # Convert array to parameter dictionary
            opt_params = {name: opt_vars[config['index']] 
                         for name, config in self.opt_config.items()}
            
            # Enhanced bounds checking for faster rejection
            if self._bounds_check(opt_params):
                return self._reject("bounds_check_failed")
            
            # Create modified parameter object
            optimized_params = self.baseline_params.create_modified_parameter_object(opt_params)
            
            # Run temporary simulation with optimized parameters.
            # return_only_summary_stats=False so we get the full SimulationData
            # object, which is needed for the mass-balance boundary CHON closure
            # check (compute_boundary_closure).
            # store_batches=False inside run_temporary_simulation keeps memory
            # minimal — only the stats dicts and parameters are populated.
            sim_data = CellCultureSimulation.run_temporary_simulation(
                params=optimized_params,
                kinetics=self.sim.kinetics,
                gas_controller=self.sim.gas_controller,
                feed_strategy=self.sim.feed_strategy,
                perfusion_strategy=self.sim.perfusion_strategy,
                operating_mode=self.sim.operating_mode,
                stoichiometry_mode=self.sim.stoichiometry_mode,
                performance_mode=self.sim.performance_mode,
                solver_method=self.sim.solver_method,
                return_only_summary_stats=False,
                wall_timeout=self.timeout,
                debug_failures=self.debug_rejections,
                **self.kwargs
            )

            # Early termination checks for sanity and performance
            if sim_data is None or sim_data == 0:
                return self._reject("temporary_simulation_failed")

            sim_summary_stats = sim_data.summary_stats
            simulated_totals = sim_data.simulated_window_totals_stats
            if not sim_summary_stats or not simulated_totals:
                return self._reject("missing_summary_or_totals")
            
            # Check simulation duration early.
            # Use the raw simulated-window time so the check is not fooled by
            # yearly projection scaling (which inflates total_simulation_time_h).
            total_time_h = float(simulated_totals.get('total_simulation_time_h', 0) or 0)
            if total_time_h < 72:
                return self._reject("short_simulation_time", f"total_simulation_time_h={total_time_h:.3f}")

            # Reject trajectories that end in depletion/failure.
            if self._has_unwanted_termination(simulated_totals):
                return self._reject(
                    "unwanted_termination",
                    (
                        f"last={simulated_totals.get('last_batch_termination_reason', '')}; "
                        f"unique={simulated_totals.get('unique_batch_termination_reasons', [])}; "
                        f"last_batch_final_volume_fraction={simulated_totals.get('last_batch_final_volume_fraction', None)}"
                    ),
                )

            if self._has_short_single_batch_runtime(simulated_totals):
                simulated_h = float(simulated_totals.get('total_simulation_time_h', 0.0) or 0.0)
                max_culture_age_h = float(getattr(self.sim, 'max_culture_age', 0.0) or 0.0)
                runtime_fraction = (simulated_h / max_culture_age_h) if max_culture_age_h > 0 else float('nan')
                return self._reject(
                    "single_short_batch_runtime",
                    (
                        f"number_of_batches={simulated_totals.get('number_of_batches', 0)}; "
                        f"simulated_h={simulated_h:.3f}; "
                        f"max_culture_age_h={max_culture_age_h:.3f}; "
                        f"runtime_fraction={runtime_fraction:.4f}"
                    ),
                )

            # Continuous modes should use the full available cultivation window.
            # Reject trajectories that terminate by any event before max_culture_age.
            if self._has_incomplete_continuous_runtime(simulated_totals):
                return self._reject(
                    "continuous_runtime_incomplete",
                    (
                        f"simulated_h={simulated_totals.get('total_simulation_time_h', None)}; "
                        f"target_h={getattr(self.sim, 'max_culture_age', None)}; "
                        f"last={simulated_totals.get('last_batch_termination_reason', '')}; "
                        f"unique={simulated_totals.get('unique_batch_termination_reasons', [])}"
                    ),
                )

            inactive_phases = self._get_inactive_required_phases(simulated_totals)
            if inactive_phases:
                return self._reject(
                    "required_phase_inactive",
                    (
                        f"inactive={inactive_phases}; "
                        f"phase_active_runtime_h={simulated_totals.get('phase_active_runtime_h', {})}"
                    ),
                )

            inactive_batch_phases = self._get_inactive_batch_phases(simulated_totals)
            if inactive_batch_phases:
                return self._reject(
                    "required_batch_phase_inactive",
                    (
                        f"minimum_active_runtime_h={self._MIN_ACTIVE_PHASE_RUNTIME_H:.6g}; "
                        f"batch_phase_active_runtime_h="
                        f"{simulated_totals.get('batch_phase_active_runtime_h', None)}; "
                        f"inactive_batch_phases={inactive_batch_phases}"
                    ),
                )

            # Batch-mode optimization can otherwise drift into starvation regimes
            # that look efficient on a per-kg basis but collapse repeated-batch output.
            if self._has_unrealistic_batch_productivity(sim_summary_stats, simulated_totals):
                return self._reject(
                    "unrealistic_batch_productivity",
                    (
                        f"candidate_batches={simulated_totals.get('number_of_batches', 0)}; "
                        f"candidate_total_DCM_kg={simulated_totals.get('total_DCM_kg', sim_summary_stats.get('total_DCM_kg', 0))}"
                    ),
                )

            # Apply graded penalties to finite, measurable near-misses. This
            # preserves hard failures above, while allowing DE to rank candidates
            # by their distance to the volume, runtime, acid/base, and closure
            # validity limits.
            soft_violations = self._get_soft_constraint_violations(sim_data, simulated_totals)
            if soft_violations is None:
                return self._reject("invalid_soft_constraint_measurement")
            if soft_violations:
                return self._soft_reject(soft_violations)
            
            # # Check for minimum amount of DCM produced --> Gave problems with optimization convergence.
            # if sim_summary_stats.get('total_DCM_kg', 0) < 5:
            #     return 1e6

            # Calculate the weighted score. Engineering desirability maps both
            # maximize and minimize KPIs onto a larger-is-better scale. Raw
            # and baseline-normalized input-per-product KPIs remain costs and
            # are negated before maximization.
            weighted_sum = 0.0
            for obj, weight in zip(self.objectives, self.weights):
                if obj not in sim_summary_stats:
                    return self._reject("missing_objective", f"objective '{obj}' not found in summary stats")

                value = sim_summary_stats[obj]
                # Apply penalty for zero, non-numeric, or non-finite values
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    return self._reject("invalid_objective_value", f"{obj}={value}")
                value = float(value)
                if not np.isfinite(value) or value == 0.0:
                    return self._reject("invalid_objective_value", f"{obj}={value}")

                try:
                    value = self._scale_objective_value(obj, value)
                except (KeyError, TypeError, ValueError) as exc:
                    return self._reject("invalid_objective_scaling", str(exc))

                if not np.isfinite(value):
                    return self._reject("non_finite_normalized_objective", f"{obj}={value}")

                direction = self.objective_directions[obj]
                score_value = (
                    value
                    if self.desirability_config is not None or direction == 'maximize'
                    else -value
                )
                weighted_sum += weight * score_value

            if not np.isfinite(weighted_sum):
                return self._reject("non_finite_weighted_sum", f"weighted_sum={weighted_sum}")
            
            # Return negative for maximization
            return -weighted_sum
        
        finally:
            # Force garbage collection to reclaim memory in long-lived
            # multiprocessing Pool workers (prevents gradual RSS growth
            # from heap fragmentation over thousands of evaluations).
            gc.collect()
    
    def _bounds_check(self, opt_params_dict):
        """Apply inexpensive consistency and washout checks before simulation.

        Parameters
        ----------
        opt_params_dict : dict
            Proposed dotted parameter paths and values.

        Returns
        -------
        bool
            ``True`` when a candidate violates feed, growth, volume, perfusion
            step-ordering, or cell-loss constraints and should be rejected.

        Notes
        -----
        For perfusion, cell loss is calculated as
        ``B + D * (1 - eff_separator)``. Parameters absent from the proposed
        mapping retain their baseline values.
        """
        
        # Glucose checks
        if 'feed_parameters.C_Glc_feed' in opt_params_dict and 'initial_conditions.C_Glc_init' in opt_params_dict:
            if opt_params_dict['feed_parameters.C_Glc_feed'] < opt_params_dict['initial_conditions.C_Glc_init']:
                return True
        
        # Glutamine checks
        if 'feed_parameters.C_Gln_feed' in opt_params_dict and 'initial_conditions.C_Gln_init' in opt_params_dict:
            if opt_params_dict['feed_parameters.C_Gln_feed'] < opt_params_dict['initial_conditions.C_Gln_init']:
                return True
        
        # Growth rate sanity
        if 'kinetic_parameters.mu_max' in opt_params_dict:
            if opt_params_dict['kinetic_parameters.mu_max'] > 0.1 or opt_params_dict['kinetic_parameters.mu_max'] < 0:
                return True
        
        # Volume checks
        if 'reactor.V_max' in opt_params_dict and 'reactor.V_batch_init' in opt_params_dict:
            if opt_params_dict['reactor.V_max'] < opt_params_dict['reactor.V_batch_init']:
                return True

        # Perfusion washout guard — applies when any TBS/TBP dilution rate parameter
        # is being optimized.  Resolve each value from the proposed dict first, falling
        # back to the baseline parameters so partial optimizations are handled correctly.
        _perf = self.baseline_params.perfusion
        _celp = self.baseline_params.cell_properties
        if _perf is not None:
            D_baseline   = opt_params_dict.get('perfusion.D_baseline',     float(_perf.D_baseline))
            D_1_mult     = opt_params_dict.get('perfusion.D_1_multiplier', float(_perf.D_1_multiplier))
            D_2_mult     = opt_params_dict.get('perfusion.D_2_multiplier', float(_perf.D_2_multiplier))
            B            = opt_params_dict.get('perfusion.B',              float(_perf.B))
            eff_sep      = opt_params_dict.get('perfusion.eff_separator',  float(_perf.eff_separator))
            mu_max       = opt_params_dict.get('cell_properties.mu_max',   float(_celp.mu_max))

            D_1 = D_baseline * D_1_mult
            D_2 = D_baseline * D_2_mult

            # Step ordering: D_baseline < D_1 < D_2
            if D_1_mult <= 1.0 or D_2_mult <= D_1_mult:
                return True

            # Perfusion washout guard.
            # In a perfusion reactor with cell retention, dilution only removes cells at
            # rate D*(1-eff_separator). The bleed stream B bypasses the separator entirely.
            # Net cell loss rate = B + D*(1-eff_separator).
            # Washout occurs when net_loss >= mu_max (growth can no longer compensate).
            cell_loss_frac = 1.0 - eff_sep  # fraction of cells passing the separator
            if B >= 0.9 * mu_max:
                return True  # Bleed alone exceeds growth
            if (D_2 * cell_loss_frac + B) >= 0.9 * mu_max:
                return True  # Highest dilution step causes washout

        return False