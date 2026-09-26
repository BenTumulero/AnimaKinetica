"""Pickle backend: loads `SimulationData` pickles via the package public API."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from animakinetica.core.simulation_data import SimulationData
from animakinetica.core.io_utils import load_pickle_object

from .base import BatchView, CapabilityFlags, DashboardModel, DataSource

__all__ = ["PickleSource"]


class PickleSource:
    """Resolve `SimulationData` pickles into `DashboardModel` instances.

    Uses only the package's public API. Accepts both pickle flavours the
    package produces: `SimulationData` (from `save()`/`save_pickle()`) and
    full `CellCultureSimulation` objects (from
    `CellCultureSimulation.save_pickle()`), resolving the latter via its
    public `.data` attribute. The live object is retained on the model so
    pages can delegate all Plotly figure creation to the package.
    """

    source_kind = "pickle"

    def discover(self, search_dir: Path) -> List[Path]:
        """Find `*.pkl` files under `search_dir` (recursive), newest first."""
        if not search_dir.exists():
            return []
        files = [p for p in search_dir.rglob("*.pkl") if p.is_file()]
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def load(self, path: Path) -> DashboardModel:
        obj = load_pickle_object(str(path))
        if hasattr(obj, "data") and hasattr(obj.data, "batches"):
            # Full CellCultureSimulation pickle — resolve to its results.
            data: SimulationData = obj.data
            pickle_object_type = "CellCultureSimulation"
            simulation_object = obj
            # Optimization flag lives only on the full simulation object
            # (set by OptimizationManager); SimulationData pickles cannot
            # report it.
            optimized = bool(getattr(obj, "optimized", False))
        elif hasattr(obj, "batches"):
            data = obj
            pickle_object_type = "SimulationData"
            simulation_object = None
            optimized = None  # Unknown: flag is not serialized on SimulationData
        else:
            raise TypeError(
                f"Pickle '{path.name}' contains a {type(obj).__name__} object, "
                "which is not a SimulationData or CellCultureSimulation result."
            )

        # Optimization results require a full simulation pickle that was
        # optimized AND carries a linked OptimizationManager with results.
        opt_manager = getattr(simulation_object, "optimization_manager", None)
        has_optimization = (
            pickle_object_type == "CellCultureSimulation"
            and optimized
            and opt_manager is not None
            and getattr(opt_manager, "optimal_parameters", None) is not None
        )

        batches: List[BatchView] = []
        for batch in data.batches:
            batches.append(
                BatchView(
                    batch_number=batch.batch_number,
                    t_end=batch.t_end,
                    Xv_final=batch.Xv_final,
                    V_final=batch.V_final,
                    termination_reason=batch.termination_reason,
                    t_points=batch.t_points,
                    y_points=batch.y_points,
                    has_controller_traces=self._has_traces(batch, "N_list", "kLa_list"),
                    has_kinetic_traces=self._has_traces(batch, "mu_list", "NF_list"),
                    has_stoich_yields=self._has_traces(batch, "Y_Glc_X_list"),
                )
            )

        capabilities = CapabilityFlags(
            has_controller_traces=any(b.has_controller_traces for b in batches),
            has_kinetic_traces=any(b.has_kinetic_traces for b in batches),
            has_stoich_yields=any(b.has_stoich_yields for b in batches),
            has_closure=(
                data.parameters is not None
                and any(b.has_kinetic_traces for b in batches)
            ),
            has_process_totals=len(batches) > 0,
            has_batch_state_trajectories=all(
                b.y_points is not None and np.size(b.y_points) > 0 for b in batches
            ) if batches else False,
            has_summary_stats=bool(data.summary_stats),
            has_parameters=data.parameters is not None,
            has_optimization=has_optimization,
        )

        return DashboardModel(
            name=path.stem,
            source_path=path,
            source_kind=self.source_kind,
            capabilities=capabilities,
            metadata={
                "simulation_type": data.simulation_type,
                "feed_strategy": data.feed_strategy,
                "perfusion_strategy": data.perfusion_strategy,
                "timestamp": data.timestamp.isoformat() if data.timestamp else None,
                "n_batches": len(data.batches),
                "pickle_object_type": pickle_object_type,
                "optimized": optimized,
            },
            summary_stats=dict(data.summary_stats),
            batches=batches,
            simulation_data=data,
            simulation_object=simulation_object,
        )

    @staticmethod
    def _has_traces(batch, *field_names: str) -> bool:
        for name in field_names:
            arr = getattr(batch, name, None)
            if arr is not None and np.size(arr) > 0:
                return True
        return False
