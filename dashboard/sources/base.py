"""Data-source abstraction for the AnimaKinetica dashboard app.

This module is app-local glue code — it is deliberately NOT part of the
`animakinetica` package. The dashboard is a pure consumer of the package's
public API. The abstraction normalizes two backends:

- Pickle backend: `SimulationData` pickles (full fidelity, all Plotly
  figures delegated to the package's `analysis` modules).
- Excel backend (future): exported `.xlsx` workbooks (reduced fidelity:
  KPIs, batch summaries, and state trajectories only).

Pages consume a `DashboardModel` and check `CapabilityFlags` to degrade
gracefully when data is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "CapabilityFlags",
    "BatchView",
    "DashboardModel",
    "DataSource",
    "UnavailableCapabilityError",
]


class UnavailableCapabilityError(RuntimeError):
    """Raised when a page requests data the active source cannot provide."""


@dataclass(frozen=True)
class CapabilityFlags:
    """Declare which data facets the active source can provide."""

    has_controller_traces: bool = False  # N/Fg/kLa/OTR/CTR/D_base/y_CO2 ...
    has_kinetic_traces: bool = False     # mu/q_*/NF_*/IF_*/lag ...
    has_stoich_yields: bool = False      # Y_*_list trajectories
    has_closure: bool = False            # CHON closure (needs parameters + traces)
    has_process_totals: bool = False     # cumulative harvest/medium/base/gas plots
    has_batch_state_trajectories: bool = False  # y_points time series
    has_summary_stats: bool = False      # KPI table / metric cards
    has_cultivation_summary: bool = False  # per-batch cultivation table
    has_closure_tables: bool = False       # raw CHON closure sheet sections
    has_parameters: bool = False           # SimulationParameters object available
    has_optimization: bool = False         # optimization results available

    def require(self, capability: str) -> None:
        if not getattr(self, capability, False):
            raise UnavailableCapabilityError(
                f"Capability '{capability}' is not available from the active data source."
            )


@dataclass
class BatchView:
    """Lightweight, source-agnostic view of one logical batch."""

    batch_number: int
    t_end: float
    Xv_final: float
    V_final: float
    termination_reason: str = "Unknown"
    t_points: Optional[Any] = None      # numpy array (h)
    y_points: Optional[Any] = None      # numpy array (n_states, n_points)
    has_controller_traces: bool = False
    has_kinetic_traces: bool = False
    has_stoich_yields: bool = False
    # Excel backend: the per-batch time-series table (long format, one row
    # per time point, labeled columns). Used for Plotly Express line plots.
    time_series_df: Optional[Any] = None


@dataclass
class DashboardModel:
    """Normalized view model consumed by all dashboard pages."""

    name: str                                   # Display name (file stem)
    source_path: Path                           # Original file location
    source_kind: str                            # "pickle" | "excel"
    capabilities: CapabilityFlags = field(default_factory=CapabilityFlags)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    batches: List[BatchView] = field(default_factory=list)
    # Pickle backend only: the live SimulationData object, used for lazy
    # plotting via the package's public API. Excel backend leaves this None.
    simulation_data: Optional[Any] = None
    # Pickle backend only: the live CellCultureSimulation object when the
    # pickle contains a full simulation (needed for optimization results).
    simulation_object: Optional[Any] = None
    # Backend-specific extras (e.g. Excel: cultivation/closure/totals sheets).
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_batches(self) -> int:
        return len(self.batches)

    def get_batch(self, batch_number: int) -> Optional[BatchView]:
        for batch in self.batches:
            if batch.batch_number == batch_number:
                return batch
        return None


@runtime_checkable
class DataSource(Protocol):
    """Interface each backend implements. Pages never import backends directly."""

    source_kind: str

    def discover(self, search_dir: Path) -> List[Path]:
        """Return candidate result files in `search_dir`, alphabetical."""
        ...

    def load(self, path: Path) -> DashboardModel:
        """Load one result file into the normalized DashboardModel."""
        ...
