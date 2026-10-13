"""Excel backend: loads exported `.xlsx` workbooks into a `DashboardModel`.

The Excel export (written by `SimulationData.save(file_format="excel")`)
contains these sheets:

- ``Metadata``: one row with simulation type, strategies, timestamps.
- ``Summary Statistics`` / ``Projected Totals`` / ``Simulated Window Totals``:
  long-format metric tables (Metric / Description / Value / Unit).
- ``Cultivation Summary``: one row per logical batch.
- ``Batch N``: per-batch time-series tables (one row per time point).
- CHON closure sheets: multi-section diagnostic reports.

The Excel backend provides KPIs, cultivation summaries, batch state
trajectories, and the raw closure tables. Controller traces (N, Fg, kLa,
OTR/CTR), kinetic traces (mu, q-rates, NF/IF), and closure *computations*
are not part of the export, so those capabilities are reported as False and
the corresponding dashboard views degrade gracefully.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BatchView, CapabilityFlags, DashboardModel, DataSource

__all__ = ["ExcelSource"]

_SHEET_METADATA = "Metadata"
_SHEET_SUMMARY = "Summary Statistics"
_SHEET_CULTIVATION = "Cultivation Summary"
_SHEET_CLOSURE = (
    "Stoich CHON Closure",
    "Boundary CHON Closure",
    "CHON Closure Delta",
)
_BATCH_SHEET_RE = re.compile(r"^Batch (\d+)$")

# The Excel export writes human-readable metric labels (from METRIC_SPECS),
# not the summary-stat keys used by SimulationData. Map the labels the KPI
# cards need back to their canonical keys so both backends expose the same
# summary_stats dict shape.
_LABEL_TO_KEY = {
    "STY": "STY_kg_m3_day",
    "WCM STY": "STY_WCM_kg_m3_day",
    "Harvested Biomass": "total_DCM_harvested_kg",
    "Harvested Wet Biomass": "total_WCM_harvested_kg",
    "Total Biomass": "total_DCM_kg",
    "Medium Conversion Efficiency": "medium_conversion_efficiency_m3_kg",
    "WCM Medium Conversion Efficiency": "medium_conversion_efficiency_m3_kg_WCM",
    "Protein Conversion Efficiency": "protein_conversion_efficiency_kg_kg",
    "Feed Conversion Efficiency": "feed_conversion_efficiency_kg_kg",
    "WCM Feed Conversion Efficiency": "feed_conversion_efficiency_kg_kg_WCM",
}

# Batch-sheet columns used for the state-trajectory panel.
_TIME_COL = "Time (h)"
_STATE_COLUMNS = [
    "Cell Density (10^6 cells/mL)",
    "Glucose (mmol/L)",
    "Glutamine (mmol/L)",
    "Lactate (mmol/L)",
    "Ammonia (mmol/L)",
    "Volume (L)",
]


class ExcelSource:
    """Load Excel exports into a reduced-fidelity `DashboardModel`."""

    source_kind = "excel"

    def discover(self, search_dir: Path) -> List[Path]:
        """Find `*.xlsx` files under `search_dir` (recursive), alphabetical."""
        if not search_dir.exists():
            return []
        files = [p for p in search_dir.rglob("*.xlsx") if p.is_file()]
        return sorted(files, key=lambda p: str(p).lower())

    def inspect_sheets(self, path: Path) -> List[str]:
        """Return the sheet names of a workbook (helper/debugging)."""
        return list(pd.read_excel(path, sheet_name=None, nrows=0).keys())

    def load(self, path: Path) -> DashboardModel:
        sheets: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None)

        metadata = self._parse_metadata(sheets)
        summary_stats = self._parse_metric_sheet(sheets.get(_SHEET_SUMMARY))
        cultivation = sheets.get(_SHEET_CULTIVATION)
        batches = self._parse_batches(sheets, cultivation)
        # Closure sheets are multi-section reports with mixed-type columns;
        # stringify them so Streamlit's Arrow serialization always succeeds.
        closure_tables = {
            name: sheets[name].astype(str).replace("nan", "")
            for name in _SHEET_CLOSURE if name in sheets
        }

        capabilities = CapabilityFlags(
            has_controller_traces=False,   # not part of the Excel export
            has_kinetic_traces=False,      # not part of the Excel export
            has_stoich_yields=False,       # not part of the Excel export
            has_closure=False,             # closure computation needs the object
            has_process_totals=cultivation is not None and len(cultivation) > 0,
            has_batch_state_trajectories=len(batches) > 0
            and all(b.time_series_df is not None for b in batches),
            has_summary_stats=bool(summary_stats),
            has_cultivation_summary=cultivation is not None and len(cultivation) > 0,
            has_closure_tables=bool(closure_tables),
        )

        return DashboardModel(
            name=path.stem,
            source_path=path,
            source_kind=self.source_kind,
            capabilities=capabilities,
            metadata=metadata,
            summary_stats=summary_stats,
            batches=batches,
            simulation_data=None,  # no live SimulationData from Excel
            # Excel-specific extras carried alongside the standard model:
            extra={
                "cultivation_summary_df": cultivation,
                "closure_tables": closure_tables,
                "projected_totals_df": sheets.get("Projected Totals"),
                "simulated_window_totals_df": sheets.get("Simulated Window Totals"),
            },
        )

    # ------------------------------------------------------------------
    # Sheet parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_metadata(sheets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        df = sheets.get(_SHEET_METADATA)
        if df is None or df.empty:
            return {}
        row = df.iloc[0]

        def _clean(value: Any, default: str = "NA") -> Any:
            """Return the value, or the default when it is NaN/missing."""
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return default
            return value

        return {
            "simulation_type": _clean(row.get("simulation_type"), "—"),
            "feed_strategy": _clean(row.get("feed_strategy")),
            "perfusion_strategy": _clean(row.get("perfusion_strategy")),
            "timestamp": _clean(row.get("simulation_timestamp"), "—"),
            "export_timestamp": _clean(row.get("export_timestamp"), "—"),
        }

    @staticmethod
    def _parse_metric_sheet(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Convert a long-format Metric/Value sheet into a stats dict.

        Keys are canonicalized to `SimulationData.summary_stats` keys via
        `_LABEL_TO_KEY` where a mapping exists; unknown labels are kept
        as-is so nothing is silently dropped.
        """
        if df is None or df.empty or "Metric" not in df.columns:
            return {}
        stats: Dict[str, Any] = {}
        for _, row in df.iterrows():
            value = row.get("Value")
            if pd.isna(value):
                continue
            label = str(row["Metric"])
            key = _LABEL_TO_KEY.get(label, label)
            try:
                stats[key] = float(value)
            except (TypeError, ValueError):
                stats[key] = value
        return stats

    def _parse_batches(
        self,
        sheets: Dict[str, pd.DataFrame],
        cultivation: Optional[pd.DataFrame],
    ) -> List[BatchView]:
        # Map batch number -> cultivation-summary row (if available).
        cult_by_number: Dict[int, pd.Series] = {}
        if cultivation is not None and not cultivation.empty:
            col = "Batch No."
            if col in cultivation.columns:
                for _, row in cultivation.iterrows():
                    try:
                        cult_by_number[int(row[col])] = row
                    except (TypeError, ValueError):
                        continue

        batches: List[BatchView] = []
        for sheet_name, df in sheets.items():
            match = _BATCH_SHEET_RE.match(sheet_name)
            if not match or df is None or df.empty:
                continue
            batch_number = int(match.group(1))
            ts = df[[c for c in _STATE_COLUMNS + [_TIME_COL] if c in df.columns]].copy()
            cult = cult_by_number.get(batch_number)
            batches.append(
                BatchView(
                    batch_number=batch_number,
                    t_end=self._safe_float(ts[_TIME_COL].iloc[-1]) if _TIME_COL in ts else 0.0,
                    Xv_final=self._safe_float(
                        ts["Cell Density (10^6 cells/mL)"].iloc[-1]
                    ) if "Cell Density (10^6 cells/mL)" in ts else 0.0,
                    V_final=self._safe_float(ts["Volume (L)"].iloc[-1])
                    if "Volume (L)" in ts else 0.0,
                    termination_reason=str(cult["Termination Reason"])
                    if cult is not None and "Termination Reason" in cult
                    else "Unknown",
                    time_series_df=ts,
                )
            )
        batches.sort(key=lambda b: b.batch_number)
        return batches

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
