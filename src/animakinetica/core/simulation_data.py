"""Store, report, balance, and export cell-culture simulation results.

``BatchData`` represents one logical batch, including ODE states, optional
controller and kinetic traces, and incremental process-boundary quantities.
``BatchDataAccumulator`` combines sequential ODE phases into one logical
batch. ``SimulationData`` aggregates batches, calculates reports, evaluates
CHON elemental closure, provides plotting shortcuts, and exports results.

Notes
-----
Batch numbers are one-based in public retrieval and plotting methods. Detailed
controller, kinetic, and stoichiometric histories require a simulation run with
``performance_mode=False``. Process-boundary arrays in ``BatchData`` contain
increments at each stored time point; report and plotting methods accumulate
them with :func:`numpy.cumsum` where a cumulative trajectory is required.

Examples
--------
>>> sim.run_simulation()
>>> output = sim.data.save("simulation_results", file_format="excel")
"""

# Imports
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

from animakinetica.models.stoichiometry import _ELEMENTS_CHON, _SPECIES_FORMULAS_CHON
from animakinetica.core.parameters import SimulationParameters, Parameter
from animakinetica.models.events import *
from animakinetica.core.io_utils import (
    load_pickle_object,
    resolve_pickle_load_path,
    resolve_pickle_save_path,
    save_pickle_object,
)

__version__ = "1.0.0"
__all__ = [
    "BatchData",
    "BatchDataAccumulator",
    "SimulationData",
    "EXPORT_DIR",
    "BATCH_TOTAL_ACCUMULATION_MAP",
    "SUMMARY_METRIC_DIRECTIONS",
    "get_script_export_dir",
]

# Global constants
_WATER_MW_G_PER_MOL = 18.01528
_MASS_BALANCE_STOICH_SHEET = 'Stoich CHON Closure'
_MASS_BALANCE_BOUNDARY_SHEET = 'Boundary CHON Closure'
_MASS_BALANCE_DELTA_SHEET = 'CHON Closure Delta'

# Totals-sheet visibility rules. These diagnostics describe the literal
# simulated window and are not scaled during yearly projection, so they are
# intentionally omitted from the projected totals report. Projection metadata
# is instead shown only alongside the simulated-window totals it describes.
_PROJECTED_TOTALS_EXCLUDED_KEYS = {
    'batch_final_volume_max_L',
    'batch_final_volume_max_fraction',
    'batch_final_volume_min_L',
    'batch_final_volume_min_fraction',
    'batch_phase_active_runtime_h',
    'batch_time_range_h',
    'batch_time_max_h',
    'batch_time_min_h',
    'last_batch_final_volume_L',
    'last_batch_final_volume_fraction',
    'last_batch_termination_reason',
    'phase_active_runtime_h',
    'unique_batch_termination_reasons',
}
_SIMULATED_WINDOW_TOTALS_EXCLUDED_KEYS = {
    'projection_mode',
    'projection_scale_factor',
    'projection_target_time_h',
    'projection_window_h',
}


# Explicit metadata for calculated KPIs, biomass accounting, diagnostics, and
# historical compatibility keys. Stream-level process-boundary totals are
# generated from _TRACKED_QUANTITIES below so labels, units, and net-flux
# conventions remain defined in one place.
_EXPLICIT_METRIC_SPECS = {
    'total_biomass_kg': {
        'label': 'Total Biomass',
        'unit': 'kg',
        'description': 'Total dry biomass harvested over the full simulation.',
    },
    'total_DCM_kg': {
        'label': 'Total Biomass',
        'unit': 'kg',
        'description': 'Total dry cell mass harvested over the full simulation.',
    },
    'total_DCM_cmol': {
        'label': 'Total Biomass',
        'unit': 'Cmmol',
        'description': 'Total dry biomass harvested expressed in carbon millimoles.',
    },
    'total_DCM_harvested_kg': {
        'label': 'Harvested Biomass',
        'unit': 'kg',
        'description': 'Productive dry biomass harvested over the full simulation, excluding non-harvested losses.',
    },
    'total_WCM_harvested_kg': {
        'label': 'Harvested Wet Biomass',
        'unit': 'kg WCM',
        'description': 'Productive wet cell mass harvested over the full simulation, excluding non-harvested losses.',
    },
    'total_DCM_harvested_cmmol': {
        'label': 'Harvested Biomass',
        'unit': 'Cmmol',
        'description': 'Productive dry biomass harvested expressed in carbon millimoles, excluding non-harvested losses.',
    },
    'total_DCM_lost_kg': {
        'label': 'Lost Biomass',
        'unit': 'kg',
        'description': 'Non-harvested dry biomass leaving the process boundary (e.g. perfusion separator leak into the permeate).',
    },
    'total_DCM_lost_cmmol': {
        'label': 'Lost Biomass',
        'unit': 'Cmmol',
        'description': 'Non-harvested dry biomass leaving the boundary expressed in carbon millimoles.',
    },
    'STY_kg_m3_day': {
        'label': 'STY',
        'unit': 'kg/m3/day',
        'description': 'Space-time yield based on harvested dry biomass, reactor size, and elapsed time.',
    },
    'STY_WCM_kg_m3_day': {
        'label': 'WCM STY',
        'unit': 'kg WCM/m3/day',
        'description': 'Space-time yield based on harvested wet cell mass, reactor size, and elapsed time.',
    },
    'medium_conversion_efficiency_m3_kg': {
        'label': 'Medium Conversion Efficiency',
        'unit': 'm3/kg DCM',
        'description': 'Total process medium input per harvested dry biomass; lower is better.',
    },
    'medium_conversion_efficiency_m3_kg_WCM': {
        'label': 'WCM Medium Conversion Efficiency',
        'unit': 'm3/kg WCM',
        'description': 'Total process medium input per harvested wet cell mass; lower is better.',
    },
    'total_base_in_L': {
        'label': 'Total Base Addition',
        'unit': 'L',
        'description': 'Total base solution added by the pH controller.',
    },
    'total_acid_in_L': {
        'label': 'Total Acid Addition',
        'unit': 'L',
        'description': 'Total acid solution added by the pH controller.',
    },
    'protein_conversion_efficiency_kg_kg': {
        'label': 'Protein Conversion Efficiency',
        'unit': 'kg Gln+AA/kg protein',
        'description': 'Glutamine plus amino-acid input per harvested biomass protein; lower is better.',
    },
    'feed_conversion_efficiency_kg_kg': {
        'label': 'Feed Conversion Efficiency',
        'unit': 'kg Glc+Gln+AA/kg DCM',
        'description': 'Glucose, glutamine, and amino-acid input per harvested dry biomass; lower is better.',
    },
    'feed_conversion_efficiency_kg_kg_WCM': {
        'label': 'WCM Feed Conversion Efficiency',
        'unit': 'kg Glc+Gln+AA/kg WCM',
        'description': 'Glucose, glutamine, and amino-acid input per harvested wet cell mass; lower is better.',
    },
    'total_nutrient_in_kg': {
        'label': 'Total Nutrients In',
        'unit': 'kg',
        'description': 'Total nutrient mass entering the reactor through initial charge and feed.',
    },
    'total_nutrient_out_kg': {
        'label': 'Total Nutrients Out',
        'unit': 'kg',
        'description': 'Total nutrient mass leaving the reactor in harvest and cell-free outlet streams.',
    },
    'total_nutrient_consumed_kg': {
        'label': 'Total Nutrients Consumed',
        'unit': 'kg',
        'description': 'Net nutrient mass consumed, equal to abs(nutrients out − nutrients in) for substrates.',
    },
    'total_lactate_net_kg': {
        'label': 'Total Lactate Net',
        'unit': 'kg',
        'description': 'Net lactate crossing the process boundary (positive = produced).',
    },
    'total_ammonium_net_kg': {
        'label': 'Total Ammonium Net',
        'unit': 'kg',
        'description': 'Net ammonium crossing the process boundary (positive = produced).',
    },
    'total_CT_net_kg': {
        'label': 'Total Inorganic Carbon Net',
        'unit': 'kg',
        'description': 'Net inorganic carbon produced by metabolism, expressed as CO2 equivalent and including gas stripping.',
    },
    'total_O2_net_kg': {
        'label': 'Total O2 Net',
        'unit': 'kg',
        'description': 'Net oxygen crossing the process boundary (negative = consumed by cells).',
    },
    'total_water_produced_kg': {
        'label': 'Total Water Produced',
        'unit': 'kg',
        'description': 'Total metabolic water produced over the full simulation.',
    },
    'total_heat_generated': {
        'label': 'Total Heat Generated',
        'unit': 'kJ',
        'description': 'Total metabolic heat generated over the full simulation.',
    },
    'total_simulation_time_h': {
        'label': 'Total Cultivation Time',
        'unit': 'h',
        'description': 'Total active modelled cultivation time represented in the simulation.',
    },
    'number_of_batches': {
        'label': 'Number of Batches',
        'unit': '',
        'description': 'Number of completed batches represented in the simulation.',
    },
    'simulated_total_simulation_time_h': {
        'label': 'Simulated Window Duration',
        'unit': 'h',
        'description': 'Actual simulated time window used to project yearly KPIs.',
    },
    'simulated_total_DCM_kg': {
        'label': 'Simulated Window Biomass',
        'unit': 'kg',
        'description': 'Dry cell mass harvested in the literal simulated window before yearly projection.',
    },
    'simulated_number_of_batches': {
        'label': 'Simulated Window Batches',
        'unit': '',
        'description': 'Actual number of batches simulated in the projection window (before yearly scaling).',
    },
    'projection_scale_factor': {
        'label': 'Projection Scale Factor',
        'unit': '',
        'description': 'Multiplier applied to additive totals to scale from the simulation window to the yearly uptime horizon.',
    },
    'projection_mode': {
        'label': 'Projection Mode',
        'unit': '',
        'description': "Projection method used: 'yearly_projection' scales from max_culture_age window to yearly uptime; 'full_simulation' covers the full yearly uptime directly.",
    },
    'projection_window_h': {
        'label': 'Projection Window',
        'unit': 'h',
        'description': 'Elapsed duration of the simulated window used for projection.',
    },
    'projection_target_time_h': {
        'label': 'Projection Target Time',
        'unit': 'h',
        'description': 'Reported horizon after projection, typically the yearly uptime.',
    },
}

_EXPLICIT_METRIC_SPECS.update({
    'total_DCM_in_kg': {
        'label': 'DCM In', 'unit': 'kg',
        'description': 'Dry cell mass entering the external process boundary with fresh inoculum charges.',
    },
    'total_DCM_in_cmmol': {
        'label': 'DCM In', 'unit': 'Cmmol',
        'description': 'Dry cell mass entering the external process boundary, expressed as carbon millimoles.',
    },
    'total_DCM_out_kg': {
        'label': 'DCM Out', 'unit': 'kg',
        'description': 'Total dry cell mass crossing the external process boundary, including harvested and lost biomass.',
    },
    'total_DCM_out_cmmol': {
        'label': 'DCM Out', 'unit': 'Cmmol',
        'description': 'Total dry cell mass crossing the external process boundary, expressed as carbon millimoles.',
    },
    'total_DCM_produced_kg': {
        'label': 'Net DCM Produced', 'unit': 'kg',
        'description': 'Net dry cell mass production across the external process boundary.',
    },
    'total_DCM_produced_cmmol': {
        'label': 'Net DCM Produced', 'unit': 'Cmmol',
        'description': 'Net dry cell mass production across the external process boundary, expressed as carbon millimoles.',
    },
    'total_medium_in_L': {
        'label': 'Total Medium In', 'unit': 'L',
        'description': 'Fresh batch-medium and feed-medium volume entering the process, excluding acid and base solutions.',
    },
    'total_outlet_volume_L': {
        'label': 'Total Outlet Volume', 'unit': 'L',
        'description': 'All liquid volume leaving the external process boundary.',
    },
    'total_permeate_volume_L': {
        'label': 'Total Permeate Volume', 'unit': 'L',
        'description': 'Cell-free permeate volume leaving the process during perfusion.',
    },
    'total_water_produced_mmol': {
        'label': 'Total Water Produced', 'unit': 'mmol',
        'description': 'Metabolic water produced during the simulation.',
    },
    'batch_time_min_h': {
        'label': 'Minimum Batch Time', 'unit': 'h',
        'description': 'Shortest logical-batch cultivation duration in the simulated window.',
    },
    'batch_time_max_h': {
        'label': 'Maximum Batch Time', 'unit': 'h',
        'description': 'Longest logical-batch cultivation duration in the simulated window.',
    },
    'batch_time_range_h': {
        'label': 'Batch Time Range', 'unit': 'h',
        'description': 'Difference between the longest and shortest logical-batch cultivation durations in the simulated window.',
    },
    'last_batch_final_volume_L': {
        'label': 'Last Batch Final Volume', 'unit': 'L',
        'description': 'Final reactor working volume of the last simulated logical batch.',
    },
    'last_batch_final_volume_fraction': {
        'label': 'Last Batch Final Volume Fraction', 'unit': 'fraction of Vmax',
        'description': 'Final working volume of the last simulated logical batch divided by maximum reactor volume.',
    },
    'batch_final_volume_min_L': {
        'label': 'Minimum Final Batch Volume', 'unit': 'L',
        'description': 'Smallest final reactor working volume among simulated logical batches.',
    },
    'batch_final_volume_max_L': {
        'label': 'Maximum Final Batch Volume', 'unit': 'L',
        'description': 'Largest final reactor working volume among simulated logical batches.',
    },
    'batch_final_volume_min_fraction': {
        'label': 'Minimum Final Batch Volume Fraction', 'unit': 'fraction of Vmax',
        'description': 'Smallest final working-volume fraction among simulated logical batches.',
    },
    'batch_final_volume_max_fraction': {
        'label': 'Maximum Final Batch Volume Fraction', 'unit': 'fraction of Vmax',
        'description': 'Largest final working-volume fraction among simulated logical batches.',
    },
    'last_batch_termination_reason': {
        'label': 'Last Batch Termination Reason', 'unit': '',
        'description': 'Termination reason recorded for the final simulated logical batch.',
    },
    'unique_batch_termination_reasons': {
        'label': 'Batch Termination Reasons', 'unit': '',
        'description': 'Unique termination reasons encountered across the simulated logical batches.',
    },
    'phase_active_runtime_h': {
        'label': 'Phase Active Runtime', 'unit': 'h by phase',
        'description': 'Aggregate active cultivation duration for each ODE phase in the simulated window.',
    },
    'batch_phase_active_runtime_h': {
        'label': 'Batch Phase Active Runtime', 'unit': 'h by batch and phase',
        'description': 'Active cultivation duration for every ODE phase in each simulated logical batch.',
    },
})

# Objective orientation for summary-stat KPIs. Metrics absent from this mapping
# default to maximize behavior in the optimization APIs.
SUMMARY_METRIC_DIRECTIONS: Dict[str, str] = {
    'medium_conversion_efficiency_m3_kg': 'minimize',
    'medium_conversion_efficiency_m3_kg_WCM': 'minimize',
    'protein_conversion_efficiency_kg_kg': 'minimize',
    'feed_conversion_efficiency_kg_kg': 'minimize',
    'feed_conversion_efficiency_kg_kg_WCM': 'minimize',
}

# ---------------------------------------------------------------------------
# Central tracked-quantities registry
# ---------------------------------------------------------------------------
# Every process-level cumulative field is defined ONCE here.  All derived
# structures (_PROCESS_CUMULATIVE_FIELDS, _DERIVED_CUMULATIVE_FIELDS,
# _INPUT_CORRECTION_DELTAS, _BATCH_TOTAL_ACCUMULATION_MAP, mass-balance
# component specs, and reporting columns) are generated from this registry
# so that adding a new tracked quantity requires only a single entry.
#
# Key   = BatchData field name for the cumulative "in" array.
# Value = dict with:
#   label           – Human-readable name for reports.
#   unit            – Reporting unit ('mmol', 'L', 'Cmol', 'kJ').
#   mw              – Molecular weight (g/mol) for unit conversion; None if N/A.
#   total_key       – Run-level totals key prefix (e.g. 'total_glucose').
#                     Derived keys: {total_key}_in, {total_key}_out,
#                     {total_key}_net.
#   out_field       – BatchData field for cumulative "out"; None if output-only
#                     or net-only.
#   net_field       – BatchData field for net production(+)/consumption(-);
#                     None if N/A.  Convention: net = out − in (production positive).
#   sign            – Always '-' (net = out - in).  Kept for registry symmetry.
#   inoc_key        – Key in the inoculum dict for carryover correction.
#                     Use '__volume__' for the volume itself (batch_medium_in).
#                     None if no inoculum correction is needed.
# ---------------------------------------------------------------------------
_TRACKED_QUANTITIES: Dict[str, Dict[str, Any]] = {
    # --- Volumes (input-only) ---
    'batch_medium_in': {
        'label': 'Batch Medium', 'unit': 'L', 'mw': None,
        'total_key': None,  # accumulated specially as batch + feed medium
        'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': '__volume__',
    },
    'feed_medium_in': {
        'label': 'Feed Medium', 'unit': 'L', 'mw': None,
        'total_key': None, 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    'base_in': {
        'label': 'Base Addition', 'unit': 'L', 'mw': None,
        'total_key': 'total_base_in', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    'acid_in': {
        'label': 'Acid Addition', 'unit': 'L', 'mw': None,
        'total_key': 'total_acid_in', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    # --- Substrates (in / out / net) ---
    'glucose_in': {
        'label': 'Glucose', 'unit': 'mmol', 'mw': 180.1572,
        'total_key': 'total_glucose', 'out_field': 'glucose_out',
        'net_field': 'glucose_net', 'sign': '-',
        'inoc_key': 'C_Glc_inoc',
    },
    'glutamine_in': {
        'label': 'Glutamine', 'unit': 'mmol', 'mw': 146.1456,
        'total_key': 'total_glutamine', 'out_field': 'glutamine_out',
        'net_field': 'glutamine_net', 'sign': '-',
        'inoc_key': 'C_Gln_inoc',
    },
    'AA_in': {
        'label': 'Amino Acids excl. Gln', 'unit': 'mmol', 'mw': None,
        'total_key': 'total_AA', 'out_field': 'AA_out',
        'net_field': 'AA_net', 'sign': '-',
        'inoc_key': 'C_AA_inoc',
    },
    # --- Products (in / out / net) ---
    'lactate_in': {
        'label': 'Lactate', 'unit': 'mmol', 'mw': 89.07,
        'total_key': 'total_lactate', 'out_field': 'lactate_out',
        'net_field': 'lactate_net', 'sign': '-',
        'inoc_key': 'C_Lac_inoc',
    },
    'ammonium_in': {
        'label': 'Ammonium', 'unit': 'mmol', 'mw': 17.031,
        'total_key': 'total_ammonium', 'out_field': 'ammonium_out',
        'net_field': 'ammonium_net', 'sign': '-',
        'inoc_key': 'C_Amm_inoc',
    },
    # --- Gases (in / out / net) ---
    'oxygen_in': {
        'label': 'O2', 'unit': 'mmol', 'mw': 31.998,
        'total_key': 'total_O2', 'out_field': 'oxygen_out',
        'net_field': 'oxygen_net', 'sign': '-',
        'inoc_key': 'C_O2_inoc',
    },
    'CT_in': {
        'label': 'CO2', 'unit': 'mmol', 'mw': 44.01,
        'total_key': 'total_CT', 'out_field': 'CT_out',
        'net_field': 'CT_net', 'sign': '-',
        'inoc_key': 'C_CT_inoc',
    },
    # --- Volumes (output-only) ---
    'outlet_volume': {
        'label': 'Outlet Volume', 'unit': 'L', 'mw': None,
        'total_key': 'total_outlet_volume', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    'permeate_volume': {
        'label': 'Permeate Volume', 'unit': 'L', 'mw': None,
        'total_key': 'total_permeate_volume', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    'harvest_volume': {
        'label': 'Harvest Volume', 'unit': 'L', 'mw': None,
        'total_key': 'total_harvest_volume', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    # --- Net-only quantities ---
    'water_produced': {
        'label': 'Water Produced', 'unit': 'mmol', 'mw': 18.01528,
        'total_key': 'total_water_produced', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    'heat_generated': {
        'label': 'Heat Generated', 'unit': 'kJ', 'mw': None,
        'total_key': 'total_heat_generated', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
    # --- Biomass (in / out / net / lost) ---
    'cells_in': {
        'label': 'Biomass In', 'unit': 'cells', 'mw': None,
        'total_key': 'total_cells', 'out_field': 'cells_out',
        'net_field': 'cells_net', 'sign': '-',
        'inoc_key': 'Xv_inoc',
    },
    'cells_lost': {
        'label': 'Biomass Lost', 'unit': 'cells', 'mw': None,
        'total_key': 'total_cells_lost', 'out_field': None,
        'net_field': None, 'sign': '+', 'inoc_key': None,
    },
}

# Derive the process cumulative fields tuple from the registry.
# This is the single source of truth for which fields are cumulative.
# Includes all fields: "in", "out", and "net" variants.
_ALL_CUMULATIVE_FIELDS: list = []
for _f, _s in _TRACKED_QUANTITIES.items():
    _ALL_CUMULATIVE_FIELDS.append(_f)
    if _s.get('out_field'):
        _ALL_CUMULATIVE_FIELDS.append(_s['out_field'])
    if _s.get('net_field'):
        _ALL_CUMULATIVE_FIELDS.append(_s['net_field'])
_PROCESS_CUMULATIVE_FIELDS: tuple = tuple(_ALL_CUMULATIVE_FIELDS)

# Derive the net field definitions from the registry.
# Maps net_field -> (out_field, in_field) so that net = out - in
# (production positive, consumption negative).
_DERIVED_CUMULATIVE_FIELDS: Dict[str, tuple] = {}
for _field, _spec in _TRACKED_QUANTITIES.items():
    if _spec.get('net_field') and _spec.get('out_field'):
        _DERIVED_CUMULATIVE_FIELDS[_spec['net_field']] = (_spec['out_field'], _field)

# Derive the inoculum input-correction deltas from the registry.
# Maps field -> callable(inoculum) returning the delta to subtract.
_INPUT_CORRECTION_DELTAS: Dict[str, Any] = {}
for _field, _spec in _TRACKED_QUANTITIES.items():
    _key = _spec.get('inoc_key')
    if _key is None:
        continue
    if _key == '__volume__':
        _INPUT_CORRECTION_DELTAS[_field] = lambda inoc, _k=_key: -float(inoc['V_inoc'])
    else:
        _INPUT_CORRECTION_DELTAS[_field] = (
            lambda inoc, _ik=_key: -float(inoc[_ik]) * float(inoc['V_inoc'])
        )

# Derive the run-level accumulation map from the registry.
# Maps total_key variants -> batch_{field} for all fields with a total_key.
# For fields with out_field/net_field, additional keys are generated
# (e.g. total_glucose_out, total_glucose_net).
def _build_batch_total_accumulation_map() -> Dict[str, str]:
    """Build the batch→run totals accumulation map from the tracked-quantities registry."""
    acc_map: Dict[str, str] = {}
    for field, spec in _TRACKED_QUANTITIES.items():
        tk = spec.get('total_key')
        if tk is None:
            continue
        has_out = spec.get('out_field') is not None
        has_net = spec.get('net_field') is not None
        if has_out or has_net:
            # Field has in/out/net variants — use {total_key}_in suffix
            acc_map[f'{tk}_in'] = f'batch_{field}'
            if has_out:
                acc_map[f'{tk}_out'] = f'batch_{spec["out_field"]}'
            if has_net:
                acc_map[f'{tk}_net'] = f'batch_{spec["net_field"]}'
        else:
            # Field is a single cumulative quantity — use total_key as-is
            acc_map[tk] = f'batch_{field}'
    return acc_map

# Add the medium_in aggregate key (sum of batch + feed) — not derivable
# from a single registry entry because it crosses two fields.
BATCH_TOTAL_ACCUMULATION_MAP = _build_batch_total_accumulation_map()


def _build_tracked_quantity_metric_specs() -> Dict[str, Dict[str, str]]:
    """Generate export metadata for tracked chemical boundary-flow totals.

    The underlying tracked-quantity registry defines each stream's label, unit,
    and in/out/net relationship. Keeping those facts here avoids maintaining a
    separate handwritten description for every chemical flow direction.
    """
    metric_specs: Dict[str, Dict[str, str]] = {}
    for _, spec in _TRACKED_QUANTITIES.items():
        total_key = spec.get('total_key')
        if total_key is None or not spec.get('out_field'):
            continue

        label = str(spec['label'])
        unit = str(spec['unit'])
        metric_specs[f'{total_key}_in_{unit}'] = {
            'label': f'{label} In',
            'unit': unit,
            'description': f'Total {label.lower()} entering the external process boundary.',
        }
        metric_specs[f'{total_key}_out_{unit}'] = {
            'label': f'{label} Out',
            'unit': unit,
            'description': f'Total {label.lower()} leaving the external process boundary.',
        }
        metric_specs[f'{total_key}_net_{unit}'] = {
            'label': f'{label} Net',
            'unit': unit,
            'description': (
                f'Net {label.lower()} crossing the external process boundary '
                '(out minus in; positive indicates net production).'
            ),
        }
    return metric_specs


# Canonical presentation metadata for all current export metrics. The legacy
# name remains as a private alias until all report consumers are migrated.
METRIC_SPECS: Dict[str, Dict[str, str]] = {
    **_EXPLICIT_METRIC_SPECS,
    **_build_tracked_quantity_metric_specs(),
}
_SUMMARY_METRIC_SPECS = METRIC_SPECS

# Add export file paths at module level
EXPORT_DIR = Path(__file__).parent.parent.parent.parent / "export" / "simulations"  # Default package export directory

def get_script_export_dir(subdir: str = "simulation") -> Path:
    """Return a directory below the calling script's directory.

    Parameters
    ----------
    subdir : str, default="simulation"
        Name of the export subdirectory.

    Returns
    -------
    pathlib.Path
        ``<calling-script-directory>/<subdir>``. Returns
        ``Path.cwd() / subdir`` when no caller frame is available.

    Notes
    -----
    This function only resolves a path; it does not create the directory.
    """
    import inspect
    # Get the caller's frame (skip this function and go to actual caller)
    frame = inspect.currentframe()
    try:
        # Go up the stack to find the actual calling script (not this module)
        caller_frame = frame.f_back
        while caller_frame and caller_frame.f_code.co_filename == __file__:
            caller_frame = caller_frame.f_back
        
        if caller_frame:
            script_path = Path(caller_frame.f_code.co_filename).resolve()
            return script_path.parent / subdir
        else:
            # Fallback to current working directory
            return Path.cwd() / subdir
    finally:
        del frame

@dataclass
class BatchData:
    """Store state trajectories and diagnostics for one logical batch.

    Parameters
    ----------
    batch_number : int
        One-based logical-batch identifier.
    t_end : float
        Active cultivation duration of the batch in h.
    Xv_final : float
        Final viable-cell density in cells/L.
    V_final : float
        Final reactor working volume in L.
    t_points : numpy.ndarray
        Stored time points in h.
    y_points : numpy.ndarray
        ODE state trajectories with shape ``(n_states, n_time_points)``.
    initial_conditions : list of Parameter, optional
        Initial-state record used for the batch.
    internal_inoculum_volume_L : float, default=0.0
        Retained broth transferred internally from the preceding batch in L;
        excluded from external fresh-input accounting.
    termination_reason : str, default="Unknown"
        Human-readable reason that integration ended.
    Notes
    -----
    Optional trace arrays are ``None`` when detailed logging is disabled.
    Process-boundary arrays such as ``glucose_in`` and ``harvest_volume`` store
    per-time-point increments, not running totals.
    """
    batch_number: int
    t_end: float
    Xv_final: float
    V_final: float
    t_points: np.ndarray
    y_points: np.ndarray
    initial_conditions: Optional[List[float]] = None
    internal_inoculum_volume_L: float = 0.0  # Internal carryover retained from the previous batch; not an external process input.
    termination_reason: str = "Unknown"  # Reason for simulation termination
    initial_state_rejected: bool = False  # True when phase entry failed a pre-integration safety check.
    D_list: Optional[np.ndarray] = None
    batch_medium_in: Optional[np.ndarray] = None  # Array over time
    feed_medium_in: Optional[np.ndarray] = None  # Array over time
    base_in: Optional[np.ndarray] = None  # Array over time
    acid_in: Optional[np.ndarray] = None  # Array over time
    outlet_volume: Optional[np.ndarray] = None  # Array over time (all liquid leaving reactor)
    permeate_volume: Optional[np.ndarray] = None  # Array over time (cell-free outlet)
    glucose_in: Optional[np.ndarray] = None  # Array over time (cumulative glucose entering)
    glucose_out: Optional[np.ndarray] = None  # Array over time (cumulative glucose leaving reactor)
    glucose_net: Optional[np.ndarray] = None  # Array over time (net glucose: negative = consumed)
    glutamine_in: Optional[np.ndarray] = None  # Array over time (cumulative glutamine entering)
    glutamine_out: Optional[np.ndarray] = None  # Array over time (cumulative glutamine leaving reactor)
    glutamine_net: Optional[np.ndarray] = None  # Array over time (net glutamine: negative = consumed)
    AA_in: Optional[np.ndarray] = None  # Array over time (cumulative amino acid entering)
    AA_out: Optional[np.ndarray] = None  # Array over time (cumulative amino acids leaving reactor)
    AA_net: Optional[np.ndarray] = None  # Array over time (net amino acids: negative = consumed)
    lactate_in: Optional[np.ndarray] = None  # Array over time (cumulative lactate entering with fresh charge)
    lactate_out: Optional[np.ndarray] = None  # Array over time (cumulative lactate leaving reactor)
    lactate_net: Optional[np.ndarray] = None  # Array over time (net lactate: positive = produced)
    ammonium_in: Optional[np.ndarray] = None  # Array over time (cumulative ammonium entering with fresh charge)
    ammonium_out: Optional[np.ndarray] = None  # Array over time (cumulative ammonium leaving reactor)
    ammonium_net: Optional[np.ndarray] = None  # Array over time (net ammonium: positive = produced)
    oxygen_in: Optional[np.ndarray] = None  # Array over time (cumulative oxygen supplied to liquid)
    oxygen_out: Optional[np.ndarray] = None  # Array over time (cumulative dissolved oxygen leaving reactor)
    oxygen_net: Optional[np.ndarray] = None  # Array over time (net oxygen: negative = consumed)
    CT_in: Optional[np.ndarray] = None  # Array over time (cumulative inorganic carbon supplied, CO2 equivalent)
    CT_out: Optional[np.ndarray] = None  # Array over time (cumulative inorganic carbon leaving, CO2 equivalent)
    CT_net: Optional[np.ndarray] = None  # Array over time (net inorganic carbon: positive = produced)
    water_produced: Optional[np.ndarray] = None  # Array over time (net metabolic water produced)
    heat_generated: Optional[np.ndarray] = None  # Array over time (cumulative heat generated, kJ)
    harvest_volume: Optional[np.ndarray] = None  # Array over time
    cells_in: Optional[np.ndarray] = None  # Array over time (biomass entering with fresh charge)
    cells_out: Optional[np.ndarray] = None  # Array over time (total biomass crossing the boundary: harvested + lost)
    cells_net: Optional[np.ndarray] = None  # Array over time (net biomass: positive = produced)
    cells_lost: Optional[np.ndarray] = None  # Array over time (non-harvested biomass leaving the boundary, e.g. perfusion separator leak)
    # Gas controller operating parameters
    N_list: Optional[np.ndarray] = None  # Agitation speed (rpm)
    Fg_list: Optional[np.ndarray] = None  # Gas flow rate (vvh; L/L/h)
    P_over_V_list: Optional[np.ndarray] = None  # Power input per volume (W/m^3)
    v_s_list: Optional[np.ndarray] = None  # Superficial gas velocity (m/s)
    y_O2_list: Optional[np.ndarray] = None  # O2 mole fraction in gas
    kLa_list: Optional[np.ndarray] = None  # Mass transfer coefficient (1/h)
    OTR_list: Optional[np.ndarray] = None  # Oxygen transfer rate (mmol/L/h)
    CTR_list: Optional[np.ndarray] = None  # CO2 transfer rate (mmol/L/h)
    C_CO2_list: Optional[np.ndarray] = None  # CO2 concentration (mmol/L)
    D_base_list: Optional[np.ndarray] = None  # Base dilution rate (1/h)
    D_acid_list: Optional[np.ndarray] = None  # Acid dilution rate (1/h)
    y_CO2_list: Optional[np.ndarray] = None  # CO2 mole fraction in gas
    osm_list: Optional[np.ndarray] = None  # Osmolality (mOsm)
    osm_background_list: Optional[np.ndarray] = None  # Osmolality background state (mOsm)
    # Kinetic model variables
    mu_list: Optional[np.ndarray] = None  # Growth rate (1/h)
    q_Glc_list: Optional[np.ndarray] = None  # Glucose uptake rate (mmol/cell/h)
    q_Gln_list: Optional[np.ndarray] = None  # Glutamine uptake rate (mmol/cell/h)
    q_Lac_list: Optional[np.ndarray] = None  # Lactate production rate (mmol/cell/h)
    q_Amm_list: Optional[np.ndarray] = None  # Ammonia production rate (mmol/cell/h)
    q_O2_list: Optional[np.ndarray] = None  # O2 uptake rate (mmol/cell/h)
    q_CO2_list: Optional[np.ndarray] = None  # CO2 production rate (mmol/cell/h)
    q_AA_list: Optional[np.ndarray] = None  # Amino acid uptake rate (mmol/cell/h)
    q_H2O_list: Optional[np.ndarray] = None  # Metabolic water production rate (mmol/cell/h)
    heat_rate_list: Optional[np.ndarray] = None  # Volumetric heat-generation rate (kJ/L/h)
    lag_term_list: Optional[np.ndarray] = None  # Lag term
    NF_glucose_list: Optional[np.ndarray] = None  # Glucose nutrient factor
    NF_glutamine_list: Optional[np.ndarray] = None  # Glutamine nutrient factor
    NF_O2_list: Optional[np.ndarray] = None  # O2 nutrient factor
    NF_list: Optional[np.ndarray] = None  # Combined nutrient factor
    IF_O2_list: Optional[np.ndarray] = None  # O2 inhibition factor
    IF_CO2_list: Optional[np.ndarray] = None  # CO2 inhibition factor
    IF_lactate_list: Optional[np.ndarray] = None  # Lactate inhibition factor
    IF_ammonia_list: Optional[np.ndarray] = None  # Ammonia inhibition factor
    IF_osmolality_list: Optional[np.ndarray] = None  # Osmolality inhibition factor
    IF_list: Optional[np.ndarray] = None  # Combined inhibition factor
    # Stoichiometric yield coefficient variables
    Y_Glc_X_list: Optional[np.ndarray] = None  # Y_Glc/X [mol Glc / mol DCM]
    Y_Lac_Glc_list: Optional[np.ndarray] = None  # Y_Lac/Glc [mol/mol]
    Y_Gln_Glc_list: Optional[np.ndarray] = None  # Y_Gln/Glc [mol/mol]
    Y_O2_Glc_list: Optional[np.ndarray] = None  # Y_O2/Glc [mol/mol]
    Y_CO2_Glc_list: Optional[np.ndarray] = None  # Y_CO2/Glc [mol/mol]
    Y_AA_Glc_list: Optional[np.ndarray] = None  # Y_AA/Glc [mol/mol]
    Y_Amm_Gln_list: Optional[np.ndarray] = None  # Y_Amm/Gln [mol/mol]
    # Non-carbonate buffer state and H+ flux terms
    C_buffer_list: Optional[np.ndarray] = None  # Buffer concentrations (n_buffers × n_points, mmol/L)
    dC_H_dt_from_buffer_list: Optional[np.ndarray] = None  # H+ rate from buffer re-equilibration (mmol/L/h)
    dC_H_dt_from_Amm_list: Optional[np.ndarray] = None  # H+ rate from ammonia production (mmol/L/h)
    
    
    # Plotting API on Batchdata
    def plot_biomass_metabolites(self, layout_kwargs: Optional[Dict] = None):
        """Plot viable-cell density and metabolite concentrations.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Biomass and metabolite figure for this batch.
        """
        from animakinetica.analysis.batch_visualization import plot_batch_biomass_metabolites
        return plot_batch_biomass_metabolites(self, batch_number=1, layout_kwargs=layout_kwargs)
    
    def plot_volume_feed(self, layout_kwargs: Optional[Dict] = None):
        """Plot working volume, dilution rates, and inlet rates.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Volume and inlet-rate figure for this batch.
        """
        from animakinetica.analysis.batch_visualization import plot_batch_volume_feed
        return plot_batch_volume_feed(self, batch_number=1, layout_kwargs=layout_kwargs)
    
    def plot_gas_transfer(self, layout_kwargs: Optional[Dict] = None):
        """Plot dissolved oxygen and gas-transfer operating variables.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Gas-transfer and hydrodynamics figure for this batch.
        """
        from animakinetica.analysis.gas_visualization import plot_batch_gas_transfer
        return plot_batch_gas_transfer(self, batch_number=1, layout_kwargs=layout_kwargs)
    
    def plot_nutrient_factors(self, layout_kwargs: Optional[Dict] = None):
        """Plot nutrient-limitation factors over time.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Nutrient-factor figure for this batch.
        """
        from animakinetica.analysis.kinetic_visualization import plot_simulated_NFs
        return plot_simulated_NFs(self, batch_number=1, layout_kwargs=layout_kwargs)
    
    def plot_inhibition_factors(self, layout_kwargs: Optional[Dict] = None):
        """Plot inhibition factors over time.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Inhibition-factor figure for this batch.
        """
        from animakinetica.analysis.kinetic_visualization import plot_simulated_IFs
        return plot_simulated_IFs(self, batch_number=1, layout_kwargs=layout_kwargs)
    
    def plot_kinetic_rates(self, layout_kwargs: Optional[Dict] = None):
        """Plot specific growth and uptake/production rates.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Kinetic-rate figure for this batch.
        """
        from animakinetica.analysis.kinetic_visualization import plot_kinetic_rates
        return plot_kinetic_rates(self, batch_number=1, layout_kwargs=layout_kwargs)
    
    def plot_kinetics(self, layout_kwargs: Optional[Dict] = None):
        """Plot nutrient, inhibition, lag, growth, and rate trajectories.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Comprehensive kinetic-state figure for this batch.
        """
        from animakinetica.analysis.kinetic_visualization import plot_kinetics
        return plot_kinetics(self, batch_number=1, layout_kwargs=layout_kwargs)

    def plot_stoichiometric_yields(self, layout_kwargs: Optional[Dict] = None):
        """Plot tracked stoichiometric yield coefficients.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Stoichiometric-yield figure for this batch.
        """
        from animakinetica.analysis.stoichiometric_visualization import plot_stoichiometric_yields
        return plot_stoichiometric_yields(self, batch_number=1, layout_kwargs=layout_kwargs)
    
    def plot_pH_control(self, layout_kwargs: Optional[Dict] = None):
        """Plot pH-control state, acid/base rates, and carbon-dioxide control.

        Parameters
        ----------
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            pH-control figure for this batch.
        """
        from animakinetica.analysis.batch_visualization import plot_batch_pH_control
        return plot_batch_pH_control(self, batch_number=1, layout_kwargs=layout_kwargs)

class BatchDataAccumulator:
    """Combine sequential ODE phases into a single logical batch.

    Use this class for operating modes that change ODE systems during a batch,
    such as a concentrated fed-batch that transitions to perfusion. It shifts
    the time axis of later phases and removes their duplicate phase-boundary
    sample. Process-boundary arrays retain their incremental representation;
    they are concatenated without value offsets.

    Examples
    --------
    >>> accumulator = BatchDataAccumulator()
    >>> for phase_data in phases:
    ...     accumulator.add_phase(phase_data)
    >>> batch = accumulator.build(batch_number=1, init_conds=initial_conditions)
    """

    # Grouped 1-D BatchData fields used for phase concatenation.
    # Derived from the module-level _TRACKED_QUANTITIES registry — the single
    # source of truth for which fields are cumulative across phases.
    _PROCESS_CUMULATIVE_FIELDS = _PROCESS_CUMULATIVE_FIELDS

    _CONTROLLER_TRACE_FIELDS = (
        'D_list', 'N_list', 'Fg_list', 'P_over_V_list', 'v_s_list', 'y_O2_list',
        'kLa_list', 'OTR_list', 'CTR_list', 'C_CO2_list', 'D_base_list', 'D_acid_list', 'y_CO2_list',
        'osm_list', 'osm_background_list',
    )

    _KINETIC_TRACE_FIELDS = (
        'mu_list', 'q_Glc_list', 'q_Gln_list', 'q_Lac_list', 'q_Amm_list',
        'q_O2_list', 'q_CO2_list', 'q_AA_list', 'q_H2O_list', 'heat_rate_list',
        'lag_term_list', 'NF_glucose_list', 'NF_glutamine_list', 'NF_O2_list', 'NF_list',
        'IF_O2_list', 'IF_CO2_list', 'IF_lactate_list', 'IF_ammonia_list',
        'IF_osmolality_list', 'IF_list',
    )

    _STOICH_TRACE_FIELDS = (
        'Y_Glc_X_list', 'Y_Lac_Glc_list', 'Y_Gln_Glc_list', 'Y_O2_Glc_list',
        'Y_CO2_Glc_list', 'Y_AA_Glc_list', 'Y_Amm_Gln_list',
    )

    _BUFFER_TRACE_FIELDS = (
        'dC_H_dt_from_buffer_list', 'dC_H_dt_from_Amm_list',
    )

    # All 1-D array fields from BatchData to concatenate across phases.
    _1D_FIELDS = (
        *_CONTROLLER_TRACE_FIELDS,
        *_PROCESS_CUMULATIVE_FIELDS,
        *_KINETIC_TRACE_FIELDS,
        *_STOICH_TRACE_FIELDS,
        *_BUFFER_TRACE_FIELDS,
    )

    _BATCH_TOTAL_FIELDS = {
        **{f'batch_{field}': field for field in _PROCESS_CUMULATIVE_FIELDS },
    }

    _INPUT_CORRECTION_DELTAS = _INPUT_CORRECTION_DELTAS

    _DERIVED_CUMULATIVE_FIELDS = _DERIVED_CUMULATIVE_FIELDS

    def __init__(self) -> None:
        self._t: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._arrays: dict = {f: None for f in self._1D_FIELDS}
        self._C_buffer: Optional[np.ndarray] = None  # 2D: n_buffers × n_points
        self._last_termination_reason: str = "Unknown"
        self._initial_state_rejected: bool = False
        self._phase_durations_h: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def t_points(self) -> Optional[np.ndarray]:
        """Combined time array across all phases added so far."""
        return self._t

    @property
    def y_points(self) -> Optional[np.ndarray]:
        """Combined state-variable array (shape: n_states × n_points) across all phases."""
        return self._y

    @property
    def phase_durations_h(self) -> dict[str, float]:
        """Accumulated active duration by ODE phase name for the logical batch."""
        return dict(self._phase_durations_h)

    @property
    def initial_state_rejected(self) -> bool:
        """Whether any added phase failed a pre-integration state check."""
        return self._initial_state_rejected

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def add_phase(self, phase_data: 'BatchData', phase_name: Optional[str] = None) -> None:
        """Add completed data for one ODE phase.

        Parameters
        ----------
        phase_data : BatchData
            Result for one integration phase. Later phases are expected to
            begin at the terminal state of the preceding phase.

        Notes
        -----
        The initial sample of every later phase is omitted to avoid duplicate
        timestamps at phase boundaries.
        """
        n = len(phase_data.t_points)
        self._last_termination_reason = phase_data.termination_reason
        self._initial_state_rejected = (
            self._initial_state_rejected or phase_data.initial_state_rejected
        )
        if phase_name is not None:
            self._phase_durations_h[phase_name] = (
                self._phase_durations_h.get(phase_name, 0.0) + float(phase_data.t_end)
            )

        if self._t is None:
            # First phase — store direct copies.
            self._t = phase_data.t_points.copy()
            self._y = phase_data.y_points.copy()
            for field in self._1D_FIELDS:
                val = getattr(phase_data, field, None)
                self._arrays[field] = val.copy() if val is not None else np.zeros(n)
            # 2D buffer concentration array (n_buffers × n_points)
            cb = getattr(phase_data, 'C_buffer_list', None)
            self._C_buffer = cb.copy() if cb is not None and cb.size > 0 else None
        else:
            # Subsequent phases restart solve_ivp at t=0 from the previous phase's
            # terminal state. Drop that duplicated boundary sample so combined
            # plots do not contain two different controller values at the same time.
            slice_start = 1 if n > 0 else 0
            if slice_start >= n:
                return

            # Subsequent phase — offset time for a continuous axis, then concatenate.
            t_offset = self._t[-1]
            self._t = np.concatenate([self._t, phase_data.t_points[slice_start:] + t_offset])
            self._y = np.concatenate([self._y, phase_data.y_points[:, slice_start:]], axis=1)
            for field in self._1D_FIELDS:
                val = getattr(phase_data, field, None)
                arr = val if val is not None else np.zeros(n)
                arr = arr[slice_start:]
                # Incremental arrays: concatenate directly without offset.
                self._arrays[field] = np.concatenate([self._arrays[field], arr])
            # 2D buffer concentration array
            cb = getattr(phase_data, 'C_buffer_list', None)
            if cb is not None and cb.size > 0:
                cb_slice = cb[:, slice_start:]
                if self._C_buffer is not None and self._C_buffer.shape[0] == cb_slice.shape[0]:
                    self._C_buffer = np.concatenate([self._C_buffer, cb_slice], axis=1)

    def build(
        self,
        batch_number: int,
        init_conds: list,
        termination_reason: Optional[str] = None,
        t_end: Optional[float] = None,
    ) -> 'BatchData':
        """Build the combined batch record.

        Parameters
        ----------
        batch_number : int
            One-based batch identifier.
        init_conds : list
            Initial-condition record retained in the resulting ``BatchData``.
        termination_reason : str, optional
            Stop reason for the logical batch. Defaults to the stop reason of
            the most recently added phase.
        t_end : float, optional
            Active batch duration in h. Defaults to the final combined time.

        Returns
        -------
        BatchData
            Combined trajectory and trace record.

        Raises
        ------
        RuntimeError
            If no phase data have been added.
        """
        if self._t is None:
            raise RuntimeError("Cannot build BatchData: no phases have been added.")
        return BatchData(
            batch_number=batch_number,
            t_end=t_end if t_end is not None else float(self._t[-1]),
            Xv_final=float(self._y[0, -1]),
            V_final=float(self._y[5, -1]),
            t_points=self._t,
            y_points=self._y,
            initial_conditions=init_conds,
            termination_reason=termination_reason or self._last_termination_reason,
            initial_state_rejected=self._initial_state_rejected,
            C_buffer_list=self._C_buffer,
            **self._arrays,
        )

    def to_batch_totals(self) -> dict[str, float | str]:
        """Return batch-level totals for run-level aggregation.

        Returns
        -------
        dict of str to float or str
            Batch duration, termination reason, final volume, and sums of each
            incremental process-boundary array.

        Raises
        ------
        RuntimeError
            If no phase data have been added.
        """
        if self._t is None or self._y is None:
            raise RuntimeError("Cannot extract batch totals: no phases have been added.")

        totals: dict[str, float | str] = {
            'batch_time': float(self._t[-1]),
            'batch_termination_reason': self._last_termination_reason,
            'batch_final_volume': float(self._y[5, -1]),
        }
        for key, field in self._BATCH_TOTAL_FIELDS.items():
            totals[key] = self._field_sum(field)
        return totals

    def apply_internal_inoculum_input_correction(self, inoculum: dict) -> None:
        """Exclude retained inoculum from fresh-input values at the batch start.

        Parameters
        ----------
        inoculum : dict
            Inoculum volume and concentrations used to calculate the carryover
            contribution.
        """
        self._assert_has_data()
        for field, delta_func in self._INPUT_CORRECTION_DELTAS.items():
            self._apply_index0_delta(field, delta_func(inoculum), clip_non_negative=True)
        self._recompute_index0_derived_fields()

    def apply_internal_inoculum_output_correction(self, V_inoc: float) -> None:
        """Exclude the retained inoculum volume from external terminal outputs.

        Parameters
        ----------
        V_inoc : float
            Broth volume retained for the next batch in L.
        """
        self._assert_has_data()
        inoculum_volume = float(V_inoc)
        if inoculum_volume <= 0.0:
            return

        y_final = self._y[:, -1]
        self._apply_terminal_delta('outlet_volume', -inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('harvest_volume', -inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('glucose_out', -float(y_final[1]) * inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('glutamine_out', -float(y_final[2]) * inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('AA_out', -float(y_final[10]) * inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('lactate_out', -float(y_final[3]) * inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('ammonium_out', -float(y_final[4]) * inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('oxygen_out', -float(y_final[6]) * inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('CT_out', -float(y_final[7]) * inoculum_volume, clip_non_negative=True)
        self._apply_terminal_delta('cells_out', -float(y_final[0]) * inoculum_volume, clip_non_negative=True)
        self._recompute_terminal_derived_fields()

    def _assert_has_data(self) -> None:
        if self._t is None or self._y is None:
            raise RuntimeError("BatchDataAccumulator contains no phase data.")

    def _field_sum(self, field: str) -> float:
        """Sum all incremental entries for a field (batch total)."""
        arr = self._arrays.get(field)
        if arr is None or len(arr) == 0:
            return 0.0
        return float(np.sum(arr))

    def _terminal_value(self, field: str) -> float:
        arr = self._arrays.get(field)
        if arr is None or len(arr) == 0:
            return 0.0
        return float(arr[-1])

    def _set_terminal_value(self, field: str, value: float, clip_non_negative: bool = False) -> None:
        arr = self._arrays.get(field)
        if arr is None or len(arr) == 0:
            return
        terminal = float(value)
        if clip_non_negative:
            terminal = max(terminal, 0.0)
        arr[-1] = terminal

    def _apply_terminal_delta(self, field: str, delta: float, clip_non_negative: bool = False) -> None:
        current = self._terminal_value(field)
        self._set_terminal_value(field, current + float(delta), clip_non_negative=clip_non_negative)

    def _apply_index0_delta(self, field: str, delta: float, clip_non_negative: bool = False) -> None:
        """Apply a correction delta at index 0 (initial charge entry)."""
        arr = self._arrays.get(field)
        if arr is None or len(arr) == 0:
            return
        new_val = arr[0] + float(delta)
        arr[0] = max(new_val, 0.0) if clip_non_negative else new_val

    def _recompute_terminal_derived_fields(self) -> None:
        for out_field, (lhs_field, rhs_field) in self._DERIVED_CUMULATIVE_FIELDS.items():
            terminal = self._terminal_value(lhs_field) - self._terminal_value(rhs_field)
            self._set_terminal_value(out_field, terminal)

    def _recompute_index0_derived_fields(self) -> None:
        """Recompute derived net fields at index 0 after input correction."""
        for out_field, (lhs_field, rhs_field) in self._DERIVED_CUMULATIVE_FIELDS.items():
            lhs_arr = self._arrays.get(lhs_field)
            rhs_arr = self._arrays.get(rhs_field)
            out_arr = self._arrays.get(out_field)
            if lhs_arr is not None and rhs_arr is not None and out_arr is not None and len(lhs_arr) > 0:
                out_arr[0] = lhs_arr[0] - rhs_arr[0]

@dataclass
class SimulationData:
    """Aggregate batches, metrics, closure reports, and exported representations.

    Parameters
    ----------
    batches : list of BatchData, optional
        Logical-batch records. Defaults to an empty list.
    summary_stats : dict, optional
        Primary reported KPIs and summary metrics.
    projected_totals_stats : dict, optional
        Totals projected from the simulated window to the target uptime.
    simulated_window_totals_stats : dict, optional
        Unprojected totals accumulated in the literal simulated time window.
    simulation_type : str, default=""
        Human-readable simulation configuration label.
    feed_strategy : str, default="NA"
        Selected liquid-feed strategy, or ``"NA"`` when no feed strategy is
        configured for the simulation.
    perfusion_strategy : str, default="NA"
        Selected perfusion strategy, or ``"NA"`` when no perfusion strategy
        is configured for the simulation.
    timestamp : datetime, optional
        Timestamp assigned when this instance is created.
    parameters : SimulationParameters, optional
        Parameter set used to generate the results.

    Notes
    -----
    Use :meth:`get_batch` for one-based batch retrieval. CHON closure reports
    need parameters and stored detailed kinetic trajectories; the latter are
    unavailable for performance-mode simulations.
    """
    batches: List[BatchData] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    projected_totals_stats: Dict[str, Any] = field(default_factory=dict)
    simulated_window_totals_stats: Dict[str, Any] = field(default_factory=dict)
    simulation_type: str = "" # Used for printing
    feed_strategy: str = "NA"
    perfusion_strategy: str = "NA"
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: SimulationParameters = None

    # Maps closure 'basis' keys to Excel report section titles.
    # Used by _build_closure_sheet_sections and _build_delta_sheet_sections
    # to derive section headings from the closure result itself.
    _CLOSURE_TITLES: Dict[str, tuple] = field(default_factory=lambda: {
        'integrated_stoichiometric_process_chon_closure': (
            'Integrated stoichiometric CHON closure summary',
            'Integrated stoichiometric component basis',
            'Configured CHON formulas used in the stoichiometric closure',
        ),
        'external_process_boundary_chon_closure': (
            'Whole-process external-boundary CHON closure summary',
            'External-boundary component basis',
            'Configured CHON formulas used in the external-boundary closure',
        ),
        'chon_closure_delta': (
            'CHON closure delta summary (boundary residual − stoichiometric residual)',
            'Per-component delta basis',
            'Configured CHON formulas used in the delta report',
        ),
    }, repr=False)

    def get_batch(self, batch_number: int) -> Optional[BatchData]:
        """Return a one-based batch record.

        Parameters
        ----------
        batch_number : int
            One-based batch identifier.

        Returns
        -------
        BatchData or None
            Matching batch record, or ``None`` when it is absent.
        """
        for batch in self.batches:
            if batch.batch_number == batch_number:
                return batch
        return None

### Export and import methods ###
    def save(self, 
            filename: str,
            add_timestamp: bool = True,
            file_format: str = 'excel',
            include_metadata: bool = True,
            include_summary_stats: bool = True,
            include_totals_stats: bool = True,
            include_mass_balance: bool = True,
            include_cultivation_summary: bool = True,
            include_batch_data: bool = True,
            include_parameters: bool = True,
            include_pickle: bool = True,
            overwrite: bool = False,
            export_dir: Optional[Path] = None,
            file_stem: Optional[str] = None) -> Path:
        """Export simulation data and optional companion files.

        Parameters
        ----------
        filename : str
            Base output name without a file extension.
        add_timestamp : bool, default=True
            Append the simulation timestamp to the output name.
        file_format : {'excel', 'csv'}, default='excel'
            Main export format. Excel produces one workbook; CSV produces a
            directory of tables.
        include_metadata, include_summary_stats, include_totals_stats : bool, optional
            Include the corresponding report tables.
        include_mass_balance : bool, default=True
            Include CHON closure sheets in Excel output when result statistics
            and parameters are available. This option does not add CSV files.
        include_cultivation_summary, include_batch_data : bool, optional
            Include the batch summary and detailed batch tables, respectively.
        include_parameters : bool, default=True
            Write the parameter set beside the main export when available.
        include_pickle : bool, default=True
            Serialize this complete ``SimulationData`` object beside the export.
        overwrite : bool, default=False
            Replace existing companion files where supported.
        export_dir : pathlib.Path, optional
            Base export directory. Defaults to :data:`EXPORT_DIR`.
        file_stem : str, optional
            Stem for Excel, parameter, and pickle companion files. When
            omitted, the timestamped export-directory name is also used as
            the file stem for backward compatibility.

        Returns
        -------
        pathlib.Path
            Workbook path for Excel output or output-directory path for CSV.

        Raises
        ------
        FileExistsError
            If the Excel workbook exists and ``overwrite`` is ``False``.
        ValueError
            If ``file_format`` is unsupported or no report content is selected.

        Examples
        --------
        >>> output = sim.data.save("my_results")
        >>> output = sim.data.save(
        ...     "my_results", export_dir=get_script_export_dir()
        ... )
        """
        # Create export directory if it doesn't exist
        export_base = export_dir if export_dir is not None else EXPORT_DIR
        export_base.mkdir(parents=True, exist_ok=True)
        # Add a timestamped subdirectory for all exports with this filename.
        # File artifacts may use a shorter independent stem.
        if add_timestamp:
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M")
            filename = f"{filename}_{timestamp_str}"       
        export_dir = export_base / filename
        export_dir.mkdir(parents=True, exist_ok=True)
        output_stem = file_stem if file_stem is not None else filename
            
        # Prepare timestamp for export
        export_timestamp = datetime.now()
        
        if file_format.lower() == 'excel':
            filepath = export_dir / f"{output_stem}.xlsx"
            
            # Check if file exists
            if filepath.exists() and not overwrite:
                raise FileExistsError(f"File {filepath} already exists. Set overwrite=True to replace it.")
            
            with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
                sheet_count = 0
                
                # 1. Metadata sheet
                if include_metadata:
                    metadata_df = self.get_metadata_df(verbose=False)
                    # Add export timestamp
                    metadata_df['export_timestamp'] = export_timestamp.isoformat()
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    sheet_count += 1
                
                # 2. Summary statistics sheet
                if include_summary_stats:
                    summary_df = self.get_summary_stats_df(verbose=False)
                    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                    sheet_count += 1

                # 3. Totals sheets
                if include_totals_stats:
                    projected_totals_df = self.get_totals_stats_df(source='projected', verbose=False)
                    projected_totals_df.to_excel(writer, sheet_name='Projected Totals', index=False)
                    simulated_totals_df = self.get_totals_stats_df(source='simulated_window', verbose=False)
                    simulated_totals_df.to_excel(writer, sheet_name='Simulated Window Totals', index=False)
                    sheet_count += 2

                # 4. Mass balance reporting sheets
                if include_mass_balance and self.summary_stats and self.parameters is not None:
                    self._write_closure_sheets(writer)
                    sheet_count += 3
                
                # 5. Cultivation summary sheet
                if include_cultivation_summary:
                    cultivation_df = self.get_cultivation_summary_df(verbose=False)
                    cultivation_df.to_excel(writer, sheet_name='Cultivation Summary', index=False)
                    sheet_count += 1
                
                # 6. Individual batch data sheets
                if include_batch_data:
                    for batch in self.batches:
                        batch_df = self.get_batch_df(batch.batch_number, verbose=False)
                        sheet_name = f'Batch {batch.batch_number}'
                        batch_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheet_count += 1
                
                if sheet_count == 0:
                    raise ValueError("No data selected for export. Enable at least one export option.")
            
            print(f"Simulation data exported to: {filepath}")
            
        elif file_format.lower() == 'csv':
            file_count = 0
            # 1. Metadata CSV
            if include_metadata:
                metadata_df = self.get_metadata_df(verbose=False)
                metadata_df['export_timestamp'] = export_timestamp.isoformat()
                metadata_path = export_dir / '01_metadata.csv'
                metadata_df.to_csv(metadata_path, index=False)
                file_count += 1
            
            # 2. Summary statistics CSV
            if include_summary_stats:
                summary_df = self.get_summary_stats_df(verbose=False)
                summary_path = export_dir / '02_summary_statistics.csv'
                summary_df.to_csv(summary_path, index=False)
                file_count += 1

            # 3. Totals CSVs
            if include_totals_stats:
                projected_totals_df = self.get_totals_stats_df(source='projected', verbose=False)
                projected_totals_path = export_dir / '03_projected_totals.csv'
                projected_totals_df.to_csv(projected_totals_path, index=False)
                simulated_totals_df = self.get_totals_stats_df(source='simulated_window', verbose=False)
                simulated_totals_path = export_dir / '04_simulated_window_totals.csv'
                simulated_totals_df.to_csv(simulated_totals_path, index=False)
                file_count += 2
            
            # 4. Cultivation summary CSV
            if include_cultivation_summary:
                cultivation_df = self.get_cultivation_summary_df(verbose=False)
                cultivation_path = export_dir / '05_cultivation_summary.csv'
                cultivation_df.to_csv(cultivation_path, index=False)
                file_count += 1
            
            # 5. Individual batch data CSVs
            if include_batch_data:
                for batch in self.batches:
                    batch_df = self.get_batch_df(batch.batch_number, verbose=False)
                    batch_path = export_dir / f'batch_{batch.batch_number:02d}.csv'
                    batch_df.to_csv(batch_path, index=False)
                    file_count += 1
            
            if file_count == 0:
                raise ValueError("No data selected for export. Enable at least one export option.")
            
            filepath = export_dir
            print(f"Simulation data exported to: {export_dir} ({file_count} files)")
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}. Use 'excel' or 'csv'.")
        
        # Save parameters if requested
        if include_parameters and self.parameters is not None:
            param_filepath = filepath.with_suffix('.parameters') # yaml extension added by ParameterIO         
            self.parameters.save_parameters(param_filepath, overwrite=overwrite)
        
        # Save pickle if requested
        if include_pickle:
            pickle_filepath = export_dir / f"{output_stem}.pkl"
            self.save_pickle(pickle_filepath, overwrite=overwrite)
        
        return filepath

    def save_pickle(self, filename: str, overwrite: bool = False, base_export_dir: Optional[Path] = None) -> Path:
        """Serialize the complete result object to a pickle file.

        Parameters
        ----------
        filename : str
            File name or path, with or without the ``.pkl`` suffix.
        overwrite : bool, default=False
            Replace an existing pickle file.
        base_export_dir : pathlib.Path, optional
            Root used for bare filenames. Defaults to :data:`EXPORT_DIR`.

        Returns
        -------
        pathlib.Path
            Resolved path of the saved pickle file.

        Notes
        -----
        Pickle preserves arrays, parameters, and nested result objects. Load
        only files from trusted sources.
        """
        # Determine export directory
        export_base = base_export_dir if base_export_dir is not None else EXPORT_DIR
        filepath = resolve_pickle_save_path(filename, export_base)

        # Save the object
        filepath = save_pickle_object(self, filepath, overwrite=overwrite)
        print(f"SimulationData object saved to: {filepath}")
        return filepath

    @classmethod
    def load_pickle(cls, filename: Union[str, Path]) -> 'SimulationData':
        """Load a serialized ``SimulationData`` object.

        Parameters
        ----------
        filename : str or pathlib.Path
            Pickle path or bare file name, with or without ``.pkl``. Bare names
            are searched recursively below :data:`EXPORT_DIR`.

        Returns
        -------
        SimulationData
            Reconstructed results, parameters, and associated batch data.

        Raises
        ------
        FileNotFoundError
            If no matching pickle file can be resolved.
        TypeError
            If the file does not contain a ``SimulationData`` instance.

        Notes
        -----
        Load only pickle files from trusted sources.
        """
        filepath = resolve_pickle_load_path(filename, EXPORT_DIR)
        data = load_pickle_object(filepath, expected_type=cls)
        
        print(f"SimulationData object loaded from: {filepath}")
        print(f"  - Simulation type: {data.simulation_type}")
        print(f"  - Number of batches: {len(data.batches)}")
        print(f"  - Timestamp: {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return data

### Helper methods to create DataFrames for export and analysis ###
    @staticmethod
    def _build_status_df(message: str) -> pd.DataFrame:
        """Return a one-row status table for reports that cannot be generated."""
        return pd.DataFrame([
            {
                'Status': 'Unavailable',
                'Message': message,
            }
        ])

    def get_metadata_df(self, verbose: bool = False) -> pd.DataFrame:
        """Return core simulation metadata as a one-row table.

        Parameters
        ----------
        verbose : bool, default=False
            Print the table when ``True``.

        Returns
        -------
        pandas.DataFrame
            Timestamp, simulation type, and configured feed/perfusion strategies.
        """
        metadata = {
            'simulation_timestamp': self.timestamp.isoformat(),
            'simulation_type': self.simulation_type,
            'feed_strategy': getattr(self, 'feed_strategy', 'NA'),
            'perfusion_strategy': getattr(self, 'perfusion_strategy', 'NA'),
        }

        df = pd.DataFrame([metadata])

        if verbose:
            print("\n=== METADATA ===")
            print(df.to_string(index=False))

        return df

    def get_summary_stats_df(self, verbose: bool = False) -> pd.DataFrame:
        """Return reported summary metrics as a labeled table.

        Parameters
        ----------
        verbose : bool, default=False
            Print the table when ``True``.

        Returns
        -------
        pandas.DataFrame
            Metric labels, descriptions, values, and units.

        Raises
        ------
        ValueError
            If summary statistics are unavailable.
        """
        if not self.summary_stats:
            raise ValueError("No summary statistics available. Run a simulation first.")
        df = self._build_metric_report_df(self.summary_stats, preserve_registry_order=True)
        
        if verbose:
            print("\n=== SUMMARY STATISTICS ===")
            print(df.to_string(index=False))
        
        return df

    @staticmethod
    def _build_metric_report_df(
        stats: Dict[str, Any],
        *,
        excluded_keys: Optional[set[str]] = None,
        preserve_registry_order: bool = False,
    ) -> pd.DataFrame:
        """Build a labeled metric table from a statistics mapping.

        Current simulation-generated metric keys resolve through
        :data:`METRIC_SPECS`, which guarantees a label, description, and unit.
        Unknown keys remain exportable for old pickles and user-extended
        statistics dictionaries with an explicit compatibility fallback.
        """
        excluded = excluded_keys or set()
        visible_keys = [key for key in stats if key not in excluded]
        if preserve_registry_order:
            ordered_keys = [key for key in METRIC_SPECS if key in visible_keys]
            ordered_keys.extend(sorted(key for key in visible_keys if key not in METRIC_SPECS))
        else:
            ordered_keys = sorted(visible_keys)

        rows = []
        for key in ordered_keys:
            spec = METRIC_SPECS.get(key)
            if spec is None:
                spec = {
                    'label': key,
                    'description': 'Unregistered legacy or user-defined metric.',
                    'unit': '',
                }
            rows.append({
                'Metric': spec['label'],
                'Description': spec['description'],
                'Value': stats[key],
                'Unit': spec['unit'],
            })
        return pd.DataFrame(rows, columns=['Metric', 'Description', 'Value', 'Unit'])

    def get_totals_stats_df(self, source: str = 'projected', verbose: bool = False) -> pd.DataFrame:
        """Return projected or literal-window totals as a labeled table.

        Parameters
        ----------
        source : {'projected', 'simulated_window'}, default='projected'
            Totals collection to report.
        verbose : bool, default=False
            Print the table when ``True``.

        Returns
        -------
        pandas.DataFrame
            Metric labels, descriptions, values, and units.

        Raises
        ------
        ValueError
            If ``source`` is invalid or the requested totals are unavailable.
        """
        if source == 'projected':
            totals_stats = self.projected_totals_stats
            excluded_keys = _PROJECTED_TOTALS_EXCLUDED_KEYS
        elif source == 'simulated_window':
            totals_stats = self.simulated_window_totals_stats
            excluded_keys = _SIMULATED_WINDOW_TOTALS_EXCLUDED_KEYS
        else:
            raise ValueError("source must be 'projected' or 'simulated_window'")

        if not totals_stats:
            raise ValueError(f"No totals statistics available for source '{source}'. Run a simulation first.")
        df = self._build_metric_report_df(totals_stats, excluded_keys=excluded_keys)

        if verbose:
            print(f"\n=== {source.upper()} TOTALS ===")
            print(df.to_string(index=False))

        return df

    def get_batch_df(self, batch_number: int, verbose: bool = False) -> pd.DataFrame:
        """Return detailed time-series data for one batch.

        Parameters
        ----------
        batch_number : int
            One-based batch identifier.
        verbose : bool, default=False
            Print the table when ``True``.

        Returns
        -------
        pandas.DataFrame
            State trajectories, available controller and yield traces, and
            cumulative versions of stored incremental boundary quantities.

        Raises
        ------
        ValueError
            If the requested batch is absent.
        """
        batch = self.get_batch(batch_number)
        if batch is None:
            raise ValueError(f"Batch {batch_number} not found in simulation data")
        
        # Create DataFrame with time-series data
        data = {
            'Time (h)': batch.t_points,
            'Cell Density (10^6 cells/mL)': batch.y_points[0] / 1e9,
            'Glucose (mmol/L)': batch.y_points[1],
            'Glutamine (mmol/L)': batch.y_points[2],
            'Lactate (mmol/L)': batch.y_points[3],
            'Ammonia (mmol/L)': batch.y_points[4],
            'Volume (L)': batch.y_points[5]
        }
        
        # Add dilution rate if available
        if batch.D_list is not None:
            data['Dilution Rate (1/h)'] = batch.D_list
        if batch.D_base_list is not None:
            data['Base Dilution Rate (1/h)'] = batch.D_base_list
            if batch.D_list is not None:
                data['Total Inlet Dilution Rate (1/h)'] = batch.D_list + batch.D_base_list
        if batch.D_acid_list is not None:
            data['Acid Dilution Rate (1/h)'] = batch.D_acid_list
        # Cumulative process arrays are now stored as incremental per-timestep;
        # wrap in np.cumsum() to export the cumulative time-series.
        if batch.feed_medium_in is not None:
            data['Feed Medium Use (L)'] = np.cumsum(batch.feed_medium_in)
        if batch.base_in is not None:
            data['Base Addition Use (L)'] = np.cumsum(batch.base_in)
        if batch.acid_in is not None:
            data['Acid Addition Use (L)'] = np.cumsum(batch.acid_in)
        if batch.outlet_volume is not None:
            data['Total Outlet Volume (L)'] = np.cumsum(batch.outlet_volume)
        if batch.permeate_volume is not None:
            data['Permeate Volume (L)'] = np.cumsum(batch.permeate_volume)
        if batch.glucose_out is not None:
            data['Glucose Out (mmol)'] = np.cumsum(batch.glucose_out)
        if batch.glutamine_out is not None:
            data['Glutamine Out (mmol)'] = np.cumsum(batch.glutamine_out)
        if batch.AA_out is not None:
            data['AA Out (mmol)'] = np.cumsum(batch.AA_out)
        if batch.glucose_net is not None:
            data['Glucose Net (mmol)'] = np.cumsum(batch.glucose_net)
        if batch.glutamine_net is not None:
            data['Glutamine Net (mmol)'] = np.cumsum(batch.glutamine_net)
        if batch.AA_net is not None:
            data['AA Net (mmol)'] = np.cumsum(batch.AA_net)
        if batch.lactate_out is not None:
            data['Lactate Out (mmol)'] = np.cumsum(batch.lactate_out)
        if batch.ammonium_out is not None:
            data['Ammonium Out (mmol)'] = np.cumsum(batch.ammonium_out)
        if batch.oxygen_in is not None:
            data['O2 In (mmol)'] = np.cumsum(batch.oxygen_in)
        if batch.oxygen_out is not None:
            data['O2 Out (mmol)'] = np.cumsum(batch.oxygen_out)
        if batch.CT_in is not None:
            data['CT In (mmol)'] = np.cumsum(batch.CT_in)
        if batch.CT_out is not None:
            data['CT Out (mmol)'] = np.cumsum(batch.CT_out)
        if batch.lactate_net is not None:
            data['Lactate Net (mmol)'] = np.cumsum(batch.lactate_net)
        if batch.ammonium_net is not None:
            data['Ammonium Net (mmol)'] = np.cumsum(batch.ammonium_net)
        if batch.oxygen_net is not None:
            data['O2 Net (mmol)'] = np.cumsum(batch.oxygen_net)
        if batch.CT_net is not None:
            data['CT Net (mmol)'] = np.cumsum(batch.CT_net)
        if batch.water_produced is not None:
            data['Water Produced (mmol)'] = np.cumsum(batch.water_produced)
        if batch.heat_generated is not None:
            data['Heat Generated (kJ)'] = np.cumsum(batch.heat_generated)
        if batch.P_over_V_list is not None:
            data['Power/Volume (W/m^3)'] = batch.P_over_V_list
        if batch.v_s_list is not None:
            data['Superficial Gas Velocity (m/s)'] = batch.v_s_list
        if batch.Y_Glc_X_list is not None:
            data['Y_Glc/X (mol/mol)'] = batch.Y_Glc_X_list
        if batch.Y_Lac_Glc_list is not None:
            data['Y_Lac/Glc (mol/mol)'] = batch.Y_Lac_Glc_list
        if batch.Y_Gln_Glc_list is not None:
            data['Y_Gln/Glc (mol/mol)'] = batch.Y_Gln_Glc_list
        if batch.Y_O2_Glc_list is not None:
            data['Y_O2/Glc (mol/mol)'] = batch.Y_O2_Glc_list
        if batch.Y_CO2_Glc_list is not None:
            data['Y_CO2/Glc (mol/mol)'] = batch.Y_CO2_Glc_list
        if batch.Y_AA_Glc_list is not None:
            data['Y_AA/Glc (mol/mol)'] = batch.Y_AA_Glc_list
        if batch.Y_Amm_Gln_list is not None:
            data['Y_Amm/Gln (mol/mol)'] = batch.Y_Amm_Gln_list
        if batch.q_H2O_list is not None:
            data['q_H2O (mmol/cell/h)'] = batch.q_H2O_list
        if batch.heat_rate_list is not None:
            data['Heat Rate (kJ/L/h)'] = batch.heat_rate_list
        
        df = pd.DataFrame(data)
        
        if verbose:
            print(f"\n=== BATCH {batch_number} DATA ===")
            print(df.to_string(index=False))
        
        return df

    def get_cultivation_summary_df(self, verbose: bool = False) -> pd.DataFrame:
        """Return a batch-level cultivation and material-use summary.

        Parameters
        ----------
        verbose : bool, default=False
            Print the table when ``True``.

        Returns
        -------
        pandas.DataFrame
            Per-batch timing, final concentrations, medium use, volume closure,
            and harvested biomass metrics.

        Raises
        ------
        ValueError
            If batch data or the simulation parameters are unavailable.
        """
        if not self.batches:
            raise ValueError("No batch data available. Run a simulation first.")
        
        if self.parameters is None:
            raise ValueError("Parameters not available in simulation data")
        
        # Extract parameters
        dry_cell_mass = self.parameters.cell_properties.dry_cell_mass
        batches = self.batches
        
        cultivation_time_h = np.asarray([batch.t_end for batch in batches], dtype=float)
        start_time_h = np.concatenate(([0.0], np.cumsum(cultivation_time_h[:-1])))
        end_time_h = start_time_h + cultivation_time_h

        batch_medium_in = np.asarray([np.sum(batch.batch_medium_in) for batch in batches], dtype=float)
        feed_medium_in = np.asarray([np.sum(batch.feed_medium_in) for batch in batches], dtype=float)
        base_addition_use = np.asarray([
            np.sum(batch.base_in) if batch.base_in is not None else 0.0
            for batch in batches
        ], dtype=float)
        acid_addition_use = np.asarray([
            np.sum(batch.acid_in) if batch.acid_in is not None else 0.0
            for batch in batches
        ], dtype=float)
        glucose_added = np.asarray([np.sum(batch.glucose_in) for batch in batches], dtype=float)
        glutamine_added = np.asarray([np.sum(batch.glutamine_in) for batch in batches], dtype=float)
        AA_added = np.asarray([np.sum(batch.AA_in) for batch in batches], dtype=float)
        total_medium_use = batch_medium_in + feed_medium_in
        total_inlet_use = total_medium_use + base_addition_use + acid_addition_use
        total_outlet_volume = np.asarray([
            np.sum(batch.outlet_volume) if batch.outlet_volume is not None else np.nan
            for batch in batches
        ], dtype=float)
        volume_closure_residual = total_inlet_use - total_outlet_volume
        harvest_volume = np.asarray([np.sum(batch.harvest_volume) for batch in batches], dtype=float)
        # cells_out is the total biomass crossing the boundary; the harvested fraction
        # excludes non-productive losses (e.g. perfusion separator leak).
        cells_out_total = np.asarray([np.sum(batch.cells_out) for batch in batches], dtype=float)
        cells_lost = np.asarray([
            np.sum(batch.cells_lost) if batch.cells_lost is not None else 0.0
            for batch in batches
        ], dtype=float)
        cells_harvested = cells_out_total - cells_lost

        accum_cultivation_time_d = end_time_h / 24

        data = {
            "Batch No.": [batch.batch_number for batch in batches],
            "Termination Reason": [batch.termination_reason for batch in batches],
            "Cultivation Time (h)": cultivation_time_h,
            "Batch Start Time (h)": start_time_h,
            "Batch End Time (h)": end_time_h,
            "Accum. Cultivation Time (d)": accum_cultivation_time_d,
            "Final Glucose (mmol/L)": [batch.y_points[1, -1] for batch in batches],
            "Final Glutamine (mmol/L)": [batch.y_points[2, -1] for batch in batches],
            "Final Lactate (mmol/L)": [batch.y_points[3, -1] for batch in batches],
            "Final Ammonia (mmol/L)": [batch.y_points[4, -1] for batch in batches],
            "Final Operating Volume (L)": [batch.V_final for batch in batches],
            "Batch Medium Use (L)": batch_medium_in,
            "Feed Medium Use (L)": feed_medium_in,
            "Base Addition Use (L)": base_addition_use,
            "Acid Addition Use (L)": acid_addition_use,
            "Total Medium Use (L)": total_medium_use,
            "Accum. Medium Use (L)": np.cumsum(total_medium_use),
            "Total Outlet Volume (L)": total_outlet_volume,
            "Accum. Total Outlet Volume (L)": np.nancumsum(total_outlet_volume),
            "Volume Closure Residual (L)": volume_closure_residual,
            "Glucose Added (mmol)": glucose_added,
            "Accum. Glucose Added (mmol)": np.cumsum(glucose_added),
            "Glutamine Added (mmol)": glutamine_added,
            "Accum. Glutamine Added (mmol)": np.cumsum(glutamine_added),
            "AA Added (mmol)": AA_added,
            "Accum. AA Added (mmol)": np.cumsum(AA_added),
            "Total Inlet Use (L)": total_inlet_use,
            "Accum. Base Addition Use (L)": np.cumsum(base_addition_use),
            "Accum. Acid Addition Use (L)": np.cumsum(acid_addition_use),
            "Accum. Total Inlet Use (L)": np.cumsum(total_inlet_use),
            "Harvest Volume (L)": harvest_volume,
            "Accum. Harvest Volume (L)": np.cumsum(harvest_volume),
            "Final Cell Density (10^6 cells/mL)": [batch.Xv_final / 1e9 for batch in batches],
            "Final Cell Density (g/L)": [batch.Xv_final * dry_cell_mass for batch in batches],
            "Cells Harvested (10^12 Cells)": cells_harvested / 1e12,
            "Cells Harvested (kg)": cells_harvested * dry_cell_mass / 1000,
            "Accum. Cell Harvest (10^12 Cells)": np.cumsum(cells_harvested) / 1e12,
            "Accum. Biomass Harvest (kg)": np.cumsum(cells_harvested * dry_cell_mass / 1000),
        }

        df = pd.DataFrame(data)
        
        if verbose:
            print("\n=== CULTIVATION SUMMARY ===")
            print(df.to_string(index=False))
        
        return df


### Mass balancing methods ----------------------------------------------------
    # These methods provide CHON elemental closure diagnostics for the simulation.
    # They are organized in logical dependency order:
    #   1. Context & configuration (private helpers)
    #   2. Core computation engine (private)
    #   3. DataFrame builders (public static)
    #   4. Stoichiometric closure (rate-integrated pathway)
    #   5. Boundary closure (black-box in/out pathway)
    #   6. Delta closure (difference between the two pathways)
    #   7. Excel export (integration with save())

# ------------------------------------------------------------------
# 1. Context & configuration helpers
# ------------------------------------------------------------------
    def _validate_closure_context(self) -> None:
        """Validate that the data needed for closure calculations is available."""
        if not self.summary_stats and not self.projected_totals_stats and not self.simulated_window_totals_stats:
            raise ValueError("No summary/totals statistics available. Run a simulation first.")
        if self.parameters is None:
            raise ValueError("Simulation parameters are required for CHON closure calculations.")

    def _get_closure_stat(self, key: str, default: float = 0.0) -> float:
        """Return a totals stat value, preferring simulated-window totals."""
        value = self.simulated_window_totals_stats.get(
            key,
            self.projected_totals_stats.get(key, self.summary_stats.get(key, default))
        )
        return float(value or 0.0)

    def _get_component_definitions(self) -> List[Dict[str, Any]]:
        """Return the 9-species component catalog used by boundary closure calculations."""
        self._validate_closure_context()
        return [
            {
                'label': 'Glucose',
                'description': 'Primary substrate entering with fresh medium and feed.',
                'in_key': 'total_glucose_in_mmol',
                'out_key': 'total_glucose_out_mmol',
                'net_key': 'total_glucose_net_mmol',
                'net_label': 'Net (−consumed)',
                'mw': 180.1572,
            },
            {
                'label': 'Glutamine',
                'description': 'Secondary substrate entering with fresh medium and feed.',
                'in_key': 'total_glutamine_in_mmol',
                'out_key': 'total_glutamine_out_mmol',
                'net_key': 'total_glutamine_net_mmol',
                'net_label': 'Net (−consumed)',
                'mw': 146.1456,
            },
            {
                'label': 'Amino Acids excl. Gln',
                'description': 'Lumped amino-acid pool entering with fresh medium and feed.',
                'in_key': 'total_AA_in_mmol',
                'out_key': 'total_AA_out_mmol',
                'net_key': 'total_AA_net_mmol',
                'net_label': 'Net (−consumed)',
                'mw': float(self.parameters.cell_properties.M_AAs_ex_glutamine),
            },
            {
                'label': 'Lactate',
                'description': 'Byproduct leaving in liquid outlet streams; net value tracks metabolic production.',
                'in_key': 'total_lactate_in_mmol',
                'out_key': 'total_lactate_out_mmol',
                'net_key': 'total_lactate_net_mmol',
                'net_label': 'Net (+produced)',
                'mw': 89.07,
            },
            {
                'label': 'Ammonium',
                'description': 'Nitrogenous byproduct leaving in liquid outlet streams; net value tracks metabolic production.',
                'in_key': 'total_ammonium_in_mmol',
                'out_key': 'total_ammonium_out_mmol',
                'net_key': 'total_ammonium_net_mmol',
                'net_label': 'Net (+produced)',
                'mw': 17.031,
            },
            {
                'label': 'CO2',
                'description': 'Total inorganic carbon tracked as CO2 equivalent, including dissolved outlet and gas stripping.',
                'in_key': 'total_CT_in_mmol',
                'out_key': 'total_CT_out_mmol',
                'net_key': 'total_CT_net_mmol',
                'net_label': 'Net (+produced)',
                'mw': 44.01,
            },
            {
                'label': 'O2',
                'description': 'Oxygen supplied via dissolved-feed oxygen and gas transfer; net value tracks cellular consumption.',
                'in_key': 'total_O2_in_mmol',
                'out_key': 'total_O2_out_mmol',
                'net_key': 'total_O2_net_mmol',
                'net_label': 'Net (−consumed)',
                'mw': 31.998,
            },
            {
                'label': 'H2O',
                'description': 'Metabolic water production inferred from the stoichiometric closure.',
                'in_key': None,
                'out_key': None,
                'net_key': 'total_water_produced_mmol',
                'net_label': 'Produced',
                'mw': 18.01528,
            },
            {
                'label': 'DCM',
                'description': 'Dry cell mass produced from the inoculum.',
                'in_key': 'total_DCM_in_cmmol',
                'out_key': 'total_DCM_out_cmmol',
                'net_key': 'total_DCM_produced_cmmol',
                'net_label': 'Produced',
                'mw': float(self.parameters.cell_properties.M_biomass),
            },
        ]

    def _get_component_specs_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Return component definitions keyed by label for O(1) lookup."""
        return {spec['label']: spec for spec in self._get_component_definitions()}

    def _get_biomass_chon_formula(self) -> Dict[str, float]:
        """Return the configured biomass CHON formula on the DCM basis."""
        return {
            'C': float(self.parameters.cell_properties.biomass_C),
            'H': float(self.parameters.cell_properties.biomass_H),
            'O': float(self.parameters.cell_properties.biomass_O),
            'N': float(self.parameters.cell_properties.biomass_N),
        }

    def _get_lumped_aa_chon_formula(self) -> Dict[str, float]:
        """Return the net CHON formula of the lumped amino-acid pool excl. Gln.

        The process-level AA stream is tracked as the net amino-acid demand used by
        the stoichiometric model: all anabolic amino acids except glutamine, with
        glutamate included using its anabolic sign. This mirrors the meaning of
        ``M_AAs_ex_glutamine`` and ``AA_anabolic_coeff`` in the stoichiometric layer.
        """
        amino_acids = (
            'Arg', 'Cys', 'His', 'Ile', 'Lys', 'Met', 'Phe', 'Thr', 'Trp', 'Tyr',
            'Val', 'Ala', 'Asn', 'Asp', 'Gly', 'Leu', 'Pro', 'Ser', 'Glu',
        )

        net_coeff = 0.0
        net_formula = {element: 0.0 for element in _ELEMENTS_CHON}
        anabolic = self.parameters.anabolic_reaction

        for species in amino_acids:
            coeff = float(getattr(anabolic, species))
            net_coeff += coeff
            for element in _ELEMENTS_CHON:
                net_formula[element] += coeff * _SPECIES_FORMULAS_CHON[species][element]

        if net_coeff == 0.0:
            raise ValueError("Lumped amino-acid CHON formula is undefined because the net anabolic AA coefficient is zero.")

        return {
            element: -net_formula[element] / abs(net_coeff)
            for element in _ELEMENTS_CHON
        }

# ------------------------------------------------------------------
# 2. Core computation engine
# ------------------------------------------------------------------
    @staticmethod
    def _compute_element_balance(component_specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate CHON totals from a component basis (reactants vs products).

        This is the central engine used by all three closure pathways
        (stoichiometric, boundary, and delta).
        """
        element_totals: Dict[str, Dict[str, float]] = {}

        for element in _ELEMENTS_CHON:
            reactants = 0.0
            products = 0.0
            for spec in component_specs:
                total_atoms = spec['amount'] * spec['formula'][element]
                if spec['side'] == 'Input':
                    reactants += total_atoms
                else:
                    products += total_atoms

            residual = products - reactants
            relative_residual_pct = 100.0 * residual / reactants if reactants else np.nan
            element_totals[element] = {
                'reactants': reactants,
                'products': products,
                'residual': residual,
                'relative_residual_pct': relative_residual_pct,
            }
        return element_totals

# ------------------------------------------------------------------
# 3. DataFrame builders (public static — convert closure dicts to tables)
# ------------------------------------------------------------------
    @staticmethod
    def build_closure_summary_df(closure: Dict[str, Any]) -> pd.DataFrame:
        """Convert a closure result dict into an element-level summary table."""
        rows = []
        for element, values in closure['elements'].items():
            rows.append({
                'Element': element,
                'Reactants (mol-atoms)': values['reactants'],
                'Products (mol-atoms)': values['products'],
                'Residual (mol-atoms)': values['residual'],
                'Relative Residual (%)': values['relative_residual_pct'],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def build_component_specs_df(component_specs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert component specifications into a tabular basis report.

        If the specs include a ``yield_on_biomass`` key (e.g. stoichiometric
        closure), a "Yield on Biomass" column is appended showing the average
        yield per Cmol of dry biomass produced.
        """
        rows = []
        for spec in component_specs:
            row = {
                'Component': spec['label'],
                'Side': spec['side'],
                'Description': spec['description'],
                'Amount': spec['amount'],
                'Amount Unit': spec['amount_unit'],
                'C': spec['formula']['C'],
                'H': spec['formula']['H'],
                'O': spec['formula']['O'],
                'N': spec['formula']['N'],
            }
            if spec.get('yield_on_biomass') is not None:
                row['Yield on Biomass'] = spec['yield_on_biomass']
                row['Yield Unit'] = spec.get('yield_unit', '')
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def build_formula_reference_df(closure: Dict[str, Any]) -> pd.DataFrame:
        """Return the CHON formulas used for a closure report."""
        return pd.DataFrame([
            {
                'Formula': 'Biomass CHON basis',
                **closure['biomass_chon_formula'],
            },
            {
                'Formula': 'Lumped amino-acid CHON basis',
                **closure['lumped_aa_chon_formula'],
            },
        ])

    @staticmethod
    def build_delta_summary_df(
        stoich_closure: Dict[str, Any],
        boundary_closure: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build an element-level comparison of stoichiometric vs boundary closure."""
        rows = []
        for element in _ELEMENTS_CHON:
            s = stoich_closure['elements'][element]
            b = boundary_closure['elements'][element]
            delta_residual = b['residual'] - s['residual']
            rows.append({
                'Element': element,
                'Stoich Reactants (mol-atoms)': s['reactants'],
                'Stoich Products (mol-atoms)': s['products'],
                'Stoich Residual (mol-atoms)': s['residual'],
                'Boundary Reactants (mol-atoms)': b['reactants'],
                'Boundary Products (mol-atoms)': b['products'],
                'Boundary Residual (mol-atoms)': b['residual'],
                'Delta Residual (mol-atoms)': delta_residual,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def build_delta_component_specs_df(component_specs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert delta component specifications into a tabular report.

        Parameters
        ----------
        component_specs : list of dict
            Delta component specifications from the ``component_specs`` entry
            returned by :meth:`compute_closure_delta`.

        Returns
        -------
        pandas.DataFrame
            Stoichiometric amount, boundary net amount, delta amount, unit, and
            CHON formula coefficients for every component.
        """
        rows = []
        for spec in component_specs:
            rows.append({
                'Component': spec['label'],
                'Side': spec['side'],
                'Description': spec['description'],
                'Stoich Amount': spec['stoich_amount'],
                'Boundary Net Amount': spec['boundary_net_amount'],
                'Delta Amount': spec['amount'],
                'Amount Unit': spec['amount_unit'],
                'C': spec['formula']['C'],
                'H': spec['formula']['H'],
                'O': spec['formula']['O'],
                'N': spec['formula']['N'],
            })
        return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 4. Stoichiometric closure — integrates simulated reaction rates
# ------------------------------------------------------------------
    @staticmethod
    def _get_batch_rate_series(batch: BatchData, attr_name: str, expected_size: int) -> np.ndarray:
        """Retrieve a stored batch rate array with shape validation."""
        values = getattr(batch, attr_name)
        if values is None:
            raise ValueError(
                f"Stoichiometric closure requires stored '{attr_name}' data for batch {batch.batch_number}. "
                "Run the simulation with performance_mode=False."
            )
        array = np.asarray(values, dtype=float)
        if array.size != expected_size:
            raise ValueError(
                f"Stoichiometric closure expected '{attr_name}' to have {expected_size} points for "
                f"batch {batch.batch_number}, got {array.size}."
            )
        return array

    def _build_stoichiometric_component_basis(self) -> List[Dict[str, Any]]:
        """Build a CHON component basis by integrating stored kinetic rate trajectories.

        Integrates μ, q_Glc, q_Gln, q_AA, q_O2, q_Lac, q_Amm, q_CO2, and q_H2O
        via trapezoidal rule over all batches. Requires ``performance_mode=False``.
        """
        if not self.batches:
            raise ValueError(
                "Stoichiometric closure requires stored batch trajectories. "
                "Run the simulation with performance_mode=False."
            )

        totals = {
            'Glucose consumed': 0.0,
            'Glutamine consumed': 0.0,
            'Amino acids consumed': 0.0,
            'O2 consumed': 0.0,
            'Dry biomass': 0.0,
            'Lactate produced': 0.0,
            'Ammonium produced': 0.0,
            'CO2 produced': 0.0,
            'H2O produced': 0.0,
        }

        cells_per_mol = float(self.parameters.cell_properties.cells_per_mol)

        for batch in self.batches:
            t_points = np.asarray(batch.t_points, dtype=float)
            if t_points.size < 2:
                continue

            y_points = np.asarray(batch.y_points, dtype=float)
            if y_points.ndim != 2 or y_points.shape[1] != t_points.size:
                raise ValueError(
                    f"Stoichiometric closure expected y_points to have shape (*, {t_points.size}) for "
                    f"batch {batch.batch_number}."
                )

            dt = np.diff(t_points)
            cell_volume = y_points[0, :] * y_points[5, :]  # Xv * V (cells)

            def _trapz_rate(rate: np.ndarray) -> float:
                """Trapezoidal integral of rate(t) * Xv(t) * V(t) over the batch."""
                integrand = rate * cell_volume
                return float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt))

            mu = self._get_batch_rate_series(batch, 'mu_list', t_points.size)
            q_glc = self._get_batch_rate_series(batch, 'q_Glc_list', t_points.size)
            q_gln = self._get_batch_rate_series(batch, 'q_Gln_list', t_points.size)
            q_aa = self._get_batch_rate_series(batch, 'q_AA_list', t_points.size)
            q_o2 = self._get_batch_rate_series(batch, 'q_O2_list', t_points.size)
            q_lac = self._get_batch_rate_series(batch, 'q_Lac_list', t_points.size)
            q_amm = self._get_batch_rate_series(batch, 'q_Amm_list', t_points.size)
            q_co2 = self._get_batch_rate_series(batch, 'q_CO2_list', t_points.size)
            q_h2o = self._get_batch_rate_series(batch, 'q_H2O_list', t_points.size)

            totals['Glucose consumed'] += _trapz_rate(q_glc) / 1000.0
            totals['Glutamine consumed'] += _trapz_rate(q_gln) / 1000.0
            totals['Amino acids consumed'] += _trapz_rate(q_aa) / 1000.0
            totals['O2 consumed'] += _trapz_rate(q_o2) / 1000.0
            totals['Dry biomass'] += _trapz_rate(mu) / cells_per_mol
            totals['Lactate produced'] += _trapz_rate(q_lac) / 1000.0
            totals['Ammonium produced'] += _trapz_rate(q_amm) / 1000.0
            totals['CO2 produced'] += _trapz_rate(q_co2) / 1000.0
            totals['H2O produced'] += _trapz_rate(q_h2o) / 1000.0

        return [
            {
                'label': 'Glucose consumed',
                'side': 'Input',
                'description': 'Integrated glucose uptake from the simulated specific reaction rates.',
                'amount': totals['Glucose consumed'],
                'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Glc'],
            },
            {
                'label': 'Glutamine consumed',
                'side': 'Input',
                'description': 'Integrated glutamine uptake from the simulated specific reaction rates.',
                'amount': totals['Glutamine consumed'],
                'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Gln'],
            },
            {
                'label': 'Amino acids consumed',
                'side': 'Input',
                'description': 'Integrated uptake of the lumped amino-acid pool from the simulated specific reaction rates.',
                'amount': totals['Amino acids consumed'],
                'amount_unit': 'mol',
                'formula': self._get_lumped_aa_chon_formula(),
            },
            {
                'label': 'O2 consumed',
                'side': 'Input',
                'description': 'Integrated oxygen uptake from the simulated specific reaction rates.',
                'amount': totals['O2 consumed'],
                'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['O2'],
            },
            {
                'label': 'Dry biomass',
                'side': 'Output',
                'description': 'Integrated biomass formation from the simulated growth rate over the stored batch trajectories.',
                'amount': totals['Dry biomass'],
                'amount_unit': 'Cmol',
                'formula': self._get_biomass_chon_formula(),
            },
            {
                'label': 'Lactate produced',
                'side': 'Output',
                'description': 'Integrated lactate formation from the simulated specific reaction rates.',
                'amount': totals['Lactate produced'],
                'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Lac'],
            },
            {
                'label': 'Ammonium produced',
                'side': 'Output',
                'description': 'Integrated ammonium formation from the simulated specific reaction rates.',
                'amount': totals['Ammonium produced'],
                'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['NH3'],
            },
            {
                'label': 'CO2 produced',
                'side': 'Output',
                'description': 'Integrated carbon dioxide formation from the simulated specific reaction rates.',
                'amount': totals['CO2 produced'],
                'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['CO2'],
            },
            {
                'label': 'H2O produced',
                'side': 'Output',
                'description': 'Integrated metabolic water formation from the simulated specific reaction rates.',
                'amount': totals['H2O produced'],
                'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['H2O'],
            },
        ]

    def compute_stoichiometric_closure(self) -> Dict[str, Any]:
        """Compute CHON elemental closure from integrated simulated reaction rates.

        Integrates the stored μ and q_* trajectories over all batches and checks
        whether the kinetic model's reaction basis is internally consistent.
        Requires ``performance_mode=False``.

        Component specs are annotated with a ``yield_on_biomass`` field giving
        the average yield per Cmol of dry biomass produced.

        Returns
        -------
        dict
            Closure basis, component specifications, element totals, maximum
            absolute residual, and the biomass and lumped-AA CHON formulas.

        Raises
        ------
        ValueError
            If parameters, summary data, batch trajectories, or required rate
            traces are unavailable.
        """
        self._validate_closure_context()

        component_specs = self._build_stoichiometric_component_basis()

        # Extract biomass amount for yield-on-biomass calculation.
        biomass_amount_Cmol = 0.0
        for spec in component_specs:
            if spec['label'] == 'Dry biomass':
                biomass_amount_Cmol = spec['amount']
                break

        if biomass_amount_Cmol > 0:
            for spec in component_specs:
                if spec['label'] == 'Dry biomass':
                    spec['yield_on_biomass'] = 1.0
                    spec['yield_unit'] = 'Cmol / Cmol DCM'
                elif spec['amount_unit'] == 'mol':
                    spec['yield_on_biomass'] = spec['amount'] / biomass_amount_Cmol
                    spec['yield_unit'] = 'mol / Cmol DCM'
                elif spec['amount_unit'] == 'Cmol':
                    spec['yield_on_biomass'] = spec['amount'] / biomass_amount_Cmol
                    spec['yield_unit'] = 'Cmol / Cmol DCM'

        element_totals = self._compute_element_balance(component_specs)

        max_abs_residual = max(abs(values['residual']) for values in element_totals.values())

        return {
            'basis': 'integrated_stoichiometric_process_chon_closure',
            'component_specs': component_specs,
            'elements': element_totals,
            'max_abs_residual': max_abs_residual,
            'biomass_chon_formula': self._get_biomass_chon_formula(),
            'lumped_aa_chon_formula': self._get_lumped_aa_chon_formula(),
        }

# ------------------------------------------------------------------
# 5. Boundary closure — black-box in/out balance
# ------------------------------------------------------------------
    def _build_boundary_component_basis(self) -> List[Dict[str, Any]]:
        """Build a CHON component basis from absolute process boundary streams.

        Treats the process as a black box: everything entering the reactor boundary
        is an input, everything leaving is an output. Internal carryover between
        repeated batches is excluded from both sides.
        """
        self._validate_closure_context()

        component_map = self._get_component_specs_lookup()
        aa_formula = self._get_lumped_aa_chon_formula()
        biomass_formula = self._get_biomass_chon_formula()
        density_g_per_L = float(self.parameters.shared_medium_properties.density)

        def _stat_amount(key: str) -> float:
            return self._get_closure_stat(key)

        def _water_L_to_mol(volume_L: float) -> float:
            return float(volume_L) * density_g_per_L / _WATER_MW_G_PER_MOL

        # Components in
        glucose_in_mol = _stat_amount(component_map['Glucose']['in_key']) / 1000.0
        glutamine_in_mol = _stat_amount(component_map['Glutamine']['in_key']) / 1000.0
        aa_in_mol = _stat_amount(component_map['Amino Acids excl. Gln']['in_key']) / 1000.0
        lactate_in_mol = _stat_amount(component_map['Lactate']['in_key']) / 1000.0
        ammonium_in_mol = _stat_amount(component_map['Ammonium']['in_key']) / 1000.0
        oxygen_in_mol = _stat_amount(component_map['O2']['in_key']) / 1000.0
        co2_in_mol = _stat_amount(component_map['CO2']['in_key']) / 1000.0
        total_DCM_in_cmol = _stat_amount('total_DCM_in_cmmol') / 1000.0

        # Components out
        glucose_out_mol = _stat_amount(component_map['Glucose']['out_key']) / 1000.0
        glutamine_out_mol = _stat_amount(component_map['Glutamine']['out_key']) / 1000.0
        aa_out_mol = _stat_amount(component_map['Amino Acids excl. Gln']['out_key']) / 1000.0
        lactate_out_mol = _stat_amount(component_map['Lactate']['out_key']) / 1000.0
        ammonium_out_mol = _stat_amount(component_map['Ammonium']['out_key']) / 1000.0
        oxygen_out_mol = _stat_amount(component_map['O2']['out_key']) / 1000.0
        co2_out_mol = _stat_amount(component_map['CO2']['out_key']) / 1000.0
        total_DCM_out_cmol = _stat_amount('total_DCM_out_cmmol') / 1000.0

        # Volumes
        medium_volume_L = _stat_amount('total_medium_in_L')
        base_volume_L = _stat_amount('total_base_in_L')
        acid_volume_L = _stat_amount('total_acid_in_L')
        total_outlet_volume_L = _stat_amount('total_outlet_volume_L')

        total_water_produced_mol = _stat_amount('total_water_produced_mmol') / 1000

        medium_water_mol = _water_L_to_mol(medium_volume_L)
        base_water_mol = _water_L_to_mol(base_volume_L)
        acid_water_mol = _water_L_to_mol(acid_volume_L)
        outlet_water_mol = _water_L_to_mol(total_outlet_volume_L)

        return [
            {
                'label': 'Fresh glucose in', 'side': 'Input',
                'description': 'Total glucose entering with fresh initial charge and feed across the simulated window.',
                'amount': glucose_in_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Glc'],
            },
            {
                'label': 'Fresh glutamine in', 'side': 'Input',
                'description': 'Total glutamine entering with fresh initial charge and feed across the simulated window.',
                'amount': glutamine_in_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Gln'],
            },
            {
                'label': 'Fresh amino acids in', 'side': 'Input',
                'description': 'Total lumped amino-acid pool entering with fresh initial charge and feed.',
                'amount': aa_in_mol, 'amount_unit': 'mol',
                'formula': aa_formula,
            },
            {
                'label': 'Fresh lactate in', 'side': 'Input',
                'description': 'Lactate entering from fresh external media only.',
                'amount': lactate_in_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Lac'],
            },
            {
                'label': 'Fresh ammonium in', 'side': 'Input',
                'description': 'Ammonium entering from fresh external media only.',
                'amount': ammonium_in_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['NH3'],
            },
            {
                'label': 'O2 in', 'side': 'Input',
                'description': 'Total oxygen crossing the reactor boundary into the liquid phase, including fresh initial charge.',
                'amount': oxygen_in_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['O2'],
            },
            {
                'label': 'CO2 in', 'side': 'Input',
                'description': 'Total inorganic carbon entering from fresh medium, initial charge, and gas transfer.',
                'amount': co2_in_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['CO2'],
            },
            {
                'label': 'Medium, base, and acid water in', 'side': 'Input',
                'description': 'Carrier water entering with fresh medium charges, feed, base solution, and acid solution.',
                'amount': medium_water_mol + base_water_mol + acid_water_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['H2O'],
            },
            {
                'label': 'DCM in', 'side': 'Input',
                'description': 'Dry biomass entering via inoculum, only with fresh externally charged broth at batch starts.',
                'amount': total_DCM_in_cmol, 'amount_unit': 'Cmol',
                'formula': biomass_formula,
            },
            {
                'label': 'Glucose out', 'side': 'Output',
                'description': 'Total glucose leaving in external liquid outlet streams.',
                'amount': glucose_out_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Glc'],
            },
            {
                'label': 'Glutamine out', 'side': 'Output',
                'description': 'Total glutamine leaving in external liquid outlet streams.',
                'amount': glutamine_out_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Gln'],
            },
            {
                'label': 'Amino acids out', 'side': 'Output',
                'description': 'Total lumped amino-acid pool leaving in external liquid outlet streams.',
                'amount': aa_out_mol, 'amount_unit': 'mol',
                'formula': aa_formula,
            },
            {
                'label': 'Lactate out', 'side': 'Output',
                'description': 'Total lactate leaving in external liquid outlet streams.',
                'amount': lactate_out_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['Lac'],
            },
            {
                'label': 'Ammonium out', 'side': 'Output',
                'description': 'Total ammonium leaving in external liquid outlet streams.',
                'amount': ammonium_out_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['NH3'],
            },
            {
                'label': 'O2 out', 'side': 'Output',
                'description': 'Total dissolved oxygen leaving the reactor boundary.',
                'amount': oxygen_out_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['O2'],
            },
            {
                'label': 'CO2 out', 'side': 'Output',
                'description': 'Total inorganic carbon leaving the reactor boundary, including dissolved outlet and gas stripping.',
                'amount': co2_out_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['CO2'],
            },
            {
                'label': 'Boundary water out', 'side': 'Output',
                'description': 'Carrier water leaving in external liquid outlet streams. Includes water produced by biomass metabolism.',
                'amount': outlet_water_mol + total_water_produced_mol, 'amount_unit': 'mol',
                'formula': _SPECIES_FORMULAS_CHON['H2O'],
            },
            {
                'label': 'DCM out', 'side': 'Output',
                'description': 'Dry biomass crossing the external process boundary in harvest and bleed streams.',
                'amount': total_DCM_out_cmol, 'amount_unit': 'Cmol',
                'formula': biomass_formula,
            },
        ]

    def compute_boundary_closure(self) -> Dict[str, Any]:
        """Compute whole-process external-boundary CHON closure for the simulated window.

        Treats the process as a black box: balances all CHON atoms entering
        versus leaving the reactor boundary. Internal carryover between repeated
        batches is excluded.

        Returns
        -------
        dict
            Closure basis, component specifications, element totals, maximum
            absolute residual, and the biomass and lumped-AA CHON formulas.

        Raises
        ------
        ValueError
            If parameters or summary/totals data are unavailable.
        """
        self._validate_closure_context()

        component_specs = self._build_boundary_component_basis()
        element_totals = self._compute_element_balance(component_specs)
        max_abs_residual = max(abs(values['residual']) for values in element_totals.values())

        return {
            'basis': 'external_process_boundary_chon_closure',
            'component_specs': component_specs,
            'elements': element_totals,
            'max_abs_residual': max_abs_residual,
            'biomass_chon_formula': self._get_biomass_chon_formula(),
            'lumped_aa_chon_formula': self._get_lumped_aa_chon_formula(),
        }

# ------------------------------------------------------------------
# 6. Delta closure — difference between stoichiometric and boundary
# ------------------------------------------------------------------
    @staticmethod
    def _get_stoich_to_boundary_label_map() -> Dict[str, Dict[str, str]]:
        """Map stoichiometric component labels to their boundary in/out labels.

        ``boundary_net = boundary_in − boundary_out`` for consumed species and
        ``boundary_out − boundary_in`` for produced species.
        """
        return {
            'Glucose consumed':      {'boundary_in': 'Fresh glucose in',         'boundary_out': 'Glucose out'},
            'Glutamine consumed':    {'boundary_in': 'Fresh glutamine in',       'boundary_out': 'Glutamine out'},
            'Amino acids consumed':  {'boundary_in': 'Fresh amino acids in',     'boundary_out': 'Amino acids out'},
            'O2 consumed':           {'boundary_in': 'O2 in',                    'boundary_out': 'O2 out'},
            'Dry biomass':           {'boundary_in': 'DCM in',                   'boundary_out': 'DCM out'},
            'Lactate produced':      {'boundary_in': 'Fresh lactate in',         'boundary_out': 'Lactate out'},
            'Ammonium produced':     {'boundary_in': 'Fresh ammonium in',        'boundary_out': 'Ammonium out'},
            'CO2 produced':          {'boundary_in': 'CO2 in',                   'boundary_out': 'CO2 out'},
            'H2O produced':          {'boundary_in': 'Medium, base, and acid water in', 'boundary_out': 'Boundary water out'},
        }

    def compute_closure_delta(
        self,
        stoich_closure: Optional[Dict[str, Any]] = None,
        boundary_closure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute the element-wise difference between boundary and stoichiometric closure.

        The stoichiometric closure predicts metabolite consumption/production from
        integrated reaction rates. The boundary closure tracks what actually crossed
        the reactor boundary. This delta surfaces any discrepancy between the two,
        useful for troubleshooting numerical drift or model inconsistencies.

        Parameters
        ----------
        stoich_closure : dict, optional
            Result from :meth:`compute_stoichiometric_closure`. Computed when
            omitted.
        boundary_closure : dict, optional
            Result from :meth:`compute_boundary_closure`. Computed when omitted.

        Returns
        -------
        dict
            Delta component basis, CHON element totals, and references to the
            stoichiometric and boundary closure results.
        """
        stoich_closure = stoich_closure or self.compute_stoichiometric_closure()
        boundary_closure = boundary_closure or self.compute_boundary_closure()

        label_map = self._get_stoich_to_boundary_label_map()
        boundary_lookup = {spec['label']: spec for spec in boundary_closure['component_specs']}

        delta_specs: List[Dict[str, Any]] = []
        for stoich_spec in stoich_closure['component_specs']:
            stoich_label = stoich_spec['label']
            mapping = label_map.get(stoich_label)
            if mapping is None:
                continue

            b_in = boundary_lookup.get(mapping['boundary_in'])
            b_out = boundary_lookup.get(mapping['boundary_out'])
            if b_in is None or b_out is None:
                continue

            stoich_amount = stoich_spec['amount']
            if stoich_spec['side'] == 'Input':
                boundary_net = b_in['amount'] - b_out['amount']
            else:
                boundary_net = b_out['amount'] - b_in['amount']

            delta = boundary_net - stoich_amount

            delta_specs.append({
                'label': f"{stoich_label} delta",
                'side': stoich_spec['side'],
                'description': (
                    f"Difference between boundary net flux and integrated stoichiometric rate for "
                    f"{stoich_label.replace(' consumed', '').replace(' produced', '').lower()}."
                ),
                'stoich_amount': stoich_amount,
                'boundary_net_amount': boundary_net,
                'amount': delta,
                'amount_unit': stoich_spec['amount_unit'],
                'formula': stoich_spec['formula'],
            })

        element_totals = self._compute_element_balance(delta_specs)
        max_abs_residual = max(abs(v['residual']) for v in element_totals.values()) if element_totals else 0.0

        return {
            'basis': 'chon_closure_delta',
            'component_specs': delta_specs,
            'elements': element_totals,
            'max_abs_residual': max_abs_residual,
            'biomass_chon_formula': self._get_biomass_chon_formula(),
            'lumped_aa_chon_formula': self._get_lumped_aa_chon_formula(),
            'stoich_closure': stoich_closure,
            'boundary_closure': boundary_closure,
        }

    def get_closure_delta_df(
        self,
        delta_result: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Return the CHON closure delta as a DataFrame.

        Compares integrated stoichiometric rates against the actual boundary net
        flux for each metabolite. A positive delta means the boundary shows more
        net consumption (or production) than the kinetic model predicts.

        Parameters
        ----------
        delta_result : dict, optional
            Result from :meth:`compute_closure_delta`. Computed when omitted.

        Returns
        -------
        pandas.DataFrame
            Component-level stoichiometric amounts, boundary net amounts, and
            their differences on the configured CHON basis.
        """
        delta_result = delta_result or self.compute_closure_delta()
        return self.build_delta_component_specs_df(delta_result['component_specs'])

# ------------------------------------------------------------------
# 7. Excel export — integration with save()
# ------------------------------------------------------------------
    @staticmethod
    def _write_sheet_sections(
        writer: pd.ExcelWriter,
        sheet_name: str,
        sections: List[Dict[str, Any]],
    ) -> None:
        """Write multiple titled DataFrame sections to a single Excel sheet."""
        worksheet = writer.book.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet
        current_row = 0

        for section in sections:
            title = section.get('title')
            dataframe = section['dataframe']
            if title:
                worksheet.write(current_row, 0, title)
                current_row += 1

            dataframe.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
            current_row += len(dataframe.index) + 3

    def _build_closure_sheet_sections(
        self,
        closure: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build the standard three-section layout for a closure report sheet.

        Sections: (1) element-level summary, (2) component basis table,
        (3) CHON formula reference. Titles are derived from the closure's
        ``basis`` key via ``_CLOSURE_TITLES``.
        """
        basis = closure.get('basis', '')
        titles = self._CLOSURE_TITLES.get(basis, ('', '', ''))
        return [
            {
                'title': titles[0],
                'dataframe': self.build_closure_summary_df(closure),
            },
            {
                'title': titles[1],
                'dataframe': self.build_component_specs_df(closure['component_specs']),
            },
            {
                'title': titles[2],
                'dataframe': self.build_formula_reference_df(closure),
            },
        ]

    def _build_delta_sheet_sections(self) -> List[Dict[str, Any]]:
        """Build the export layout for the CHON closure delta sheet.

        Sections: (1) element-level stoich-vs-boundary comparison,
        (2) per-component delta basis, (3) CHON formula reference.
        """
        delta_result = self.compute_closure_delta()
        titles = self._CLOSURE_TITLES.get('chon_closure_delta', ('', '', ''))
        return [
            {
                'title': titles[0],
                'dataframe': self.build_delta_summary_df(
                    delta_result['stoich_closure'],
                    delta_result['boundary_closure'],
                ),
            },
            {
                'title': titles[1],
                'dataframe': self.build_delta_component_specs_df(delta_result['component_specs']),
            },
            {
                'title': titles[2],
                'dataframe': self.build_formula_reference_df(delta_result),
            },
        ]

    def _write_closure_sheets(self, writer: pd.ExcelWriter) -> None:
        """Write the three CHON closure workbook sheets.

        Sheets written:
        1. ``Stoich CHON Closure`` — integrated from simulated reaction rates.
        2. ``Boundary CHON Closure`` — whole-process black-box element balance.
        3. ``CHON Closure Delta`` — difference between boundary and stoichiometric.
        """
        builders = [
            (_MASS_BALANCE_STOICH_SHEET, self._build_stoich_sheet_sections),
            (_MASS_BALANCE_BOUNDARY_SHEET, self._build_boundary_sheet_sections),
            (_MASS_BALANCE_DELTA_SHEET, self._build_delta_sheet_sections),
        ]
        for sheet_name, builder in builders:
            try:
                sections = builder()
            except ValueError as exc:
                sections = [{'title': sheet_name, 'dataframe': self._build_status_df(str(exc))}]
            self._write_sheet_sections(writer, sheet_name, sections)

    def _build_stoich_sheet_sections(self) -> List[Dict[str, Any]]:
        """Build the report sections for the stoichiometric CHON closure sheet."""
        closure = self.compute_stoichiometric_closure()
        return self._build_closure_sheet_sections(closure)

    def _build_boundary_sheet_sections(self) -> List[Dict[str, Any]]:
        """Build the report sections for the external boundary CHON closure sheet."""
        closure = self.compute_boundary_closure()
        return self._build_closure_sheet_sections(closure)


### Plotting API on SimulationData with lazy imports to avoid circular dependencies ###
# Note: Visualization functions have been moved to dedicated modules
# Comprehensive plotting API - enables .plot_*() notation for all visualization functions
    
    # Batch-level plots (accept batch_number parameter)
    def plot_biomass_metabolites(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot viable-cell density and metabolite concentrations.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Biomass and metabolite figure.
        """
        from animakinetica.analysis.batch_visualization import plot_batch_biomass_metabolites
        return plot_batch_biomass_metabolites(self, batch_number=batch_number, layout_kwargs=layout_kwargs)
    
    def plot_volume_feed(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot working volume, dilution rates, and inlet rates.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Volume and inlet-rate figure.
        """
        from animakinetica.analysis.batch_visualization import plot_batch_volume_feed
        return plot_batch_volume_feed(self, batch_number=batch_number, layout_kwargs=layout_kwargs)
    
    def plot_gas_transfer(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot dissolved oxygen and gas-transfer operating variables.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Gas-transfer and hydrodynamics figure.
        """
        from animakinetica.analysis.gas_visualization import plot_batch_gas_transfer
        return plot_batch_gas_transfer(self, batch_number=batch_number, layout_kwargs=layout_kwargs)
    
    def plot_pH_control(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot pH-control state, acid/base rates, and carbon-dioxide control.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            pH-control figure.
        """
        from animakinetica.analysis.batch_visualization import plot_batch_pH_control
        return plot_batch_pH_control(self, batch_number=batch_number, layout_kwargs=layout_kwargs)

    def plot_nutrient_factors(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot nutrient-limitation factors over time.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Nutrient-factor figure.
        """
        from animakinetica.analysis.kinetic_visualization import plot_simulated_NFs
        return plot_simulated_NFs(self, batch_number=batch_number, layout_kwargs=layout_kwargs)
    
    def plot_inhibition_factors(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot inhibition factors over time.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Inhibition-factor figure.
        """
        from animakinetica.analysis.kinetic_visualization import plot_simulated_IFs
        return plot_simulated_IFs(self, batch_number=batch_number, layout_kwargs=layout_kwargs)
    
    def plot_kinetic_rates(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot specific growth and uptake/production rates.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Kinetic-rate figure.
        """
        from animakinetica.analysis.kinetic_visualization import plot_kinetic_rates
        return plot_kinetic_rates(self, batch_number=batch_number, layout_kwargs=layout_kwargs)
    
    def plot_kinetics(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot nutrient, inhibition, lag, growth, and rate trajectories.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Comprehensive kinetic-state figure.
        """
        from animakinetica.analysis.kinetic_visualization import plot_kinetics
        return plot_kinetics(self, batch_number=batch_number, layout_kwargs=layout_kwargs)

    def plot_stoichiometric_yields(self, batch_number: int = 1, layout_kwargs: Optional[Dict] = None):
        """Plot tracked stoichiometric yield coefficients.

        Parameters
        ----------
        batch_number : int, default=1
            One-based batch identifier.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Stoichiometric-yield figure.
        """
        from animakinetica.analysis.stoichiometric_visualization import plot_stoichiometric_yields
        return plot_stoichiometric_yields(self, batch_number=batch_number, layout_kwargs=layout_kwargs)
    
    # Simulation-level plots (multi-batch cumulative metrics)
    def plot_cumulative_harvest(self, max_batches: Optional[int] = None, layout_kwargs: Optional[Dict] = None):
        """Plot cumulative harvested biomass and liquid volume.

        Parameters
        ----------
        max_batches : int, optional
            Maximum number of leading batches to include. Defaults to all.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Cumulative harvest figure.
        """
        from animakinetica.analysis.process_visualization import plot_simulation_cumulative_harvest
        return plot_simulation_cumulative_harvest(self, max_batches=max_batches, layout_kwargs=layout_kwargs)
    
    def plot_cumulative_medium_use(self, max_batches: Optional[int] = None, layout_kwargs: Optional[Dict] = None):
        """Plot cumulative batch, feed, and base medium use.

        Parameters
        ----------
        max_batches : int, optional
            Maximum number of leading batches to include. Defaults to all.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Cumulative medium-use figure.
        """
        from animakinetica.analysis.process_visualization import plot_simulation_cumulative_medium_use
        return plot_simulation_cumulative_medium_use(self, max_batches=max_batches, layout_kwargs=layout_kwargs)
    
    def plot_cumulative_base_use(self, max_batches: Optional[int] = None, layout_kwargs: Optional[Dict] = None):
        """Plot cumulative pH-control base consumption.

        Parameters
        ----------
        max_batches : int, optional
            Maximum number of leading batches to include. Defaults to all.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Cumulative base-use figure.
        """
        from animakinetica.analysis.process_visualization import plot_simulation_cumulative_base_use
        return plot_simulation_cumulative_base_use(self, max_batches=max_batches, layout_kwargs=layout_kwargs)

    def plot_cumulative_gas_use(self, max_batches: Optional[int] = None, layout_kwargs: Optional[Dict] = None):
        """Plot cumulative sparging-gas consumption.

        Parameters
        ----------
        max_batches : int, optional
            Maximum number of leading batches to include. Defaults to all.
        layout_kwargs : dict, optional
            Layout options merged into the default Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Cumulative gas-use figure.
        """
        from animakinetica.analysis.gas_visualization import plot_simulation_cumulative_gas_use
        return plot_simulation_cumulative_gas_use(self, max_batches=max_batches, layout_kwargs=layout_kwargs)
    
    def plot_profiles(self, max_batches: Optional[int] = None, layout_kwargs: Optional[Dict] = None):
        """Plot campaign profiles for tracked states and controls.

        Parameters
        ----------
        max_batches : int, optional
            Maximum number of leading batches to include. ``None`` includes all
            available batches.
        layout_kwargs : dict, optional
            Layout options merged into each default Plotly layout.

        Returns
        -------
        list of plotly.graph_objects.Figure
            Figures for the state and control variables provided by
            :func:`animakinetica.analysis.process_visualization.plot_simulation_profiles`.
        """
        from animakinetica.analysis.process_visualization import plot_simulation_profiles
        return plot_simulation_profiles(self, max_batches=max_batches, layout_kwargs=layout_kwargs)
    
    def plot_all(self, individual_batch: int = 1, max_batches_to_plot: int = 5):
        """Generate the standard set of batch and campaign Plotly figures.

        Parameters
        ----------
        individual_batch : int, default=1
            One-based batch identifier used for batch-level figures.
        max_batches_to_plot : int, default=5
            Maximum number of leading batches used for campaign-level figures.

        Returns
        -------
        dict of str to plotly.graph_objects.Figure
            Sixteen figures keyed by descriptive identifiers. The five profile
            figures are stored separately as cell-density, glucose, glutamine,
            lactate, and ammonia profiles.
        """
        # Lazy imports to avoid circular dependencies
        from animakinetica.analysis.batch_visualization import (
            plot_batch_biomass_metabolites,
            plot_batch_volume_feed
        )
        from animakinetica.analysis.gas_visualization import (
            plot_batch_gas_transfer,
            plot_simulation_cumulative_gas_use
        )
        from animakinetica.analysis.kinetic_visualization import (
            plot_simulated_NFs,
            plot_simulated_IFs,
            plot_kinetic_rates,
            plot_kinetics
        )
        from animakinetica.analysis.stoichiometric_visualization import (
            plot_stoichiometric_yields,
        )
        from animakinetica.analysis.process_visualization import (
            plot_simulation_cumulative_harvest,
            plot_simulation_cumulative_medium_use,
            plot_simulation_profiles
        )
        
        print(f"\nGenerating all plots for {self.simulation_type}...\n")
        
        figures = {}
        
        # Batch-level state variable plots (3 figures)
        print(f"[1/16] Plotting batch {individual_batch}: Biomass & Metabolites...")
        figures['batch_biomass_metabolites'] = plot_batch_biomass_metabolites(self, individual_batch)
        
        print(f"[2/16] Plotting batch {individual_batch}: Volume & Feed...")
        figures['batch_volume_feed'] = plot_batch_volume_feed(self, individual_batch)
        
        print(f"[3/16] Plotting batch {individual_batch}: Gas Transfer...")
        figures['batch_gas_transfer'] = plot_batch_gas_transfer(self, individual_batch)
        
        # Kinetic plots (4 figures)
        print(f"[4/16] Plotting batch {individual_batch}: Nutrient Factors...")
        figures['batch_nutrient_factors'] = plot_simulated_NFs(self, individual_batch)
        
        print(f"[5/16] Plotting batch {individual_batch}: Inhibition Factors...")
        figures['batch_inhibition_factors'] = plot_simulated_IFs(self, individual_batch)
        
        print(f"[6/16] Plotting batch {individual_batch}: Kinetic Rates...")
        figures['batch_kinetic_rates'] = plot_kinetic_rates(self, individual_batch)
        
        print(f"[7/16] Plotting batch {individual_batch}: Complete Kinetics Overview...")
        figures['batch_kinetics'] = plot_kinetics(self, individual_batch)
        
        # Stoichiometric plot (1 figure)
        print(f"[8/16] Plotting batch {individual_batch}: Stoichiometric Yields...")
        figures['batch_stoichiometric_yields'] = plot_stoichiometric_yields(self, individual_batch)

        # Simulation-level cumulative plots (3 figures)
        print(f"[9/16] Plotting cumulative harvest...")
        figures['cumulative_harvest'] = plot_simulation_cumulative_harvest(self, max_batches=max_batches_to_plot)
        
        print(f"[10/16] Plotting cumulative medium use...")
        figures['cumulative_medium'] = plot_simulation_cumulative_medium_use(self, max_batches=max_batches_to_plot)
        
        print(f"[11/16] Plotting cumulative gas use...")
        figures['cumulative_gas'] = plot_simulation_cumulative_gas_use(self, max_batches=max_batches_to_plot)
        
        # Profile plots (5 figures)
        print(f"[12-16/16] Plotting simulation profiles...")
        profile_figs = plot_simulation_profiles(self, max_batches=max_batches_to_plot)
        figures['profile_cell_density'] = profile_figs[0]
        figures['profile_glucose'] = profile_figs[1]
        figures['profile_glutamine'] = profile_figs[2]
        figures['profile_lactate'] = profile_figs[3]
        figures['profile_ammonia'] = profile_figs[4]
        
        print(f"\n✓ All {len(figures)} plots generated successfully!\n")
        return figures
