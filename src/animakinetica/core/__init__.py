"""Public APIs for simulation setup, execution, and result handling.

The public core API provides :class:`CellCultureSimulation` for configuring
and running simulations, :class:`SimulationParameters` for YAML-backed model
parameters, and :class:`SimulationData` / :class:`BatchData` for analysis and
export. Use :func:`get_script_export_dir` when campaign scripts should keep
their exports next to the calling script.
"""

__version__ = "1.0.0"
__author__ = "Ben Tumulero"

from animakinetica.core.simulation import CellCultureSimulation
from animakinetica.core.parameters import (
    Parameter,
    ParameterIO,
    ParameterValidationError,
    SimulationParameters,
    list_parameter_sets,
)
from animakinetica.core.simulation_data import SimulationData, BatchData, get_script_export_dir

__all__ = [
    'CellCultureSimulation',
    'Parameter',
    'ParameterIO',
    'ParameterValidationError',
    'SimulationParameters',
    'list_parameter_sets',
    'SimulationData',
    'BatchData',
    'get_script_export_dir',
]