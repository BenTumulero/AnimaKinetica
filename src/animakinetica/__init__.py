"""AnimaKinetica animal-cell-culture simulation framework.

The package-level API provides the primary simulation class and the most common
configuration enums. Import specialised analysis, result-handling, controller,
and visualization APIs from :mod:`animakinetica.analysis`,
:mod:`animakinetica.core`, or :mod:`animakinetica.models`.
"""

__version__ = "1.0.0"
__author__ = "Ben Tumulero"

from animakinetica.core.simulation import CellCultureSimulation
from animakinetica.core.parameters import SimulationParameters
from animakinetica.models.feed_strategies import FeedStrategy
from animakinetica.models.gas_controllers import GasController
from animakinetica.models.kinetics import Kinetics
from animakinetica.models.odesystems import OperatingMode
from animakinetica.models.perfusion_strategies import PerfusionStrategy
from animakinetica.models.stoichiometry import StoichiometryMode

__all__ = [
    "CellCultureSimulation",
    "SimulationParameters",
    "Kinetics",
    "OperatingMode",
    "FeedStrategy",
    "PerfusionStrategy",
    "GasController",
    "StoichiometryMode",
]