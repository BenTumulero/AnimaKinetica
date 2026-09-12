"""Public model-selection APIs for the cell-culture simulation engine.

The package exposes enums and factory functions for kinetic, operating-mode,
gas-control, feed, perfusion, and stoichiometric models. Import concrete model
classes and terminal event functions from their respective submodules when
customisation beyond these public selection APIs is required.
"""

__version__ = "1.0.0"
__author__ = "Ben Tumulero"

from animakinetica.models.kinetics import Kinetics, create_kinetic_model
from animakinetica.models.odesystems import OperatingMode, create_ode_systems
from animakinetica.models.stoichiometry import StoichiometricModel, StoichiometryMode
from animakinetica.models.feed_strategies import FeedStrategy, create_feed_model
from animakinetica.models.perfusion_strategies import (
    PerfusionStrategy,
    create_perfusion_strategy,
)
from animakinetica.models.gas_controllers import (
    GasController,
    create_gas_controller_model,
)
from animakinetica.models.pH_controller import pHControllerModel

__all__ = [
    # Kinetics
    'Kinetics',
    'create_kinetic_model',
    # Operating modes
    'OperatingMode',
    'create_ode_systems',
    # Stoichiometry
    'StoichiometricModel',
    'StoichiometryMode',
    # Feed control
    'FeedStrategy',
    'create_feed_model',
    # Perfusion control
    'PerfusionStrategy',
    'create_perfusion_strategy',
    # Gas and pH control
    'GasController',
    'create_gas_controller_model',
    'pHControllerModel',
]