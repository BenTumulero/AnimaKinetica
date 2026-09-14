"""Analysis, optimization, sensitivity, and visualization utilities.

The package exposes the public analysis API for AnimaKinetica. Plotting
functions accept either a :class:`CellCultureSimulation`, a
:class:`SimulationData` instance, or (for batch-level plots) a
:class:`BatchData` instance. Batch numbers use the public one-based convention.

The plotting functions return Plotly figures and do not display or save them;
call ``fig.show()`` or ``fig.write_html(...)`` in the calling application.
"""

__version__ = "1.0.0"


from animakinetica.analysis.optimization import OptimizationManager
from animakinetica.analysis.sensitivity import SensitivityAnalyzer

# Kinetic visualization (nutrient factors, inhibition factors, kinetic rates)
from animakinetica.analysis.kinetic_visualization import (
    plot_inhibition_profiles,
    plot_single_inhibition,
    plot_nutrient_factors,
    plot_simulated_NFs,
    plot_simulated_IFs,
    plot_kinetic_rates,
    plot_kinetics,
)

# Batch-level visualization (state variables: biomass, metabolites, volume, feed)
from animakinetica.analysis.batch_visualization import (
    plot_batch_biomass_metabolites,
    plot_batch_volume_feed,
    plot_batch_pH_control,
)

# Gas transfer visualization (O2, CO2, stirring, gas flow)
from animakinetica.analysis.gas_visualization import (
    plot_batch_gas_transfer,
    plot_simulation_cumulative_gas_use,
)

# Process-level visualization (cumulative metrics across batches)
from animakinetica.analysis.process_visualization import (
    plot_simulation_cumulative_harvest,
    plot_simulation_cumulative_medium_use,
    plot_simulation_cumulative_base_use,
    plot_simulation_profiles,
)

# Stoichiometric visualization (yield coefficient trajectories)
from animakinetica.analysis.stoichiometric_visualization import (
    plot_stoichiometric_yields,
)

__all__ = [
    # Analysis tools
    'OptimizationManager',
    'SensitivityAnalyzer',
    
    # Kinetic visualization
    'plot_inhibition_profiles',
    'plot_single_inhibition',
    'plot_nutrient_factors',
    'plot_simulated_NFs',
    'plot_simulated_IFs',
    'plot_kinetic_rates',
    'plot_kinetics',
    
    # Batch visualization
    'plot_batch_biomass_metabolites',
    'plot_batch_volume_feed',
    'plot_batch_pH_control',
    
    # Gas visualization
    'plot_batch_gas_transfer',
    'plot_simulation_cumulative_gas_use',
    
    # Process visualization
    'plot_simulation_cumulative_harvest',
    'plot_simulation_cumulative_medium_use',
    'plot_simulation_cumulative_base_use',
    'plot_simulation_profiles',

    # Stoichiometric visualization
    'plot_stoichiometric_yields',
]