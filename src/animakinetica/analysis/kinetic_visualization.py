"""Visualize kinetic-model response curves and recorded batch histories.

Model-level functions evaluate an initialized
:class:`~animakinetica.models.kinetics.KineticModel` over selected
concentration ranges. Simulation-level functions plot arrays recorded in one
:class:`~animakinetica.core.simulation_data.BatchData` object. They accept a
:class:`~animakinetica.core.simulation.CellCultureSimulation`, its
:class:`~animakinetica.core.simulation_data.SimulationData`, or ``BatchData``
directly. Batch identifiers are one-based unless a ``BatchData`` object is
provided.

Simulation-level plots require detailed histories. Run the simulation with
``performance_mode=False`` (the default); performance-mode evaluations omit
the kinetic trace arrays used here.

Examples
--------
Create a model-level inhibition plot::

    from animakinetica.analysis.kinetic_visualization import plot_inhibition_profiles
    from animakinetica.core.parameters import SimulationParameters
    from animakinetica.models.kinetics import Kinetics, create_kinetic_model
    from animakinetica.models.stoichiometry import StoichiometricModel

    params = SimulationParameters.load_parameters("base_parameters")
    stoichiometry = StoichiometricModel(params)
    model = create_kinetic_model(Kinetics.STOICHIOMETRIC, params, stoichiometry)
    plot_inhibition_profiles(model).show()

Plot recorded kinetics from a completed simulation::

    from animakinetica.analysis.kinetic_visualization import plot_kinetics

    plot_kinetics(simulation, batch_number=1).show()
"""

# Imports
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple, Union, TYPE_CHECKING

from animakinetica.analysis._visualization_utils import get_batch_data

if TYPE_CHECKING:
    from animakinetica.core.simulation import CellCultureSimulation
    from animakinetica.core.simulation_data import SimulationData, BatchData
    
from animakinetica.models.kinetics import KineticModel

# Module metadata
__version__ = "1.0.0"

__all__ = [
    "plot_inhibition_profiles",
    "plot_single_inhibition",
    "plot_nutrient_factors",
    "plot_simulated_NFs",
    "plot_simulated_IFs",
    "plot_kinetic_rates",
    "plot_kinetics",
]


def plot_inhibition_profiles(
    kinetic_model: KineticModel,
    n_points: int = 200,
    show_ic_markers: bool = True,
    title: Optional[str] = None,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot all inhibitor-response profiles for a kinetic model.

    The figure contains lactate, ammonia, O₂, CO₂, and osmolality panels.
    Each profile varies one input while holding the remaining inputs at the
    reference values implemented by :func:`plot_single_inhibition`.
    
    Parameters
    ----------
    kinetic_model : KineticModel
        Initialized model whose inhibition parameters define the profiles.
    n_points : int, default=200
        Number of evaluation points per profile.
    show_ic_markers : bool, default=True
        Whether to annotate the active model's applicable IC50 or IC100 value.
    title : str, optional
        Figure title. By default, identifies the configured lactate, ammonia,
        and osmolality inhibition models.
    layout_kwargs : dict, optional
        Keyword arguments merged into the Plotly layout.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing five inhibition-factor profiles.
    """
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=3,
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    inhibitor_panels = [
        ("lactate", 1, 1),
        ("ammonia", 1, 2),
        ("o2", 1, 3),
        ("co2", 2, 1),
        ("osmolality", 2, 2),
    ]

    for inhibitor, row, col in inhibitor_panels:
        single_fig = plot_single_inhibition(
            kinetic_model=kinetic_model,
            inhibitor=inhibitor,
            n_points=n_points,
            show_ic_markers=show_ic_markers,
            layout_kwargs={"showlegend": False}
        )

        for trace in single_fig.data:
            fig.add_trace(trace, row=row, col=col)

        fig.update_xaxes(title_text=single_fig.layout.xaxis.title.text, row=row, col=col)
        fig.update_yaxes(title_text=single_fig.layout.yaxis.title.text, row=row, col=col)
    

    # Overall layout
    if title is None:
        title = f"Inhibition Factor Profiles<br><sub>Models: Lac={kinetic_model.inhibition_model_Lac}, " \
                f"Amm={kinetic_model.inhibition_model_Amm}, Osm={kinetic_model.inhibition_model_Osm}</sub>"
    
    default_layout = dict(
        title=title,
        title_x=0.5,
        height=700,
        width=1400,
        showlegend=False,
        hovermode='closest',
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    # Set all y-axes to [0, 1.1] range
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig.update_yaxes(range=[0, 1.1], row=row, col=col)
    
    return fig


def plot_single_inhibition(
    kinetic_model: KineticModel,
    inhibitor: str,
    concentration_range: Optional[Tuple[float, float]] = None,
    n_points: int = 200,
    show_ic_markers: bool = True,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot one inhibitor-response curve for a kinetic model.

    The plotted inhibitor varies over ``concentration_range``. All other
    inputs to :meth:`KineticModel.calculate_inhibition_factors` are held at
    the fixed reference values used by this function. The IC annotation, when
    enabled, reflects the active inhibition-model form.
    
    Parameters
    ----------
    kinetic_model : KineticModel
        Initialized model whose inhibition parameters define the profile.
    inhibitor : str
        Inhibitor to plot: ``"lactate"``, ``"ammonia"``, ``"O2"``,
        ``"CO2"``, or ``"osmolality"``. Case is ignored.
    concentration_range : tuple of (float, float), optional
        Inclusive lower and nominal upper plotting limits. The displayed range
        extends 20% beyond the nominal upper limit for concentration profiles.
        If omitted, a model-specific range is selected.
    n_points : int, default=200
        Number of evaluation points.
    show_ic_markers : bool, default=True
        Whether to annotate the applicable IC50 or IC100 value.
    layout_kwargs : dict, optional
        Keyword arguments merged into the Plotly layout.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing the selected inhibition-factor profile.

    Raises
    ------
    ValueError
        If ``inhibitor`` is not supported.
    """
    def _get_selected_ic(model_name: str, ic50: Optional[float], ic100: Optional[float]) -> Tuple[Optional[str], Optional[float]]:
        """Return the IC marker that matches the selected inhibition model."""
        if model_name in ["linear", "power_decay"]:
            return "IC100", ic100
        if model_name in ["exponential", "hyperbolic"]:
            return "IC50", ic50
        return None, None

    
    inhibitor = inhibitor.lower()
    selected_ic_type: Optional[str] = None
    selected_ic_value: Optional[float] = None
    ylabel = "Inhibition Factor"
    
    if inhibitor == 'lactate':
        if concentration_range is None:
            concentration_range = (0, kinetic_model.IC100_Lac)
        
        conc = _get_concentration_range(concentration_range[0], concentration_range[1], n_points)
        # Use centralized method with baseline values for other inhibitors
        IF = np.array([
            kinetic_model.calculate_inhibition_factors(
                C_Lac=c, C_Amm=0.0, C_O2=0.2, C_CO2=10.0, Osm=kinetic_model.Osm_baseline_inhibition
            )['IF_lactate'] for c in conc
        ])
        
        xlabel = "Lactate Concentration (mmol/L)"
        ylabel = "Lactate Inhibition Factor"
        title = f"Lactate Inhibition Profile ({kinetic_model.inhibition_model_Lac} model)"
        color = '#E74C3C'
        selected_ic_type, selected_ic_value = _get_selected_ic(
            kinetic_model.inhibition_model_Lac,
            kinetic_model.IC50_Lac,
            kinetic_model.IC100_Lac
        )
        
    elif inhibitor == 'ammonia':
        if concentration_range is None:
            concentration_range = (0, kinetic_model.IC100_Amm)
        
        conc = _get_concentration_range(concentration_range[0], concentration_range[1], n_points)
        # Use centralized method with baseline values for other inhibitors
        IF = np.array([
            kinetic_model.calculate_inhibition_factors(
                C_Lac=0.0, C_Amm=c, C_O2=0.2, C_CO2=10.0, Osm=kinetic_model.Osm_baseline_inhibition
            )['IF_ammonia'] for c in conc
        ])
        
        xlabel = "Ammonia Concentration (mmol/L)"
        ylabel = "Ammonia Inhibition Factor"
        title = f"Ammonia Inhibition Profile ({kinetic_model.inhibition_model_Amm} model)"
        color = '#3498DB'
        selected_ic_type, selected_ic_value = _get_selected_ic(
            kinetic_model.inhibition_model_Amm,
            kinetic_model.IC50_Amm,
            kinetic_model.IC100_Amm
        )
        
    elif inhibitor in ['o2', 'oxygen']:
        if concentration_range is None:
            concentration_range = (0, kinetic_model.IC50_O2 * 3)
        
        conc = _get_concentration_range(concentration_range[0], concentration_range[1], n_points)
        # Use centralized method with baseline values for other inhibitors
        IF = np.array([
            kinetic_model.calculate_inhibition_factors(
                C_Lac=0.0, C_Amm=0.0, C_O2=c, C_CO2=10.0, Osm=kinetic_model.Osm_baseline_inhibition
            )['IF_O2'] for c in conc
        ])
        
        xlabel = "O₂ Concentration (mmol/L)"
        ylabel = "O₂ Inhibition Factor"
        title = f"O₂ Inhibition Profile (Hypoxia, n={kinetic_model.n_O2})"
        color = '#2ECC71'
        selected_ic_type, selected_ic_value = 'IC50', kinetic_model.IC50_O2
        
    elif inhibitor in ['co2', 'carbon dioxide']:
        if concentration_range is None:
            concentration_range = (0, kinetic_model.IC100_CO2)
        
        conc = _get_concentration_range(concentration_range[0], concentration_range[1], n_points)
        # Use centralized method with baseline values for other inhibitors
        IF = np.array([
            kinetic_model.calculate_inhibition_factors(
                C_Lac=0.0, C_Amm=0.0, C_O2=0.2, C_CO2=c, Osm=kinetic_model.Osm_baseline_inhibition
            )['IF_CO2'] for c in conc
        ])
        
        xlabel = "CO₂ Concentration (mmol/L)"
        ylabel = "CO₂ Inhibition Factor"
        title = f"CO₂ Inhibition Profile (n={kinetic_model.n_CO2})"
        color = '#9B59B6'
        selected_ic_type, selected_ic_value = 'IC100', kinetic_model.IC100_CO2
        
    elif inhibitor == 'osmolality':
        Osm_baseline = kinetic_model.Osm_baseline_inhibition
        if concentration_range is None:
            concentration_range = (
                max(0, Osm_baseline - kinetic_model.IC100_Osm_effective),
                Osm_baseline + kinetic_model.IC100_Osm_effective
            )
        
        conc = np.linspace(concentration_range[0], concentration_range[1], n_points)
        # Use centralized method with baseline values for other inhibitors
        IF = np.array([
            kinetic_model.calculate_inhibition_factors(
                C_Lac=0.0, C_Amm=0.0, C_O2=0.2, C_CO2=10.0, Osm=osm
            )['IF_osmolality'] for osm in conc
        ])
        
        xlabel = "Osmolality (mOsm)"
        ylabel = "Osmolality Inhibition Factor"
        title = f"Osmolality Inhibition Profile ({kinetic_model.inhibition_model_Osm} model)"
        color = '#F39C12'
        osm_ic_type, osm_ic_value_effective = _get_selected_ic(
            kinetic_model.inhibition_model_Osm,
            kinetic_model.IC50_Osm_effective,
            kinetic_model.IC100_Osm_effective
        )
        selected_ic_type = osm_ic_type
        selected_ic_value = (
            Osm_baseline + osm_ic_value_effective
            if osm_ic_value_effective is not None else None
        )
    else:
        raise ValueError(f"Unknown inhibitor type: {inhibitor}. Valid options: 'lactate', 'ammonia', 'O2', 'CO2', 'osmolality'")
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=conc, y=IF, mode='lines',
                   line=dict(color=color, width=3),
                   hovertemplate='%{x:.2f}<br>IF: %{y:.3f}<extra></extra>')
    )
    
    # Add one IC marker matching the active inhibition model
    if show_ic_markers:
        if selected_ic_value is not None and selected_ic_type is not None:
            idx = np.argmin(np.abs(conc - selected_ic_value))
            fig.add_trace(
                go.Scatter(
                    x=[selected_ic_value], y=[IF[idx]],
                    mode='markers+text',
                    marker=dict(size=10, color='red', symbol='circle'),
                    text=[f'{selected_ic_type}: {selected_ic_value:.2f}'],
                    textposition='top left',
                    textfont=dict(size=11, color='red'),
                    hovertemplate=f'{selected_ic_type}: {selected_ic_value:.2f}<br>IF: {IF[idx]:.3f}<extra></extra>',
                    showlegend=False
                )
            )
    
    # Layout
    default_layout = dict(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        yaxis_range=[0, 1.1],
        height=500,
        width=700,
        template='plotly_white',
        hovermode='x',
        showlegend=False
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_nutrient_factors(
    kinetic_model: KineticModel,
    n_points: int = 200,
    show_k_markers: bool = True,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot Monod nutrient-factor profiles for glucose, glutamine, and O₂.

    Each panel varies one nutrient while holding the other nutrients at
    non-limiting reference concentrations. Total amino acids are fixed at a
    non-limiting concentration because their factor is not displayed.
    
    Parameters
    ----------
    kinetic_model : KineticModel
        Initialized model whose Monod constants define the profiles.
    n_points : int, default=200
        Number of evaluation points per nutrient.
    show_k_markers : bool, default=True
        Whether to annotate each half-saturation constant.
    layout_kwargs : dict, optional
        Keyword arguments merged into the Plotly layout.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing glucose, glutamine, and dissolved-oxygen profiles.
    """
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Glucose', 'Glutamine', 'O₂'),
        horizontal_spacing=0.12
    )
    
    # Define nutrient configurations
    nutrients = [
        {'name': 'Glucose', 'K': kinetic_model.K_Glc, 'color': '#E74C3C', 'marker_color': 'darkred', 'col': 1, 'unit': 'mmol/L', 'nf_key': 'NF_glucose', 'baseline': 25.0},
        {'name': 'Glutamine', 'K': kinetic_model.K_Gln, 'color': '#3498DB', 'marker_color': 'darkblue', 'col': 2, 'unit': 'mmol/L', 'nf_key': 'NF_glutamine', 'baseline': 4.0},
        {'name': 'O₂', 'K': kinetic_model.K_O2, 'color': '#2ECC71', 'marker_color': 'darkgreen', 'col': 3, 'unit': 'mmol/L', 'nf_key': 'NF_O2', 'baseline': 0.2}
    ]
    
    # Plot each nutrient factor
    for nutrient in nutrients:
        C_range = _get_concentration_range(0, nutrient['K'] * 5, n_points)
        
        # Calculate nutrient factors using centralized method from kinetic model
        NF = np.zeros(n_points)
        for i, c in enumerate(C_range):
            # Set baseline concentrations for other nutrients, vary the current one
            if nutrient['name'] == 'Glucose':
                nf_dict = kinetic_model.calculate_nutrient_factors(
                    C_Glc=c, C_Gln=4.0, C_AA=100.0, C_O2=0.2
                )
            elif nutrient['name'] == 'Glutamine':
                nf_dict = kinetic_model.calculate_nutrient_factors(
                    C_Glc=25.0, C_Gln=c, C_AA=100.0, C_O2=0.2
                )
            else:  # O2
                nf_dict = kinetic_model.calculate_nutrient_factors(
                    C_Glc=25.0, C_Gln=4.0, C_AA=100.0, C_O2=c
                )
            NF[i] = nf_dict[nutrient['nf_key']]
        
        fig.add_trace(
            go.Scatter(x=C_range, y=NF, mode='lines',
                       line=dict(color=nutrient['color'], width=3),
                       name=nutrient['name']),
            row=1, col=nutrient['col']
        )
        
        if show_k_markers:
            idx_k = np.argmin(np.abs(C_range - nutrient['K']))
            fig.add_trace(
                go.Scatter(x=[nutrient['K']], y=[NF[idx_k]],
                          mode='markers+text',
                          marker=dict(size=10, color=nutrient['marker_color'], symbol='diamond'),
                          text=['K'],
                          textposition='top center',
                          hovertemplate=f"K: {nutrient['K']:.3f} {nutrient['unit']}<br>NF: {NF[idx_k]:.3f}<extra></extra>",
                          showlegend=False),
                row=1, col=nutrient['col']
            )
        
        fig.update_xaxes(title_text=f"{nutrient['name']} ({nutrient['unit']})", row=1, col=nutrient['col'])
        fig.update_yaxes(title_text="Nutrient Factor", row=1, col=nutrient['col'])
    
    # Layout
    default_layout = dict(
        title="Nutrient Factor Profiles (Monod Kinetics)",
        title_x=0.5,
        height=400,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    # Set all y-axes to [0, 1.1]
    for col in [1, 2, 3]:
        fig.update_yaxes(range=[0, 1.1], row=1, col=col)
    
    return fig


# Helper functions
def _get_concentration_range(c_min: float, c_max: float, n_points: int) -> np.ndarray:
    """Generate an evenly spaced plotting range.

    Parameters
    ----------
    c_min : float
        Lower bound of the range.
    c_max : float
        Nominal upper bound before extension.
    n_points : int
        Number of points in the returned array.

    Returns
    -------
    numpy.ndarray
        Evenly spaced values from ``c_min`` through ``1.2 * c_max``.
    """
    return np.linspace(c_min, c_max * 1.2, n_points)


# Simulation-based visualization functions
def plot_simulated_NFs(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot recorded nutrient factors for one simulation batch.

    Plots the individual glucose, glutamine, and dissolved-oxygen factors plus
    their recorded combined product. The amino-acid factor is not shown.
    
    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Completed result source containing detailed kinetic histories.
    batch_number : int, default=1
        One-based batch identifier. Ignored when ``source`` is ``BatchData``.
    layout_kwargs : dict, optional
        Keyword arguments merged into the Plotly layout.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing nutrient-factor time courses.

    Raises
    ------
    ValueError
        If the requested batch is absent.
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Extract time and NF data
    t = batch_data.t_points
    NF_glucose = batch_data.NF_glucose_list
    NF_glutamine = batch_data.NF_glutamine_list
    NF_O2 = batch_data.NF_O2_list
    NF_combined = batch_data.NF_list
    
    # Create figure
    fig = go.Figure()
    
    # Plot individual nutrient factors
    fig.add_trace(
        go.Scatter(x=t, y=NF_glucose, mode='lines',
                   name='Glucose NF',
                   line=dict(color='#E74C3C', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=NF_glutamine, mode='lines',
                   name='Glutamine NF',
                   line=dict(color='#3498DB', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=NF_O2, mode='lines',
                   name='O₂ NF',
                   line=dict(color='#2ECC71', width=2))
    )
    
    # Plot combined nutrient factor
    fig.add_trace(
        go.Scatter(x=t, y=NF_combined, mode='lines',
                   name='Combined NF',
                   line=dict(color='#34495E', width=3, dash='dash'))
    )
    
    # Layout
    default_layout = dict(
        title=f"Nutrient Factors - Batch {batch_number}",
        xaxis_title="Time (h)",
        yaxis_title="Nutrient Factor",
        yaxis_range=[0, 1.1],
        height=500,
        width=900,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_simulated_IFs(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot recorded inhibition factors for one simulation batch.

    Plots lactate, ammonia, O₂, CO₂, and osmolality factors plus their
    recorded combined product.
    
    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Completed result source containing detailed kinetic histories.
    batch_number : int, default=1
        One-based batch identifier. Ignored when ``source`` is ``BatchData``.
    layout_kwargs : dict, optional
        Keyword arguments merged into the Plotly layout.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing inhibition-factor time courses.

    Raises
    ------
    ValueError
        If the requested batch is absent.
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Extract time and IF data
    t = batch_data.t_points
    IF_lactate = batch_data.IF_lactate_list
    IF_ammonia = batch_data.IF_ammonia_list
    IF_O2 = batch_data.IF_O2_list
    IF_CO2 = batch_data.IF_CO2_list
    IF_osmolality = batch_data.IF_osmolality_list
    IF_combined = batch_data.IF_list
    
    # Create figure
    fig = go.Figure()
    
    # Plot individual inhibition factors
    fig.add_trace(
        go.Scatter(x=t, y=IF_lactate, mode='lines',
                   name='Lactate IF',
                   line=dict(color='#E74C3C', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=IF_ammonia, mode='lines',
                   name='Ammonia IF',
                   line=dict(color='#3498DB', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=IF_O2, mode='lines',
                   name='O₂ IF (Hypoxia)',
                   line=dict(color='#2ECC71', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=IF_CO2, mode='lines',
                   name='CO₂ IF',
                   line=dict(color='#9B59B6', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=IF_osmolality, mode='lines',
                   name='Osmolality IF',
                   line=dict(color='#F39C12', width=2))
    )
    
    # Plot combined inhibition factor
    fig.add_trace(
        go.Scatter(x=t, y=IF_combined, mode='lines',
                   name='Combined IF',
                   line=dict(color='#34495E', width=3, dash='dash'))
    )
    
    # Layout
    default_layout = dict(
        title=f"Inhibition Factors - Batch {batch_number}",
        xaxis_title="Time (h)",
        yaxis_title="Inhibition Factor",
        yaxis_range=[0, 1.1],
        height=500,
        width=900,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_kinetic_rates(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot recorded growth and specific-rate time courses for one batch.

    Uptake rates are multiplied by $10^9$ and negated so that substrate and
    O₂ consumption plot as positive values. Product formation retains its
    recorded sign. The rate panels therefore use mmol/$10^9$ cells/h.
    
    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Completed result source containing detailed kinetic histories.
    batch_number : int, default=1
        One-based batch identifier. Ignored when ``source`` is ``BatchData``.
    layout_kwargs : dict, optional
        Keyword arguments merged into the Plotly layout.
    
    Returns
    -------
    plotly.graph_objects.Figure
        Figure with growth, substrate, metabolite, and gas-exchange panels.
        
    Returns
    -------
    Raises
    ------
    ValueError
        If the requested batch is absent.
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Extract time and kinetic rate data
    t = batch_data.t_points
    mu = batch_data.mu_list
    q_Glc = batch_data.q_Glc_list * 1e9  # Convert to mmol/1e9 cells/h
    q_Gln = batch_data.q_Gln_list * 1e9  # Convert to mmol/1e9 cells/h
    q_Lac = batch_data.q_Lac_list * 1e9  # Convert to mmol/1e9 cells/h
    q_Amm = batch_data.q_Amm_list * 1e9  # Convert to mmol/1e9 cells/h
    q_O2 = batch_data.q_O2_list * 1e9  # Convert to mmol/1e9 cells/h
    q_CO2 = batch_data.q_CO2_list * 1e9  # Convert to mmol/1e9 cells/h
    
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Growth Rate (μ)', 'Substrate Consumption', 'Metabolite Production', 'Gas Exchange'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # 1. Growth rate
    fig.add_trace(
        go.Scatter(x=t, y=mu, mode='lines',
                   name='μ',
                   line=dict(color='#34495E', width=3)),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Time (h)", row=1, col=1)
    fig.update_yaxes(title_text="μ (1/h)", row=1, col=1)
    
    # 2. Substrate consumption (negative rates)
    fig.add_trace(
        go.Scatter(x=t, y=-q_Glc, mode='lines',
                   name='Glucose',
                   line=dict(color='#E74C3C', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=t, y=-q_Gln, mode='lines',
                   name='Glutamine',
                   line=dict(color='#3498DB', width=2)),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Time (h)", row=1, col=2)
    fig.update_yaxes(title_text="q (mmol/1e9 cells/h)", row=1, col=2)
    
    # 3. Metabolite production (positive rates)
    fig.add_trace(
        go.Scatter(x=t, y=q_Lac, mode='lines',
                   name='Lactate',
                   line=dict(color='#E67E22', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=q_Amm, mode='lines',
                   name='Ammonia',
                   line=dict(color='#95A5A6', width=2)),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
    fig.update_yaxes(title_text="q (mmol/1e9 cells/h)", row=2, col=1)
    
    # 4. Gas exchange
    fig.add_trace(
        go.Scatter(x=t, y=-q_O2, mode='lines',
                   name='O₂ Uptake',
                   line=dict(color='#2ECC71', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=t, y=q_CO2, mode='lines',
                   name='CO₂ Production',
                   line=dict(color='#9B59B6', width=2)),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Time (h)", row=2, col=2)
    fig.update_yaxes(title_text="q (mmol/1e9 cells/h)", row=2, col=2)
    
    # Overall layout
    default_layout = dict(
        title=f"Kinetic Rates - Batch {batch_number}",
        height=700,
        width=1200,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_kinetics(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot a consolidated kinetic overview for one simulation batch.

    The figure combines nutrient and inhibition factors, growth rate, specific
    rates, their product (NF × IF), and the lag term when it was recorded.
    Specific-rate panels use recorded rates multiplied by $10^9$; uptake rates
    are negated to display consumption as positive values.
    
    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Completed result source containing detailed kinetic histories.
    batch_number : int, default=1
        One-based batch identifier. Ignored when ``source`` is ``BatchData``.
    layout_kwargs : dict, optional
        Keyword arguments merged into the Plotly layout.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing the consolidated kinetic time courses.

    Raises
    ------
    ValueError
        If the requested batch is absent.
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Extract all data
    t = batch_data.t_points
    
    # Create subplots: 3 rows, 3 columns
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Nutrient Factors', 'Inhibition Factors', 'Growth Rate (μ)',
            'Substrate Consumption', 'Metabolite Production', 'Gas Exchange',
            'Combined NF × IF', 'Lag Term', ''
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10
    )
    
    # Row 1, Col 1: Nutrient Factors
    fig.add_trace(go.Scatter(x=t, y=batch_data.NF_glucose_list, mode='lines', name='Glc NF', 
                            line=dict(color='#E74C3C', width=1.5), legendgroup='NF'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=batch_data.NF_glutamine_list, mode='lines', name='Gln NF',
                            line=dict(color='#3498DB', width=1.5), legendgroup='NF'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=batch_data.NF_O2_list, mode='lines', name='O₂ NF',
                            line=dict(color='#2ECC71', width=1.5), legendgroup='NF'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=batch_data.NF_list, mode='lines', name='Combined NF',
                            line=dict(color='#34495E', width=2, dash='dash'), legendgroup='NF'), row=1, col=1)
    fig.update_yaxes(title_text="NF", range=[0, 1.1], row=1, col=1)
    
    # Row 1, Col 2: Inhibition Factors
    fig.add_trace(go.Scatter(x=t, y=batch_data.IF_lactate_list, mode='lines', name='Lac IF',
                            line=dict(color='#E74C3C', width=1.5), legendgroup='IF'), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=batch_data.IF_ammonia_list, mode='lines', name='Amm IF',
                            line=dict(color='#3498DB', width=1.5), legendgroup='IF'), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=batch_data.IF_O2_list, mode='lines', name='O₂ IF',
                            line=dict(color='#2ECC71', width=1.5), legendgroup='IF'), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=batch_data.IF_CO2_list, mode='lines', name='CO₂ IF',
                            line=dict(color='#9B59B6', width=1.5), legendgroup='IF'), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=batch_data.IF_osmolality_list, mode='lines', name='Osm IF',
                            line=dict(color='#F39C12', width=1.5), legendgroup='IF'), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=batch_data.IF_list, mode='lines', name='Combined IF',
                            line=dict(color='#34495E', width=2, dash='dash'), legendgroup='IF'), row=1, col=2)
    fig.update_yaxes(title_text="IF", range=[0, 1.1], row=1, col=2)
    
    # Row 1, Col 3: Growth Rate
    fig.add_trace(go.Scatter(x=t, y=batch_data.mu_list, mode='lines', name='μ',
                            line=dict(color='#34495E', width=2), showlegend=False), row=1, col=3)
    fig.update_yaxes(title_text="μ (1/h)", row=1, col=3)
    
    # Row 2, Col 1: Substrate Consumption
    fig.add_trace(go.Scatter(x=t, y=-batch_data.q_Glc_list*1e9, mode='lines', name='q_Glc',
                            line=dict(color='#E74C3C', width=1.5), legendgroup='rates'), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=-batch_data.q_Gln_list*1e9, mode='lines', name='q_Gln',
                            line=dict(color='#3498DB', width=1.5), legendgroup='rates'), row=2, col=1)
    fig.update_yaxes(title_text="q (mmol/10e9 cells/h)", row=2, col=1)
    
    # Row 2, Col 2: Metabolite Production
    fig.add_trace(go.Scatter(x=t, y=batch_data.q_Lac_list*1e9, mode='lines', name='q_Lac',
                            line=dict(color='#E67E22', width=1.5), legendgroup='rates'), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=batch_data.q_Amm_list*1e9, mode='lines', name='q_Amm',
                            line=dict(color='#95A5A6', width=1.5), legendgroup='rates'), row=2, col=2)
    fig.update_yaxes(title_text="q (mmol/10e9 cells/h)", row=2, col=2)
    
    # Row 2, Col 3: Gas Exchange
    fig.add_trace(go.Scatter(x=t, y=-batch_data.q_O2_list*1e9, mode='lines', name='q_O₂',
                            line=dict(color='#2ECC71', width=1.5), legendgroup='rates'), row=2, col=3)
    fig.add_trace(go.Scatter(x=t, y=batch_data.q_CO2_list*1e9, mode='lines', name='q_CO₂',
                            line=dict(color='#9B59B6', width=1.5), legendgroup='rates'), row=2, col=3)
    fig.update_yaxes(title_text="q (mmol/10e9 cells/h)", row=2, col=3)
    
    # Row 3, Col 1: Combined NF × IF
    combined = batch_data.NF_list * batch_data.IF_list
    fig.add_trace(go.Scatter(x=t, y=batch_data.NF_list, mode='lines', name='NF',
                            line=dict(color='#2ECC71', width=1.5), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=batch_data.IF_list, mode='lines', name='IF',
                            line=dict(color='#E74C3C', width=1.5), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=combined, mode='lines', name='NF × IF',
                            line=dict(color='#34495E', width=2, dash='dash'), showlegend=False), row=3, col=1)
    fig.update_yaxes(title_text="Factor", range=[0, 1.1], row=3, col=1)
    
    # Row 3, Col 2: Lag Term (if available)
    if batch_data.lag_term_list is not None:
        fig.add_trace(go.Scatter(x=t, y=batch_data.lag_term_list, mode='lines', name='Lag',
                                line=dict(color='#7F8C8D', width=2), showlegend=False), row=3, col=2)
        fig.update_yaxes(title_text="Lag Term", range=[0, 1.1], row=3, col=2)
    
    # Update all x-axes to show time
    for row in [1, 2, 3]:
        for col in [1, 2, 3]:
            fig.update_xaxes(title_text="Time (h)", row=row, col=col)
    
    # Overall layout
    default_layout = dict(
        title=f"Complete Kinetics Overview - Batch {batch_number}",
        height=1000,
        width=1400,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig
