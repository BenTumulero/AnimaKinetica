"""Visualize campaign-level material use, harvest, and batch profiles.

The public functions accept a
:class:`~animakinetica.core.simulation.CellCultureSimulation` or its
:class:`~animakinetica.core.simulation_data.SimulationData`. They join the
stored trajectories from successive
:class:`~animakinetica.core.simulation_data.BatchData` records onto an elapsed
campaign-time axis without inserting offline turnaround gaps.

The cumulative plots use incremental process-boundary arrays and reconstruct
running totals with :func:`numpy.cumsum`. Profile plots show stored ODE states
and, when detailed logging is available, controller-derived quantities.
Consequently, profile figures for dissolved CO2 and osmolality are omitted when
the corresponding detailed trace is unavailable, as can occur when
``performance_mode=True``.

Functions
---------
plot_simulation_cumulative_harvest
    Plot cumulative wet-cell biomass leaving as harvest and harvest volume.
plot_simulation_cumulative_medium_use
    Plot cumulative batch medium, feed medium, base solution, and their sum.
plot_simulation_cumulative_base_use
    Plot cumulative base-solution addition used for pH control.
plot_simulation_profiles
    Plot available state and controller profiles for successive batches.

Examples
--------
>>> sim.run_simulation(number_of_batches=10)
>>> fig = plot_simulation_cumulative_harvest(sim, max_batches=5)
>>> fig.show()
>>> profiles = plot_simulation_profiles(sim.data, max_batches=5)
>>> profiles[0].show()
"""

# Imports
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, Union, List, TYPE_CHECKING

from animakinetica.analysis._visualization_utils import get_simulation_data

if TYPE_CHECKING:
    from animakinetica.core.simulation import CellCultureSimulation
    from animakinetica.core.simulation_data import SimulationData

# Module metadata
__version__ = "1.0.0"

__all__ = [
    "plot_simulation_cumulative_harvest",
    "plot_simulation_cumulative_medium_use",
    "plot_simulation_cumulative_base_use",
    "plot_simulation_profiles",
]


def plot_simulation_cumulative_harvest(
    source: Union['CellCultureSimulation', 'SimulationData'],
    max_batches: Optional[int] = None,
    layout_kwargs: Optional[Dict] = None,
) -> go.Figure:
    """Plot cumulative harvested wet-cell biomass and harvest volume.

    The biomass trace is calculated from the incremental ``BatchData.cells_out``
    array after subtracting ``BatchData.cells_lost``. It therefore excludes
    non-harvested cell losses, such as a cell-separator leak. Cell counts are
    converted to kg using ``parameters.cell_properties.wet_cell_mass``.
    
    Parameters
    ----------
    source : CellCultureSimulation or SimulationData
        Simulation results to visualize.
    max_batches : int, optional
        Maximum number of leading batches to include. If ``None``, include all
        available batches.
    layout_kwargs : dict, optional
        Keyword arguments that override the default Plotly layout.
    Returns
    -------
    plotly.graph_objects.Figure
        Figure with biomass on the primary y-axis and volume on the secondary
        y-axis.
        
    Examples
    --------
    >>> from animakinetica.core.simulation import CellCultureSimulation
    >>> from animakinetica.analysis.process_visualization import plot_simulation_cumulative_harvest
    >>> sim = CellCultureSimulation(params, Kinetics.EXCESS, OperatingMode.FEDBATCH)
    >>> sim.run_simulation(number_of_batches=10)
    >>> fig = plot_simulation_cumulative_harvest(sim, max_batches=5)
    >>> fig.show()
    """
    # Get simulation data
    sim_data = get_simulation_data(source)
    
    if sim_data.parameters is None:
        raise ValueError("Parameters not available in simulation data")
    
    wet_cell_mass = float(sim_data.parameters.cell_properties.wet_cell_mass)
    n_batches = len(sim_data.batches) if max_batches is None else min(max_batches, len(sim_data.batches))
    
    # Build continuous time series across all batches
    all_times = []
    all_biomass = []
    all_volume = []
    
    t_offset = 0
    cumul_biomass_offset = 0
    cumul_volume_offset = 0
    
    for batch_idx in range(n_batches):
        batch = sim_data.batches[batch_idx]
        
        # Get time points and adjust for offset
        t_batch = np.asarray(batch.t_points, dtype=float) + t_offset
        
        # Get incremental harvest arrays and convert to cumulative time-series.
        # cells_out and cells_lost are incremental per-timestep; cumsum to get running totals.
        # cells_out is the total boundary biomass; exclude non-harvested losses
        # (e.g. perfusion separator leak) so the plot reflects harvested product.
        cells_harvested = np.cumsum(np.asarray(batch.cells_out, dtype=float))
        if batch.cells_lost is not None:
            cells_harvested = cells_harvested - np.cumsum(np.asarray(batch.cells_lost, dtype=float))
        biomass_batch = cells_harvested * wet_cell_mass / 1000  # Convert to kg
        volume_batch = np.cumsum(np.asarray(batch.harvest_volume, dtype=float))
        
        # Add offset from previous batches
        biomass_batch = biomass_batch + cumul_biomass_offset
        volume_batch = volume_batch + cumul_volume_offset
        
        # Append to overall arrays
        all_times.extend(t_batch)
        all_biomass.extend(biomass_batch)
        all_volume.extend(volume_batch)
        
        # Update offsets for next batch (np.sum of incremental = batch total)
        t_offset = t_batch[-1]
        cumul_biomass_offset = biomass_batch[-1]
        cumul_volume_offset = volume_batch[-1]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Cumulative biomass on primary y-axis (left)
    fig.add_trace(
        go.Scatter(x=all_times, y=all_biomass, mode='lines',
                  line=dict(color='#d62728', width=2),
                  name='Cumulative Biomass',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Biomass: %{y:.2f} kg<extra></extra>'),
        secondary_y=False
    )
    
    # Cumulative volume on secondary y-axis (right)
    fig.add_trace(
        go.Scatter(x=all_times, y=all_volume, mode='lines',
                  line=dict(color='#9467bd', width=2),
                  name='Cumulative Volume',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Volume: %{y:.2f} L<extra></extra>'),
        secondary_y=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (h)")
    fig.update_yaxes(title_text="Biomass (kg)", secondary_y=False)
    fig.update_yaxes(title_text="Volume (L)", secondary_y=True)
    
    # Get simulation type for title
    sim_type = getattr(sim_data, 'simulation_type', 'Simulation')
    
    # Update layout
    default_layout = dict(
        title_text=f"Cumulative Harvest - {sim_type}",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        height=400,
        width=800,
        font=dict(size=12),
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_simulation_cumulative_medium_use(
    source: Union['CellCultureSimulation', 'SimulationData'],
    max_batches: Optional[int] = None,
    layout_kwargs: Optional[Dict] = None,
) -> go.Figure:
    """Plot cumulative external batch medium, feed medium, and base solution.

    The total trace is the sum of the three stored incremental inputs:
    ``BatchData.batch_medium_in``, ``BatchData.feed_medium_in``, and
    ``BatchData.base_in``. Base solution is shown separately because it is
    added by pH control rather than as culture medium.
    
    Parameters
    ----------
    source : CellCultureSimulation or SimulationData
        Simulation results to visualize.
    max_batches : int, optional
        Maximum number of leading batches to include. If ``None``, include all
        available batches.
    layout_kwargs : dict, optional
        Keyword arguments that override the default Plotly layout.
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing cumulative input-volume profiles in L.
        
    Examples
    --------
    >>> from animakinetica.analysis.process_visualization import plot_simulation_cumulative_medium_use
    >>> fig = plot_simulation_cumulative_medium_use(sim, max_batches=10)
    >>> fig.show()
    """
    # Get simulation data
    sim_data = get_simulation_data(source)
    
    n_batches = len(sim_data.batches) if max_batches is None else min(max_batches, len(sim_data.batches))
    
    # Build continuous time series across all batches
    all_times = []
    all_batch_medium = []
    all_feed_medium = []
    all_base_medium = []
    all_total_medium = []
    
    t_offset = 0
    cumul_batch_offset = 0
    cumul_feed_offset = 0
    cumul_base_offset = 0
    
    for batch_idx in range(n_batches):
        batch = sim_data.batches[batch_idx]
        
        # Get time points and adjust for offset
        t_batch = np.asarray(batch.t_points, dtype=float) + t_offset
        
        # Get incremental medium use arrays and cumsum to get cumulative time-series
        batch_medium = np.cumsum(np.asarray(batch.batch_medium_in, dtype=float)) + cumul_batch_offset
        feed_medium = np.cumsum(np.asarray(batch.feed_medium_in, dtype=float)) + cumul_feed_offset
        base_medium_raw = batch.base_in if batch.base_in is not None else np.zeros_like(batch.t_points, dtype=float)
        base_medium = np.cumsum(np.asarray(base_medium_raw, dtype=float)) + cumul_base_offset
        total_medium = batch_medium + feed_medium + base_medium
        
        # Append to overall arrays
        all_times.extend(t_batch)
        all_batch_medium.extend(batch_medium)
        all_feed_medium.extend(feed_medium)
        all_base_medium.extend(base_medium)
        all_total_medium.extend(total_medium)
        
        # Update offsets for next batch (sum of incremental = batch total)
        t_offset = t_batch[-1]
        cumul_batch_offset = batch_medium[-1]
        cumul_feed_offset = feed_medium[-1]
        cumul_base_offset = base_medium[-1]
    
    # Create figure
    fig = go.Figure()
    
    # Total medium
    fig.add_trace(
        go.Scatter(x=all_times, y=all_total_medium, mode='lines',
                  line=dict(color='#2ca02c', width=2),
                  name='Total Medium',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Total Medium: %{y:.2f} L<extra></extra>')
    )
    
    # Feed medium
    fig.add_trace(
        go.Scatter(x=all_times, y=all_feed_medium, mode='lines',
                  line=dict(color='#ff7f0e', width=2),
                  name='Feed Medium',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Feed Medium: %{y:.2f} L<extra></extra>')
    )

    # Base medium
    fig.add_trace(
        go.Scatter(x=all_times, y=all_base_medium, mode='lines',
                  line=dict(color='#d62728', width=2),
                  name='Base Medium',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Base Medium: %{y:.2f} L<extra></extra>')
    )
    
    # Batch medium
    fig.add_trace(
        go.Scatter(x=all_times, y=all_batch_medium, mode='lines',
                  line=dict(color='#1f77b4', width=2),
                  name='Batch Medium',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Batch Medium: %{y:.2f} L<extra></extra>')
    )
    
    # Get simulation type for title
    sim_type = getattr(sim_data, 'simulation_type', 'Simulation')
    
    # Update layout
    default_layout = dict(
        title_text=f"Cumulative Medium Use - {sim_type}",
        xaxis_title="Time (h)",
        yaxis_title="Medium Use (L)",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        height=400,
        width=1000,
        font=dict(size=12),
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig



def plot_simulation_cumulative_base_use(
    source: Union['CellCultureSimulation', 'SimulationData'],
    max_batches: Optional[int] = None,
    layout_kwargs: Optional[Dict] = None,
) -> go.Figure:
    """Plot cumulative base-solution addition for pH control.

    The plot uses incremental ``BatchData.base_in`` values when present. For
    legacy results without that array, it estimates volume increments from the
    trapezoidal average of ``BatchData.D_base_list`` and reactor volume:
    $D_{base} V \Delta t$.
    
    Parameters
    ----------
    source : CellCultureSimulation or SimulationData
        Simulation results to visualize.
    max_batches : int, optional
        Maximum number of leading batches to include. If ``None``, include all
        available batches.
    layout_kwargs : dict, optional
        Keyword arguments that override the default Plotly layout.
    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing the cumulative base-solution volume in L.
        
    Examples
    --------
    >>> from animakinetica.core.simulation import CellCultureSimulation
    >>> from animakinetica.analysis.process_visualization import plot_simulation_cumulative_base_use
    >>> sim = CellCultureSimulation(params, Kinetics.EXCESS, OperatingMode.FEDBATCH)
    >>> sim.run_simulation(number_of_batches=10)
    >>> fig = plot_simulation_cumulative_base_use(sim, max_batches=5)
    >>> fig.show()
    """
    # Get simulation data
    sim_data = get_simulation_data(source)
    
    n_batches = len(sim_data.batches) if max_batches is None else min(max_batches, len(sim_data.batches))
    
    # Build continuous time series across all batches
    all_times = []
    all_base_use = []
    
    t_offset = 0
    cumul_base_offset = 0
    
    for batch_idx in range(n_batches):
        batch = sim_data.batches[batch_idx]
        
        # Get time points and adjust for offset
        t = np.asarray(batch.t_points, dtype=float)
        t_batch = t + t_offset

        if batch.base_in is not None:
            base_cumul_batch = np.cumsum(np.asarray(batch.base_in, dtype=float)) + cumul_base_offset
        else:
            if batch.D_base_list is None:
                raise ValueError("Base medium use data not available for all batches")
            V = np.asarray(batch.y_points[5], dtype=float)
            D_base = np.asarray(batch.D_base_list, dtype=float)
            dt = np.diff(t)
            D_base_avg = (D_base[:-1] + D_base[1:]) / 2
            V_avg = (V[:-1] + V[1:]) / 2
            base_increments = D_base_avg * V_avg * dt
            base_cumul_batch = np.zeros(len(t))
            base_cumul_batch[1:] = np.cumsum(base_increments)
            base_cumul_batch = base_cumul_batch + cumul_base_offset
        
        # Append to overall arrays
        all_times.extend(t_batch)
        all_base_use.extend(base_cumul_batch)
        
        # Update offsets for next batch
        t_offset = t_batch[-1]
        cumul_base_offset = base_cumul_batch[-1]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=all_times, y=all_base_use, mode='lines',
                  line=dict(color='#d62728', width=2),
                  name='Cumulative Base Use',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Base Use: %{y:.2f} L<extra></extra>')
    )
    
    # Get simulation type for title
    sim_type = getattr(sim_data, 'simulation_type', 'Simulation')
    
    # Update layout
    default_layout = dict(
        title_text=f"Cumulative Base Use - {sim_type}",
        xaxis_title="Time (h)",
        yaxis_title="Base Volume (L)",
        showlegend=False,
        height=400,
        width=800,
        font=dict(size=12),
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_simulation_profiles(
    source: Union['CellCultureSimulation', 'SimulationData'],
    max_batches: int = 1,
    layout_kwargs: Optional[Dict] = None,
) -> List[go.Figure]:
    """Return available state and controller profiles over campaign time.

    The function creates a separate figure for each available variable: viable
    cell density, glucose, glutamine, lactate, ammonia, total amino acids,
    dissolved oxygen, total dissolved carbon, pH, dissolved CO2, osmolality,
    and sodium concentration. pH is calculated from the hydrogen-ion state as
    ``-log10(C_H)``. Dissolved CO2 and osmolality require ``BatchData`` trace
    arrays and are omitted if no selected batch contains the required data.

    Each trace represents one logical batch, shifted onto a common elapsed
    campaign-time axis. This preserves discontinuities between batches rather
    than implying that process states are continuous across reinoculation.
    
    Parameters
    ----------
    source : CellCultureSimulation or SimulationData
        Simulation results to visualize.
    max_batches : int, optional
        Maximum number of leading batches to include. Defaults to ``1``. If
        ``None``, include all available batches.
    layout_kwargs : dict, optional
        Keyword arguments that override the default Plotly layout of every
        returned figure.
    Returns
    -------
    list of plotly.graph_objects.Figure
        Figures in this order when data are available: cell density, glucose,
        glutamine, lactate, ammonia, amino acids, dissolved oxygen, total
        dissolved carbon, pH, dissolved CO2, osmolality, and sodium.
        
    Examples
    --------
    >>> from animakinetica.analysis.process_visualization import plot_simulation_profiles
    >>> figs = plot_simulation_profiles(sim, max_batches=5)
    >>> figs[0].show()  # Cell density
    >>> figs[1].show()  # Glucose
    >>> figs[5].show()  # Amino acids
    >>> figs[8].show()  # pH
    >>> figs[10].show()  # Osmolality
    >>> figs[11].show()  # Na⁺ concentration
    """
    # Get simulation data
    sim_data = get_simulation_data(source)
    
    if sim_data.parameters is None:
        raise ValueError("Parameters not available in simulation data")
    
    dry_cell_mass = float(sim_data.parameters.cell_properties.dry_cell_mass)
    n_batches = max_batches if max_batches is not None else len(sim_data.batches)
    
    # Generate colors for batches
    colors = px.colors.qualitative.Set1 if n_batches <= 9 else px.colors.qualitative.Light24
    
    figures = []
    
    # Define plot configurations: (title, y_label, data_source, index/conversion)
    # data_source: 'y' for y_points array, 'list' for batch attribute list, 'pH' for pH conversion
    plot_configs = [
        ("Cell Density Profile", "Cell Density (10⁶ cells/mL)", "y", 0, lambda y: y / 1e9),
        ("Glucose Profile", "Glucose (mmol/L)", "y", 1, lambda y: y),
        ("Glutamine Profile", "Glutamine (mmol/L)", "y", 2, lambda y: y),
        ("Lactate Profile", "Lactate (mmol/L)", "y", 3, lambda y: y),
        ("Ammonia Profile", "Ammonia (mmol/L)", "y", 4, lambda y: y),
        ("Amino Acids Profile", "Amino Acids (mmol/L)", "y", 10, lambda y: y),
        ("Dissolved Oxygen Profile", "Dissolved O₂ (mmol/L)", "y", 6, lambda y: y),
        ("Total Dissolved Carbon Profile", "Total Dissolved Carbon (mmol/L)", "y", 7, lambda y: y),
        ("pH Profile", "pH", "pH", 8, lambda C_H: -np.log10(C_H + 1e-14)),  # Add small value to avoid log(0)
        ("Dissolved CO₂ Profile", "Dissolved CO₂ (mmol/L)", "list", "C_CO2_list", lambda y: y),
        ("Osmolality Profile", "Osmolality (mOsm/kg)", "list", "osm_list", lambda y: y),
        ("Added Sodium Profile", "Added Na⁺ from Base (mmol/L)", "y", 9, lambda y: y),
    ]
    
    # Get simulation type for titles
    sim_type = getattr(sim_data, 'simulation_type', 'Simulation')
    
    # Create a figure for each variable
    for title, y_label, data_source, index_or_attr, conversion_func in plot_configs:
        fig = go.Figure()
        
        t_offset = 0
        for batch_idx in range(n_batches):
            batch = sim_data.batches[batch_idx]
            t = np.asarray(batch.t_points, dtype=float)
            t_shifted = t + t_offset
            
            # Extract data based on source type
            if data_source == "y" or data_source == "pH":
                y = np.asarray(batch.y_points, dtype=float)
                y_data = conversion_func(y[index_or_attr])
            elif data_source == "list":
                # Get data from batch attribute list
                list_data = getattr(batch, index_or_attr, None)
                if list_data is None:
                    # Skip this plot if data not available
                    continue
                y_data = conversion_func(np.asarray(list_data, dtype=float))
            
            # Determine hover format based on variable
            if "pH" in title:
                hover_format = f'Time: %{{x:.1f}} h<br>{y_label}: %{{y:.2f}}<extra></extra>'
            else:
                hover_format = f'Time: %{{x:.1f}} h<br>{y_label}: %{{y:.2f}}<extra></extra>'
            
            fig.add_trace(
                go.Scatter(
                    x=t_shifted,
                    y=y_data,
                    mode='lines',
                    name=f'Batch {batch.batch_number}',
                    line=dict(width=2, color=colors[batch_idx % len(colors)]),
                    hovertemplate=hover_format
                )
            )
            
            t_offset = t_shifted[-1]
        
        # Skip empty figures (when data not available for any batch)
        if len(fig.data) == 0:
            continue
        
        # Update layout
        default_layout = dict(
            title_text=f"{title} - {sim_type}",
            xaxis_title="Time (h)",
            yaxis_title=y_label,
            height=500,
            width=1000,
            font=dict(size=12),
            hovermode='x unified',
            template='plotly_white'
        )
        
        if layout_kwargs:
            default_layout.update(layout_kwargs)
        
        fig.update_layout(**default_layout)
        
        figures.append(fig)
    
    return figures
