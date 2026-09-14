"""Create Plotly figures from batch-level simulation results.

The plotting functions accept a
:class:`~animakinetica.core.simulation.CellCultureSimulation`, its
:class:`~animakinetica.core.simulation_data.SimulationData`, or one
:class:`~animakinetica.core.simulation_data.BatchData`. Batch identifiers are
one-based when a simulation or aggregate data object is supplied. For a
``BatchData`` source, ``batch_number`` is ignored.

Use :func:`plot_batch_biomass_metabolites` for viable-cell and metabolite
profiles, :func:`plot_batch_volume_feed` for culture volume and calculated
inlet rates, and :func:`plot_batch_pH_control` for pH-control and gas-phase
carbon-dioxide traces. The pH-control figure requires detailed auxiliary
traces retained by a simulation run with ``performance_mode=False``.
"""

# Imports
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Union, TYPE_CHECKING

from animakinetica.analysis._visualization_utils import get_batch_data

if TYPE_CHECKING:
    from animakinetica.core.simulation import CellCultureSimulation
    from animakinetica.core.simulation_data import SimulationData, BatchData
else:
    from animakinetica.core.simulation_data import BatchData

# Module metadata
__version__ = "1.0.0"

__all__ = [
    "plot_batch_biomass_metabolites",
    "plot_batch_volume_feed",
    "plot_batch_pH_control",
]


def plot_batch_biomass_metabolites(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot viable-cell density and metabolite concentrations for one batch.

    The figure contains a viable-cell-density panel and a metabolite panel.
    Glucose, glutamine, lactate, and, when present, total amino acids use the
    primary metabolite axis. Ammonia uses a secondary axis because its scale
    can differ substantially. Cell density is converted from cells/L to
    $10^6$ cells/mL.

    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Result source resolved by
        :func:`~animakinetica.analysis._visualization_utils.get_batch_data`.
    batch_number : int, default=1
        One-based logical-batch identifier. Ignored when ``source`` is a
        ``BatchData`` instance.
    layout_kwargs : dict, optional
        Keyword arguments passed to :meth:`plotly.graph_objects.Figure.update_layout`.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with viable-cell-density and metabolite trajectories.

    Raises
    ------
    ValueError
        If the selected batch is not available in ``source``.

    Examples
    --------
    >>> fig = plot_batch_biomass_metabolites(simulation, batch_number=1)
    >>> fig.show()
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Ensure data is in basic numpy/python types (not Parameter objects)
    t = np.asarray(batch_data.t_points, dtype=float)
    y = np.asarray(batch_data.y_points, dtype=float)
    
    # Define colors
    colors = {
        'cell_density': '#1f77b4',
        'glucose': '#2ca02c',
        'glutamine': '#9467bd',
        'lactate': '#ff7f0e',
        'ammonia': '#d62728',
        'amino_acids': '#8c564b'
    }

    # Create subplots: 1 row, 2 columns with secondary y-axis on column 2
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cell Density", "Metabolites"),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        horizontal_spacing=0.12
    )
    
    # Cell density in subplot 1 (no legend)
    fig.add_trace(
        go.Scatter(x=t, y=y[0] / 1e9, mode='lines', 
                  line=dict(color=colors['cell_density'], width=2),
                  name='Cell Density',
                  showlegend=False,
                  hovertemplate='Time: %{x:.1f} h<br>Cell Density: %{y:.2f} 10⁶ cells/mL<extra></extra>'),
        row=1, col=1
    )
    
    # Glucose (primary y-axis - left in subplot 2)
    fig.add_trace(
        go.Scatter(x=t, y=y[1], mode='lines',
                  line=dict(color=colors['glucose'], width=2),
                  name='Glucose',
                  hovertemplate='Time: %{x:.1f} h<br>Glucose: %{y:.2f} mmol/L<extra></extra>'),
        row=1, col=2, secondary_y=False
    )
    
    # Glutamine (primary y-axis - left in subplot 2)
    fig.add_trace(
        go.Scatter(x=t, y=y[2], mode='lines',
                  line=dict(color=colors['glutamine'], width=2),
                  name='Glutamine',
                  hovertemplate='Time: %{x:.1f} h<br>Glutamine: %{y:.2f} mmol/L<extra></extra>'),
        row=1, col=2, secondary_y=False
    )
    
    # Lactate (primary y-axis - left in subplot 2)
    fig.add_trace(
        go.Scatter(x=t, y=y[3], mode='lines',
                  line=dict(color=colors['lactate'], width=2),
                  name='Lactate',
                  hovertemplate='Time: %{x:.1f} h<br>Lactate: %{y:.2f} mmol/L<extra></extra>'),
        row=1, col=2, secondary_y=False
    )
    
    # Ammonia (secondary y-axis - right in subplot 2)
    fig.add_trace(
        go.Scatter(x=t, y=y[4], mode='lines',
                  line=dict(color=colors['ammonia'], width=2),
                  name='Ammonia',
                  hovertemplate='Time: %{x:.1f} h<br>Ammonia: %{y:.2f} mmol/L<extra></extra>'),
        row=1, col=2, secondary_y=True
    )

    # Amino acids (primary y-axis - left in subplot 2, alongside other metabolites)
    if y.shape[0] > 10:
        fig.add_trace(
            go.Scatter(x=t, y=y[10], mode='lines',
                      line=dict(color=colors['amino_acids'], width=2),
                      name='Amino Acids',
                      hovertemplate='Time: %{x:.1f} h<br>Amino Acids: %{y:.2f} mmol/L<extra></extra>'),
            row=1, col=2, secondary_y=False
        )
    
    # Update axes
    fig.update_xaxes(title_text="Time (h)", row=1, col=1)
    fig.update_xaxes(title_text="Time (h)", row=1, col=2)
    fig.update_yaxes(title_text="10⁶ cells/mL", row=1, col=1)
    fig.update_yaxes(title_text="Glc, Gln, Lac, AA (mmol/L)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Ammonia (mmol/L)", row=1, col=2, secondary_y=True)
    
    # Update layout
    default_layout = dict(
        title_text=f"Batch {batch_number}: Biomass and Metabolite Progression",
        showlegend=True,
        legend=dict(x=0.4, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        height=600,
        width=1400,
        font=dict(size=12),
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_batch_volume_feed(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot culture volume, dilution rates, and calculated inlet rates.

    Process dilution, base-addition dilution, and their sum are plotted with
    culture volume. The corresponding inlet rates are calculated as dilution
    rate times culture volume. Missing process or base dilution traces are
    treated as zero, which supports operating modes without that action.

    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Result source resolved by
        :func:`~animakinetica.analysis._visualization_utils.get_batch_data`.
    batch_number : int, default=1
        One-based logical-batch identifier. Ignored when ``source`` is a
        ``BatchData`` instance.
    layout_kwargs : dict, optional
        Keyword arguments passed to :meth:`plotly.graph_objects.Figure.update_layout`.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with volume, dilution-rate, and inlet-rate trajectories.

    Raises
    ------
    ValueError
        If the selected batch is not available in ``source``.

    Examples
    --------
    >>> fig = plot_batch_volume_feed(simulation.data, batch_number=1)
    >>> fig.show()
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Ensure data is in basic numpy/python types (not Parameter objects)
    t = np.asarray(batch_data.t_points, dtype=float)
    y = np.asarray(batch_data.y_points, dtype=float)
    D_list = np.asarray(batch_data.D_list, dtype=float) if batch_data.D_list is not None else None
    D_base_list = np.asarray(batch_data.D_base_list, dtype=float) if batch_data.D_base_list is not None else None

    if D_list is None:
        D_list = np.zeros_like(t)
    if D_base_list is None:
        D_base_list = np.zeros_like(t)

    D_total_list = D_list + D_base_list
    F_process_list = D_list * y[5]
    F_base_list = D_base_list * y[5]
    F_total_list = D_total_list * y[5]

    # Create figure with two subplots: Volume+Dilution Rate | Feed Rate
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Volume and Dilution Rate", "Feed Rate"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        horizontal_spacing=0.12
    )

    # Volume on primary y-axis (left) of subplot 1
    fig.add_trace(
        go.Scatter(x=t, y=y[5], mode='lines',
                  line=dict(color='#8c564b', width=2),
                  name='Volume',
                  hovertemplate='Time: %{x:.1f} h<br>Volume: %{y:.2f} L<extra></extra>'),
        row=1, col=1, secondary_y=False
    )

    # Inlet dilution rates on secondary y-axis (right) of subplot 1
    fig.add_trace(
        go.Scatter(x=t, y=D_list, mode='lines',
                  line=dict(color='#e377c2', width=2),
                  name='Process Dilution Rate',
                  hovertemplate='Time: %{x:.1f} h<br>D_process: %{y:.4f} 1/h<extra></extra>'),
        row=1, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=t, y=D_base_list, mode='lines',
                  line=dict(color='#d62728', width=2, dash='dot'),
                  name='Base Dilution Rate',
                  hovertemplate='Time: %{x:.1f} h<br>D_base: %{y:.4f} 1/h<extra></extra>'),
        row=1, col=1, secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=t, y=D_total_list, mode='lines',
                  line=dict(color='#9467bd', width=2, dash='dash'),
                  name='Total Inlet Dilution Rate',
                  hovertemplate='Time: %{x:.1f} h<br>D_total: %{y:.4f} 1/h<extra></extra>'),
        row=1, col=1, secondary_y=True
    )

    # Inlet rates in subplot 2
    fig.add_trace(
        go.Scatter(x=t, y=F_process_list, mode='lines',
                  line=dict(color='#17becf', width=2),
                  name='Process Feed Rate',
                  hovertemplate='Time: %{x:.1f} h<br>F_process: %{y:.4f} L/h<extra></extra>'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=t, y=F_base_list, mode='lines',
                  line=dict(color='#ff7f0e', width=2, dash='dot'),
                  name='Base Feed Rate',
                  hovertemplate='Time: %{x:.1f} h<br>F_base: %{y:.4f} L/h<extra></extra>'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=t, y=F_total_list, mode='lines',
                  line=dict(color='#2ca02c', width=2, dash='dash'),
                  name='Total Inlet Rate',
                  hovertemplate='Time: %{x:.1f} h<br>F_total: %{y:.4f} L/h<extra></extra>'),
        row=1, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Time (h)", row=1, col=1)
    fig.update_xaxes(title_text="Time (h)", row=1, col=2)
    fig.update_yaxes(title_text="Volume (L)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Dilution Rate (1/h)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Feed Rate (L/h)", row=1, col=2)

    # Update layout
    default_layout = dict(
        title_text=f"Batch {batch_number}: Volume and Inlet Rate Progression",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        height=400,
        width=1400,
        font=dict(size=12),
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_batch_pH_control(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot pH-control and carbon-dioxide variables for one batch.

    The six panels show pH calculated from the hydrogen-ion state, base and
    acid dilution rates, dissolved CO2 concentration, sodium concentration,
    and the inlet gas CO2 mole fraction. The acid-rate panel remains empty
    when no acid-addition trace is stored.

    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Result source resolved by
        :func:`~animakinetica.analysis._visualization_utils.get_batch_data`.
    batch_number : int, default=1
        One-based logical-batch identifier. Ignored when ``source`` is a
        ``BatchData`` instance.
    layout_kwargs : dict, optional
        Keyword arguments passed to :meth:`plotly.graph_objects.Figure.update_layout`.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with pH-control and carbon-dioxide trajectories.

    Raises
    ------
    ValueError
        If the selected batch is absent, or required pH-control traces were
        not stored. Detailed traces are normally available only when the
        simulation ran with ``performance_mode=False``.

    Examples
    --------
    >>> fig = plot_batch_pH_control(simulation, batch_number=1)
    >>> fig.show()
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Ensure data is in basic numpy/python types (not Parameter objects)
    t = np.asarray(batch_data.t_points, dtype=float)
    y = np.asarray(batch_data.y_points, dtype=float)
    
    # Check if pH control data is available
    if batch_data.D_base_list is None or batch_data.C_CO2_list is None:
        raise ValueError(f"pH control data not available for batch {batch_number}")
    
    # Extract D_acid (may be None for simulations without acid addition)
    D_acid_list = np.asarray(batch_data.D_acid_list, dtype=float) if batch_data.D_acid_list is not None else None
    
    # Create subplots: 1 row, 6 columns
    fig = make_subplots(
        rows=1, cols=6,
        subplot_titles=(
            "pH",
            "Base Addition Rate",
            "Acid Addition Rate",
            "Dissolved CO₂",
            "Na⁺ Concentration",
            "y_CO₂ in gas phase"
        ),
        horizontal_spacing=0.08
    )
    
    # Define colors
    colors = ['#1f77b4', '#d62728', '#e377c2', '#9467bd', '#ff7f0e', '#2ca02c']
    
    # 1. pH (calculated from H+ concentration, y[8])
    C_H = y[8] if y.shape[0] > 8 else None
    if C_H is not None:
        pH = -np.log10(C_H + 1e-14)  # Add small value to avoid log(0)
        fig.add_trace(
            go.Scatter(x=t, y=pH, mode='lines',
                      line=dict(color=colors[0], width=2),
                      name='pH',
                      hovertemplate='Time: %{x:.1f} h<br>pH: %{y:.2f}<extra></extra>'),
            row=1, col=1
        )
    
    # 2. Base dilution rate
    D_base = np.asarray(batch_data.D_base_list, dtype=float)
    fig.add_trace(
        go.Scatter(x=t, y=D_base, mode='lines',
                  line=dict(color=colors[1], width=2),
                  name='D_base',
                  hovertemplate='Time: %{x:.1f} h<br>D_base: %{y:.4f} 1/h<extra></extra>'),
        row=1, col=2
    )
    
    # 3. Acid dilution rate
    if D_acid_list is not None:
        fig.add_trace(
            go.Scatter(x=t, y=D_acid_list, mode='lines',
                      line=dict(color=colors[2], width=2),
                      name='D_acid',
                      hovertemplate='Time: %{x:.1f} h<br>D_acid: %{y:.4f} 1/h<extra></extra>'),
            row=1, col=3
        )
    
    # 4. Dissolved CO2
    C_CO2 = np.asarray(batch_data.C_CO2_list, dtype=float)
    fig.add_trace(
        go.Scatter(x=t, y=C_CO2, mode='lines',
                  line=dict(color=colors[3], width=2),
                  name='C_CO₂',
                  hovertemplate='Time: %{x:.1f} h<br>C_CO₂: %{y:.2f} mmol/L<extra></extra>'),
        row=1, col=4
    )
    
    # 5. Added sodium concentration from base/acid addition (y[9])
    C_Na = y[9] if y.shape[0] > 9 else None
    if C_Na is not None:
        fig.add_trace(
            go.Scatter(x=t, y=C_Na, mode='lines',
                      line=dict(color=colors[4], width=2),
                      name='Added Na⁺',
                      hovertemplate='Time: %{x:.1f} h<br>Added Na⁺: %{y:.1f} mmol/L<extra></extra>'),
            row=1, col=5
        )
    
    # 6. CO2 in gas phase
    y_CO2_gas = np.asarray(batch_data.y_CO2_list, dtype=float)
    fig.add_trace(
        go.Scatter(x=t, y=y_CO2_gas, mode='lines',
                  line=dict(color=colors[5], width=2),
                  name='y_CO₂ (gas)',
                  hovertemplate='Time: %{x:.1f} h<br>y_CO₂ (gas): %{y:.2f}<extra></extra>'),
        row=1, col=6
    )

    # Update axes
    for col in range(1, 7):
        fig.update_xaxes(title_text="Time (h)", row=1, col=col)
    
    fig.update_yaxes(title_text="pH", row=1, col=1)
    fig.update_yaxes(title_text="D_base (1/h)", row=1, col=2)
    fig.update_yaxes(title_text="D_acid (1/h)", row=1, col=3)
    fig.update_yaxes(title_text="C_CO₂ (mmol/L)", row=1, col=4)
    fig.update_yaxes(title_text="Added Na⁺ (mmol/L)", row=1, col=5)
    fig.update_yaxes(title_text="y_CO₂ (gas)", row=1, col=6)
    
    # Update layout
    default_layout = dict(
        title_text=f"Batch {batch_number}: pH Control Parameters",
        showlegend=False,
        height=400,
        width=2100,
        font=dict(size=12),
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig
