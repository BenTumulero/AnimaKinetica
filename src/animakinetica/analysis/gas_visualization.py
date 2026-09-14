"""Visualize gas-transfer, gas-control, and hydrodynamic simulation results.

This module provides Plotly figures for the gas-control histories stored in
:class:`~animakinetica.core.simulation_data.BatchData` and for gas use
integrated across batches in
:class:`~animakinetica.core.simulation_data.SimulationData`. The batch-level
figure includes dissolved oxygen (DO), agitation speed, gas flow, inlet oxygen
mole fraction, power input per liquid volume, and superficial gas velocity.

Detailed gas-control histories are available only when the simulation was run
with ``performance_mode=False``. Gas flow is reported in vvh (gas volume per
liquid volume per hour); cumulative gas use is calculated from this flow and
the simulated liquid volume, then displayed in m³.
"""

# Imports
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Union, TYPE_CHECKING

from animakinetica.analysis._visualization_utils import get_batch_data, get_simulation_data

if TYPE_CHECKING:
    from animakinetica.core.simulation import CellCultureSimulation
    from animakinetica.core.simulation_data import SimulationData, BatchData
else:
    from animakinetica.core.simulation_data import BatchData

# Module metadata
__version__ = "1.0.0"

__all__ = [
    "plot_batch_gas_transfer",
    "plot_simulation_cumulative_gas_use",
]


def plot_batch_gas_transfer(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot gas-transfer and hydrodynamic profiles for one logical batch.

    The returned figure has six panels for dissolved oxygen, agitation speed,
    gas flow, inlet oxygen mole fraction, power input per liquid volume, and
    superficial gas velocity. The power-input and superficial-velocity panels
    are left empty when their optional histories are not present.

    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Simulation result source. A
        :class:`~animakinetica.core.simulation.CellCultureSimulation` is
        resolved through its ``data`` attribute. When ``source`` is a
        :class:`~animakinetica.core.simulation_data.BatchData`,
        ``batch_number`` is ignored.
    batch_number : int, optional
        One-based logical-batch identifier to plot. Default is ``1``.
    layout_kwargs : dict, optional
        Keyword arguments merged into the default
        :meth:`plotly.graph_objects.Figure.update_layout` configuration.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive six-panel gas-transfer and hydrodynamic figure.

    Raises
    ------
    TypeError
        If ``source`` is not a supported simulation-result object.
    ValueError
        If the requested batch is absent or its required gas-control histories
        (``N_list``, ``Fg_list``, or ``y_O2_list``) are unavailable.

    Notes
    -----
    Gas-control histories are not retained for simulations run with
    ``performance_mode=True``. Units are mmol/L for DO, rpm for agitation,
    vvh for gas flow, W/m³ for power input per volume, and m/s for superficial
    gas velocity.

    Examples
    --------
    >>> from animakinetica.core.simulation import CellCultureSimulation
    >>> from animakinetica.analysis.gas_visualization import plot_batch_gas_transfer
    >>> sim = CellCultureSimulation(params, performance_mode=False)
    >>> sim.run_simulation()
    >>> fig = plot_batch_gas_transfer(sim, batch_number=1)
    >>> fig.show()
    """
    # Get batch data
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")
    
    # Ensure data is in basic numpy/python types (not Parameter objects)
    t = np.asarray(batch_data.t_points, dtype=float)
    y = np.asarray(batch_data.y_points, dtype=float)
    
    # Check if gas transfer data is available
    if (batch_data.N_list is None or batch_data.Fg_list is None or 
        batch_data.y_O2_list is None):
        raise ValueError(f"Gas transfer data not available for batch {batch_number}")
    
    # Create subplots: 2 rows, 3 columns
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Dissolved Oxygen",
            "Stirring Speed",
            "Gas Flow Rate",
            "O₂ Mole Fraction",
            "Power Input per Volume",
            "Superficial Gas Velocity"
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.14
    )
    
    # Define colors
    colors = ['#17becf', '#bcbd22', '#7f7f7f', '#e377c2', '#1f77b4', '#ff7f0e']
    
    # Dissolved oxygen is tracked as y[6] in the ODE system
    dissolved_O2 = y[6] if y.shape[0] > 6 else None
    
    if dissolved_O2 is not None:
        fig.add_trace(
            go.Scatter(x=t, y=dissolved_O2, mode='lines',
                      line=dict(color=colors[0], width=2),
                      name='Dissolved O₂',
                      hovertemplate='Time: %{x:.1f} h<br>DO: %{y:.3f} mmol/L<extra></extra>'),
            row=1, col=1
        )
    
    # Stirring speed
    fig.add_trace(
        go.Scatter(x=t, y=batch_data.N_list, mode='lines',
                  line=dict(color=colors[1], width=2),
                  name='Stirring Speed',
                  hovertemplate='Time: %{x:.1f} h<br>N: %{y:.1f} rpm<extra></extra>'),
        row=1, col=2
    )
    
    # Gas flow rate
    fig.add_trace(
        go.Scatter(x=t, y=batch_data.Fg_list, mode='lines',
                  line=dict(color=colors[2], width=2),
                  name='Gas Flow Rate',
                  hovertemplate='Time: %{x:.1f} h<br>Fg: %{y:.3f} vvh<extra></extra>'),
        row=1, col=3
    )
    
    # Partial oxygen pressure
    fig.add_trace(
        go.Scatter(x=t, y=batch_data.y_O2_list, mode='lines',
                  line=dict(color=colors[3], width=2),
                  name='y_O₂',
                  hovertemplate='Time: %{x:.1f} h<br>y_O₂: %{y:.3f}<extra></extra>'),
        row=2, col=1
    )

    if batch_data.P_over_V_list is not None:
        fig.add_trace(
            go.Scatter(x=t, y=batch_data.P_over_V_list, mode='lines',
                      line=dict(color=colors[4], width=2),
                      name='P/V',
                      hovertemplate='Time: %{x:.1f} h<br>P/V: %{y:.2f} W/m³<extra></extra>'),
            row=2, col=2
        )

    if batch_data.v_s_list is not None:
        fig.add_trace(
            go.Scatter(x=t, y=batch_data.v_s_list, mode='lines',
                      line=dict(color=colors[5], width=2),
                      name='vₛ',
                      hovertemplate='Time: %{x:.1f} h<br>vₛ: %{y:.5f} m/s<extra></extra>'),
            row=2, col=3
        )
    
    # Update axes
    for row in range(1, 3):
        for col in range(1, 4):
            fig.update_xaxes(title_text="Time (h)", row=row, col=col)
    
    fig.update_yaxes(title_text="DO (mmol/L)", row=1, col=1)
    fig.update_yaxes(title_text="RPM", row=1, col=2)
    fig.update_yaxes(title_text="vvh", row=1, col=3)
    fig.update_yaxes(title_text="Mole Fraction", row=2, col=1)
    fig.update_yaxes(title_text="W/m³", row=2, col=2)
    fig.update_yaxes(title_text="m/s", row=2, col=3)
    
    # Update layout
    default_layout = dict(
        title_text=f"Batch {batch_number}: Gas Transfer and Hydrodynamic Parameters",
        showlegend=False,
        height=750,
        width=1400,
        font=dict(size=12),
        template='plotly_white'
    )
    
    if layout_kwargs:
        default_layout.update(layout_kwargs)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_simulation_cumulative_gas_use(
    source: Union['CellCultureSimulation', 'SimulationData'],
    max_batches: Optional[int] = None,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot cumulative inlet-gas use over selected logical batches.

    For each batch, gas use is integrated with the trapezoidal rule from the
    gas-flow history and liquid-volume trajectory. Individual batch time axes
    are concatenated so the returned trace is continuous in cumulative gas
    volume.

    Parameters
    ----------
    source : CellCultureSimulation or SimulationData
        Simulation result source. A
        :class:`~animakinetica.core.simulation.CellCultureSimulation` is
        resolved through its ``data`` attribute.
    max_batches : int, optional
        Maximum number of leading batches to include. If ``None``, includes
        every batch in the simulation. Default is ``None``.
    layout_kwargs : dict, optional
        Keyword arguments merged into the default
        :meth:`plotly.graph_objects.Figure.update_layout` configuration.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure of cumulative inlet-gas volume in m³ versus
        concatenated simulated time in h.

    Raises
    ------
    TypeError
        If ``source`` is not a
        :class:`~animakinetica.core.simulation.CellCultureSimulation` or
        :class:`~animakinetica.core.simulation_data.SimulationData` instance.
    ValueError
        If gas-flow history is unavailable for a selected batch.

    Notes
    -----
    At each integration interval, the gas volume is calculated as
    ``mean(Fg) * mean(V) * dt``, where ``Fg`` is in vvh, ``V`` is in L, and
    ``dt`` is in h. The resulting volume in L is converted to m³. The plot
    includes inlet-gas volume only; it is not a gas-component material balance.

    Examples
    --------
    >>> from animakinetica.core.simulation import CellCultureSimulation
    >>> from animakinetica.analysis.gas_visualization import plot_simulation_cumulative_gas_use
    >>> sim = CellCultureSimulation(params, performance_mode=False)
    >>> sim.run_simulation(number_of_batches=10)
    >>> fig = plot_simulation_cumulative_gas_use(sim, max_batches=5)
    >>> fig.show()
    """
    # Get simulation data
    sim_data = get_simulation_data(source)
    
    n_batches = len(sim_data.batches) if max_batches is None else min(max_batches, len(sim_data.batches))
    
    # Build continuous time series across all batches
    all_times = []
    all_gas_use = []
    
    t_offset = 0
    cumul_gas_offset = 0
    
    for batch_idx in range(n_batches):
        batch = sim_data.batches[batch_idx]
        
        # Check if gas data is available
        if batch.Fg_list is None:
            raise ValueError("Gas flow rate data not available for all batches")
        
        # Get time points and adjust for offset
        t = np.asarray(batch.t_points, dtype=float)
        t_batch = t + t_offset
        
        # Calculate cumulative gas use at each timepoint within batch
        # Fg is in vvh (volume per volume per hour)
        # V is in L, dt is in hours
        V = np.asarray(batch.y_points[5], dtype=float)  # Volume in L
        Fg = np.asarray(batch.Fg_list, dtype=float)  # vvh
        
        # Calculate incremental gas use between timepoints
        dt = np.diff(t)  # h
        Fg_avg = (Fg[:-1] + Fg[1:]) / 2
        V_avg = (V[:-1] + V[1:]) / 2
        gas_increments = Fg_avg * V_avg * dt  # L
        
        # Create cumulative array for this batch
        gas_cumul_batch = np.zeros(len(t))
        gas_cumul_batch[1:] = np.cumsum(gas_increments)
        
        # Convert to m³ for more readable units (1 m³ = 1000 L)
        gas_cumul_batch = gas_cumul_batch / 1000 + cumul_gas_offset
        
        # Append to overall arrays
        all_times.extend(t_batch)
        all_gas_use.extend(gas_cumul_batch)
        
        # Update offsets for next batch
        t_offset = t_batch[-1]
        cumul_gas_offset = gas_cumul_batch[-1]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=all_times, y=all_gas_use, mode='lines',
                  line=dict(color='#17becf', width=2),
                  name='Cumulative Gas Use',
                  fill='tozeroy',
                  hovertemplate='Time: %{x:.1f} h<br>Gas Use: %{y:.2f} m³<extra></extra>')
    )
    
    # Get simulation type for title
    sim_type = getattr(sim_data, 'simulation_type', 'Simulation')
    
    # Update layout
    default_layout = dict(
        title_text=f"Cumulative Gas Use - {sim_type}",
        xaxis_title="Time (h)",
        yaxis_title="Gas Volume (m³)",
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
