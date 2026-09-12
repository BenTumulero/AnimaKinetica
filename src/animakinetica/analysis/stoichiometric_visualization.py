"""Plot stoichiometric yield-coefficient histories for a simulated batch.

The :func:`plot_stoichiometric_yields` function visualizes the seven yield
coefficients recorded from
:class:`~animakinetica.models.stoichiometry.StoichiometricModel` during ODE
integration. It accepts a completed
:class:`~animakinetica.core.simulation.CellCultureSimulation`, its
:class:`~animakinetica.core.simulation_data.SimulationData`, or an individual
:class:`~animakinetica.core.simulation_data.BatchData` instance.

Yield histories are available only when detailed batch logging is enabled.
Run :meth:`~animakinetica.core.simulation.CellCultureSimulation.run_simulation`
with stored batch data; performance-mode evaluations do not retain these
traces. Batch identifiers are one-based unless ``source`` is already a
``BatchData`` instance.
"""

from typing import Optional, Dict, Union, TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from animakinetica.analysis._visualization_utils import get_batch_data

if TYPE_CHECKING:
    from animakinetica.core.simulation import CellCultureSimulation
    from animakinetica.core.simulation_data import SimulationData, BatchData

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = [
    "plot_stoichiometric_yields",
]


def plot_stoichiometric_yields(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int = 1,
    layout_kwargs: Optional[Dict] = None
) -> go.Figure:
    """Plot the seven recorded stoichiometric yield coefficients for one batch.

    The figure contains ``Y_Glc/X``, ``Y_Lac/Glc``, ``Y_Gln/Glc``,
    ``Y_O2/Glc``, ``Y_CO2/Glc``, ``Y_AA/Glc``, and ``Y_Amm/Gln``. The data are
    read from the corresponding
    :class:`~animakinetica.core.simulation_data.BatchData` ``*_list`` fields,
    which are interpolated onto the accepted ODE-solver time points.

    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Completed simulation, aggregate simulation results, or one batch of
        results. A ``CellCultureSimulation`` is resolved through its
        :attr:`~animakinetica.core.simulation.CellCultureSimulation.data`
        attribute. ``batch_number`` is ignored for a ``BatchData`` instance.
    batch_number : int, default=1
        One-based logical-batch identifier for a simulation or
        ``SimulationData`` source.
    layout_kwargs : dict, optional
        Plotly layout keyword arguments merged into the default layout. Values
        supplied here override the default title, dimensions, template, hover
        mode, and legend visibility.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3-by-3 subplot figure. Seven panels contain yield time
        courses in mol/mol; the final two panels are empty.

    Raises
    ------
    TypeError
        If ``source`` is not a supported simulation-result object.
    ValueError
        If the selected batch does not exist, or if any yield history is
        unavailable. Re-run the simulation with detailed batch logging enabled
        when histories are absent.

    Notes
    -----
    ``Y_Glc/X`` is glucose consumed per dry-cell-mass (DCM) amount;
    all other displayed coefficients are molar ratios indicated by their panel
    labels. For fixed stoichiometry, the trajectories are normally constant.
    Under excess-coupled stoichiometry, they can vary with the kinetic state.
    """
    batch_data = get_batch_data(source, batch_number)
    if batch_data is None:
        raise ValueError(f"Batch {batch_number} not found")

    t = batch_data.t_points

    y_series = {
        "Y_Glc/X": batch_data.Y_Glc_X_list,
        "Y_Lac/Glc": batch_data.Y_Lac_Glc_list,
        "Y_Gln/Glc": batch_data.Y_Gln_Glc_list,
        "Y_O2/Glc": batch_data.Y_O2_Glc_list,
        "Y_CO2/Glc": batch_data.Y_CO2_Glc_list,
        "Y_AA/Glc": batch_data.Y_AA_Glc_list,
        "Y_Amm/Gln": batch_data.Y_Amm_Gln_list,
    }

    missing = [name for name, values in y_series.items() if values is None]
    if missing:
        raise ValueError(
            "Missing stoichiometric yield arrays in BatchData: "
            + ", ".join(missing)
            + ". Re-run simulation with detailed logging enabled."
        )

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Y_Glc/X",
            "Y_Lac/Glc",
            "Y_Gln/Glc",
            "Y_O2/Glc",
            "Y_CO2/Glc",
            "Y_AA/Glc",
            "Y_Amm/Gln",
            "",
            "",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    traces = [
        ("Y_Glc/X", "#1F77B4", 1, 1),
        ("Y_Lac/Glc", "#FF7F0E", 1, 2),
        ("Y_Gln/Glc", "#2CA02C", 1, 3),
        ("Y_O2/Glc", "#D62728", 2, 1),
        ("Y_CO2/Glc", "#9467BD", 2, 2),
        ("Y_AA/Glc", "#8C564B", 2, 3),
        ("Y_Amm/Gln", "#17BECF", 3, 1),
    ]

    for series_name, color, row, col in traces:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_series[series_name],
                mode="lines",
                name=series_name,
                line=dict(color=color, width=2),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Time (h)", row=row, col=col)
        fig.update_yaxes(title_text="Yield (mol/mol)", row=row, col=col)

    default_layout = dict(
        title=f"Stoichiometric Yield Coefficients - Batch {batch_number}",
        title_x=0.5,
        height=900,
        width=1300,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )

    if layout_kwargs:
        default_layout.update(layout_kwargs)

    fig.update_layout(**default_layout)
    return fig
