"""Normalize visualization result sources to simulation or batch data.

Visualization functions accept a
:class:`~animakinetica.core.simulation.CellCultureSimulation`, its associated
:class:`~animakinetica.core.simulation_data.SimulationData`, or an individual
:class:`~animakinetica.core.simulation_data.BatchData`. The helpers in this
module resolve those source forms consistently. Batch numbers are one-based;
when the source is already a ``BatchData`` instance, ``batch_number`` is
ignored.

These are internal helpers and are not a stable public API.
"""

from typing import TYPE_CHECKING, Union, Optional

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = ["get_batch_data", "get_simulation_data"]

if TYPE_CHECKING:
    from animakinetica.core.simulation_data import BatchData, SimulationData
    from animakinetica.core.simulation import CellCultureSimulation
else:
    # For runtime - avoid circular imports
    try:
        from animakinetica.core.simulation_data import BatchData
    except ImportError:
        BatchData = None


def get_batch_data(
    source: Union['CellCultureSimulation', 'SimulationData', 'BatchData'],
    batch_number: int
) -> Optional['BatchData']:
    """Resolve a result source to one batch record.

    Parameters
    ----------
    source : CellCultureSimulation, SimulationData, or BatchData
        Simulation result source. A ``CellCultureSimulation`` is resolved via
        its :attr:`~animakinetica.core.simulation.CellCultureSimulation.data`
        attribute. A ``SimulationData`` instance is queried with
        :meth:`~animakinetica.core.simulation_data.SimulationData.get_batch`.
        A ``BatchData`` instance is returned unchanged.
    batch_number : int
        One-based logical-batch identifier. Ignored when ``source`` is a
        ``BatchData`` instance.

    Returns
    -------
    BatchData or None
        Resolved batch record, or ``None`` when the requested batch is absent
        from a simulation or ``SimulationData`` source.

    Raises
    ------
    TypeError
        If ``source`` is not a supported result source.

    Notes
    -----
    The helper does not validate that ``batch_number`` is positive. Batch
    lookup follows :meth:`SimulationData.get_batch` and therefore matches the
    ``BatchData.batch_number`` field rather than a list position.
    """
    if isinstance(source, BatchData):
        return source
    elif hasattr(source, 'data'):  # CellCultureSimulation
        return source.data.get_batch(batch_number)
    elif hasattr(source, 'get_batch'):  # SimulationData
        return source.get_batch(batch_number)
    else:
        raise TypeError(f"Expected CellCultureSimulation, SimulationData, or BatchData, got {type(source)}")


def get_simulation_data(
    source: Union['CellCultureSimulation', 'SimulationData']
) -> 'SimulationData':
    """Resolve a result source to its aggregate data container.

    Parameters
    ----------
    source : CellCultureSimulation or SimulationData
        Simulation result source. A ``CellCultureSimulation`` is resolved via
        its :attr:`~animakinetica.core.simulation.CellCultureSimulation.data`
        attribute; a ``SimulationData`` instance is returned unchanged.

    Returns
    -------
    SimulationData
        Aggregate container that holds batch records and summary statistics.

    Raises
    ------
    TypeError
        If ``source`` is not a supported simulation or simulation-data source.

    """
    if hasattr(source, 'data'):  # CellCultureSimulation
        return source.data
    elif hasattr(source, 'batches'):  # SimulationData
        return source
    else:
        raise TypeError(f"Expected CellCultureSimulation or SimulationData, got {type(source)}")
