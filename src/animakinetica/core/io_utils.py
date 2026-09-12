"""Internal helpers for pickle serialization and path resolution.

Save paths are created relative to a caller-provided export root unless an
absolute path is supplied. Load operations first try the direct path and then
recursively search the root; when several files share a name, the most recently
modified file is selected.
"""

from pathlib import Path
import pickle
from typing import Any, Optional, Type, TypeVar, Union

__version__ = "1.0.0"
__author__ = "Ben Tumulero"
__all__ = [
    "normalize_pickle_filename",
    "resolve_pickle_save_path",
    "resolve_pickle_load_path",
    "save_pickle_object",
    "load_pickle_object",
]


T = TypeVar("T")


def normalize_pickle_filename(filename: Union[str, Path]) -> Path:
    """Normalize a pickle filename to a :class:`~pathlib.Path`.

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename or path, with or without a suffix.

    Returns
    -------
    pathlib.Path
        Path with a ``.pkl`` suffix. Existing non-``.pkl`` suffixes are
        replaced.
    """
    filepath = Path(filename)
    if filepath.suffix != ".pkl":
        filepath = filepath.with_suffix(".pkl")
    return filepath


def resolve_pickle_save_path(filename: Union[str, Path], export_root: Union[str, Path]) -> Path:
    """Resolve a pickle destination path.

    Parameters
    ----------
    filename : str or pathlib.Path
        Requested destination, with an optional suffix.
    export_root : str or pathlib.Path
        Directory prepended when ``filename`` is relative.

    Returns
    -------
    pathlib.Path
        Normalized destination path. This function does not create directories.
    """
    filepath = normalize_pickle_filename(filename)
    root = Path(export_root)
    if not filepath.is_absolute():
        filepath = root / filepath
    return filepath


def resolve_pickle_load_path(filename: Union[str, Path], search_root: Union[str, Path]) -> Path:
    """Resolve an existing pickle by direct lookup and recursive search.

    Relative names are first resolved under ``search_root``. If that path does
    not exist, the root is searched recursively by filename; the most recently
    modified match is returned.

    Parameters
    ----------
    filename : str or pathlib.Path
        Pickle filename or path, with an optional suffix.
    search_root : str or pathlib.Path
        Root directory for direct and recursive lookup.

    Returns
    -------
    pathlib.Path
        Existing pickle path.

    Raises
    ------
    FileNotFoundError
        If no matching pickle is found.
    """
    filepath = normalize_pickle_filename(filename)
    root = Path(search_root)

    attempted_direct = filepath if filepath.is_absolute() else root / filepath

    if filepath.is_absolute() and filepath.exists():
        return filepath

    if attempted_direct.exists():
        return attempted_direct

    if root.exists():
        matches = list(root.rglob(filepath.name))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if len(matches) > 1:
                print(f"Found {len(matches)} matching files. Using most recent: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        f"Pickle file not found. Attempted path: {attempted_direct}\n"
        f"Search root: {root} (recursive search enabled)."
    )


def save_pickle_object(obj: Any, filepath: Union[str, Path], overwrite: bool = False) -> Path:
    """Serialize an object with the highest available pickle protocol.

    Parameters
    ----------
    obj : Any
        Object to serialize. It must be pickle-compatible.
    filepath : str or pathlib.Path
        Destination path. Parent directories are created as needed.
    overwrite : bool, default=False
        Whether to replace an existing file.

    Returns
    -------
    pathlib.Path
        Written file path.

    Raises
    ------
    FileExistsError
        If the destination exists and ``overwrite`` is ``False``.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Pickle file already exists: {path}. Set overwrite=True to replace it."
        )

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_pickle_object(filepath: Union[str, Path], expected_type: Optional[Type[T]] = None) -> T:
    """Deserialize an object and optionally verify its type.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Existing pickle file to load.
    expected_type : type[T], optional
        Required type of the deserialized object.

    Returns
    -------
    T
        Deserialized object.

    Raises
    ------
    TypeError
        If the object is not an instance of ``expected_type``.
    """
    path = Path(filepath)
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if expected_type is not None and not isinstance(obj, expected_type):
        raise TypeError(
            f"Loaded object is {type(obj).__name__}, expected {expected_type.__name__}."
        )

    return obj