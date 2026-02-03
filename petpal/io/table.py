"""
Module for reading and writing tables as TSV and CSV files.
"""
import os
import tempfile
from typing import Optional
from collections.abc import Callable
from pathlib import Path
import dataclasses
import pandas as pd

from ..utils.useful_functions import coerce_outpath_extension


def get_tabular_separator(ext: str) -> str:
    """Get the separator corresponding to a given tabular data filetype.
    
    '.csv' will return ',' while '.tsv' will return '\t'. Any other input will raise a
    ValueError.
    
    Args:
        ext (str): Extension to get matching separator for.
    
    Returns:
        sep (str): Separator matched from extension.
    
    Raises:
        ValueError: If extension is not .csv or .tsv.
    """
    matching_separators = {'.csv': ',', '.tsv': '\t'}
    try:
        return matching_separators[ext]
    except ValueError as exc:
        error_msg = f"Only accepted extensions are {matching_separators.keys()}. Got {ext}."
        raise ValueError(error_msg) from exc


@dataclasses.dataclass
class TableSaver:
    """
    Class for saving Pandas Database objects as CSV or TSV files based on a provided path.

    - Default behavior writes atomically (write temp file + os.replace) to avoid partial files.
    - Accepts an injectable writer callable for testing or alternative persistence backends.
    """
    def __init__(self, saver: Optional[Callable[[pd.DataFrame, str], None]] = None):
        self._saver = saver or self._atomic_save

    def _atomic_save_csv(self, df: pd.DataFrame, path: str) -> None:
        """Write CSV via a temporary file and os.replace."""
        csv_path = coerce_outpath_extension(path=path, ext='.csv')
        self._atomic_save(df=df, path=csv_path)

    def _atomic_save_tsv(self, df: pd.DataFrame, path: str) -> None:
        """Write CSV via a temporary file and os.replace."""
        tsv_path = coerce_outpath_extension(path=path, ext='.tsv')
        self._atomic_save(df=df, path=tsv_path)

    def _atomic_save(self, df: pd.DataFrame, path: str):
        dirpath = os.path.dirname(os.path.abspath(path)) or "."
        suffix = Path(path).suffix
        sep = get_tabular_separator(ext=suffix)
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_petpal_", dir=dirpath, suffix=suffix)
        os.close(fd)
        try:
            df.to_csv(tmp_path, sep=sep)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def save(self, df: pd.DataFrame, path: str) -> None:
        """Public CSV write API that delegates to the configured writer."""
        self._saver(df, path)
