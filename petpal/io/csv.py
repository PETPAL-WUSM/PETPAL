"""
Module for reading and writing csv files.
"""
import os
import tempfile
from typing import Optional
from collections.abc import Callable
from pathlib import Path
import dataclasses
import pandas as pd

@dataclasses.dataclass
class CsvWriter:
    """
    Class for writing Pandas Database objects to CSV based on a provided path.

    - Default behavior writes atomically (write temp file + os.replace) to avoid partial files.
    - Accepts an injectable writer callable for testing or alternative persistence backends.
    """
    def __init__(self, writer: Optional[Callable[[pd.DataFrame, str], None]] = None):
        self._writer = writer or self._atomic_write

    @staticmethod
    def outpath_as_csv(path: str):
        """Coerce output path to .csv suffix."""
        path_obj = Path(path)
        while path_obj.suffix!='':
            path_obj = path_obj.with_suffix('')
        path_obj_csv_suffix = path_obj.with_suffix('.csv')
        return str(path_obj_csv_suffix.absolute())

    def _atomic_write(self, df: pd.DataFrame, path: str) -> None:
        """Write CSV atomically via a temporary file and os.replace."""
        csv_path = self.outpath_as_csv(path=path)
        dirpath = os.path.dirname(os.path.abspath(csv_path)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_petpal_", dir=dirpath, suffix=".csv")
        os.close(fd)
        try:
            df.to_csv(tmp_path)
            os.replace(tmp_path, csv_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def write(self, df: pd.DataFrame, path: str) -> None:
        """Public CSV write API that delegates to the configured writer."""
        self._writer(df, path)
