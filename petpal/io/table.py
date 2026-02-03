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


def get_tabular_separator(ext: str) -> str:
    """Get the separator corresponding to a given tabular data filetype.
    
    '.csv' will return ',' while '.tsv' and '.txt' will return '\t'. Any other input will raise a
    ValueError.
    
    Args:
        ext (str): Extension to get matching separator for.
    
    Returns:
        sep (str): Separator matched from extension.
    
    Raises:
        ValueError: If extension is not .csv or .tsv.
    """
    matching_separators = {'.csv': ',', '.tsv': '\t', '.txt': '\t'}
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

    Example:

        .. code-block:: python

            import pandas as pd
            from petpal.io.table import TableSaver

            table_saver = TableSaver()
            my_data = pd.DataFrame(data={'time': [0, 1, 2], 'value': [1, 4, 9]})
            
            # when file extension is .csv, uses commas to separate values
            table_saver.save(my_data, 'table.csv')

            # when file extension is .tsv or .txt, uses tabs to separate values
            table_saver.save(my_data, 'table.txt')

    :ivar _saver: Injectable tabular data saving function that saves a dataframe to a file.
    """
    def __init__(self, saver: Optional[Callable[[pd.DataFrame, str], None]] = None):
        self._saver = saver or self._atomic_save

    def _atomic_save(self, df: pd.DataFrame, path: str):
        """Saves the data from a Pandas DataFrame object as a tabular file, such as CSV or TSV.
        
        Args:
            df (pd.DataFrame): Pandas DataFrame with data to be saved.
            path (str): Path to file where data is saved.
        """
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
        """API that applies the table saving function assigned to `self._saver`."""
        self._saver(df, path)
