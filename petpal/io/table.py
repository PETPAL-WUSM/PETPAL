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
import numpy as np

from ..utils.scan_timing import ScanTimingInfo
from ..utils.time_activity_curve import TimeActivityCurve


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
        """API that applies the table saving function assigned to `self._saver`.
        
        Args:
            df (pd.DataFrame): Pandas DataFrame with data to be saved.
            path (str): Path to file where data is saved.
        """
        self._saver(df, path)


class RegionalTacsLoader:
    """Load regional TACs from a spreadsheet"""
    def __init__(self):
        self.tacs_sheet = pd.DataFrame()

    def load_tacs_sheet(self,tacs_path: str):
        """Load TACs from a spreadsheet."""
        if Path(tacs_path).suffix=='.tsv':
            tacs_sheet = pd.read_csv(tacs_path, sep=r'\s+', engine='python')
        else:
            tacs_sheet = pd.read_csv(tacs_path, sep=None, engine='python')
        self.tacs_sheet = tacs_sheet
        self.validate_tacs_sheet_columns()
        self.normalize_tacs_sheet_types()
        return tacs_sheet

    def validate_tacs_sheet_columns(self):
        """Validate presence of required columns in the TACs sheet without mutating it.

        Checks that:
         - 'frame_start(min)' and 'frame_end(min)' exist
         - there is at least one region activity column (columns after the first two that do not
           end with '_unc')
         - for every region activity column there is a corresponding uncertainty column named
           '<region>_unc'
        """
        columns = self.tacs_sheet.columns
        required_time_cols = ['frame_start(min)', 'frame_end(min)']
        missing = [c for c in required_time_cols if c not in columns]
        if missing:
            raise KeyError(f"Missing required timing column(s): {missing}")

        region_candidates = [c for c in columns[2:] if not c.endswith('_unc')]
        if not region_candidates:
            raise KeyError("TACs sheet must include at least one region activity column (columns "
                           "after frame start/end).")

    def normalize_tacs_sheet_types(self):
        """Convert relevant TACs sheet columns to numeric and persist back to self.tacs_sheet.

        This method mutates self.tacs_sheet and raises ValueError if any required numeric
        column contains non-numeric or missing values after conversion.
        """
        tacs_sheet = self.tacs_sheet.copy()

        for col in ['frame_start(min)', 'frame_end(min)']:
            converted = pd.to_numeric(tacs_sheet[col], errors='coerce')
            if converted.isna().any():
                raise ValueError(f"Frame timing column '{col}' must contain numeric values for "
                                 "every frame.")
            tacs_sheet[col] = converted.astype(np.float64)

        region_activity_cols = [c for c in tacs_sheet.columns[2:] if not c.endswith('_unc')]
        for region_col in region_activity_cols:
            activity = pd.to_numeric(tacs_sheet[region_col], errors='coerce')
            if activity.isna().any():
                raise ValueError(f"Region activity column '{region_col}' must contain numeric "
                                 "values for every frame.")
            tacs_sheet[region_col] = activity.astype(np.float64)

            unc_col = f"{region_col}_unc"
            if unc_col in tacs_sheet.columns:
                uncertainty = pd.to_numeric(tacs_sheet[unc_col], errors='coerce')
                if uncertainty.isna().any():
                    raise ValueError(f"Uncertainty column '{unc_col}' must contain numeric values "
                                    "for every frame.")
                tacs_sheet[unc_col] = uncertainty.astype(np.float64)

        self.tacs_sheet = tacs_sheet

    def timing(self):
        """Get scan timing info from tacs sheet"""
        tacs_sheet = self.tacs_sheet
        frame_starts = tacs_sheet['frame_start(min)']
        frame_ends = tacs_sheet['frame_end(min)']
        scan_timing = ScanTimingInfo.from_start_end(frame_starts=frame_starts,
                                                    frame_ends=frame_ends)
        return scan_timing

    def get_regions(self):
        """Get list of regions based on column names.
        
        The first two columns in the TACs sheet are time starts and ends. Those are skipped.
        Remaining columns are each region name followed by its uncertainty."""
        regions_list = []
        tacs_sheet_columns = self.tacs_sheet.columns[2:]
        for column in tacs_sheet_columns:
            if '_unc' not in column:
                regions_list += [column]

        return regions_list

    def tacs_dictionary(self):
        """Organize DataFrame into dictionary of TAC objects"""
        tacs_sheet = self.tacs_sheet
        regions = self.get_regions()
        tacs = {}
        frame_starts = self.timing().start.to_numpy()
        for region in regions:
            region_activity = tacs_sheet[region].to_numpy()
            if f'{region}_unc' in tacs_sheet.columns:
                region_uncertainty = tacs_sheet[f'{region}_unc'].to_numpy()
                tac = TimeActivityCurve(times=frame_starts,
                                        activity=region_activity,
                                        uncertainty=region_uncertainty)
            else:
                tac = TimeActivityCurve(times=frame_starts,
                                        activity=region_activity)
            tacs[region] = tac
        return tacs


    def load(self, tacs_path: str):
        self.load_tacs_sheet(tacs_path=tacs_path)
        return self.tacs_dictionary()
