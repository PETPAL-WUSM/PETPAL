"""Merge TACs from two different spreadsheets and save the result. Useful when working with
multiple types of segmentations in one study."""
import pandas as pd
from ..io.table import TableSaver, RegionalTacsLoader
from ..meta.auto_cli import auto_cli


class MergeTacSheets:
    def __init__(self):
        self.tacs_loader = RegionalTacsLoader()
        self.table_saver = TableSaver()

    def validate_tac_sheet_timing_identical(self, tac_sheet_left: pd.DataFrame, tac_sheet_right: pd.DataFrame):
        tac_sheet_left_times = tac_sheet_left['frame_start(min)']
        tac_sheet_right_times = tac_sheet_right['frame_start(min)']
        
        try:
            (tac_sheet_left_times==tac_sheet_right_times).all()
        except ValueError:
            raise ValueError("TAC sheet timing labels do not match. Left sheet length: "
                            f"{tac_sheet_left_times.size}, Right sheet length: "
                            f"{tac_sheet_right_times.size}")

        if not (tac_sheet_left_times==tac_sheet_right_times).all():
            raise ValueError("TAC sheet timing labels match, but the values are not identical.")

    def __call__(self, tac_sheet_left_path: str, tac_sheet_right_path: str, out_merged_tacs_path: str):
        """Merge TACs from two different spreadsheets and save the result. Useful when working with
        multiple types of segmentations in one study.
        
        Args:
            tac_sheet_left_path (str): Path to a regional multitac spreadsheet.
            tac_sheet_right_path (str): Path to a distinct regional multitac spreadsheet
                with one or more regions not included in the left TACs spreadsheet.
            out_merged_tacs_path (str): Path to where merged regional multitac spreadsheet
                will be saved.
        """
        tac_sheet_left = self.tacs_loader.load_tacs_sheet(tacs_path=tac_sheet_left_path)
        tac_sheet_right = self.tacs_loader.load_tacs_sheet(tacs_path=tac_sheet_right_path)

        self.validate_tac_sheet_timing_identical(tac_sheet_left, tac_sheet_right)

        tac_sheet_merged = pd.merge(left=tac_sheet_left, right=tac_sheet_right)
        self.table_saver.save(df=tac_sheet_merged, path=out_merged_tacs_path)

def main():
    auto_cli(petpal_class=MergeTacSheets)

if __name__=='__main__':
    main()
