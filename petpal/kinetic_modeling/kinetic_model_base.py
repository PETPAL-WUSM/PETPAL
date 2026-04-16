"""Base classes for kinetic analysis"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Sequence, Protocol, Dict, Type, Any, Optional, Callable
import tempfile
import os
import dataclasses
import numpy as np
import pandas as pd
import ants
from petpal.kinetic_modeling import graphical_analysis, reference_tissue_models
from petpal.utils.time_activity_curve import TimeActivityCurve
from petpal.utils.scan_timing import ScanTimingInfo
from petpal.io.table import TableSaver


class RegionalTacsLoader:
    """Load regional TACs from a spreadsheet"""
    def __init__(self):
        self.tacs_sheet = pd.DataFrame()

    def load_tacs_sheet(self,tacs_path: str):
        """Load TACs from a spreadsheet."""
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

        missing_unc = [f"{r}_unc" for r in region_candidates if f"{r}_unc" not in columns]
        if missing_unc:
            raise KeyError(f"Missing uncertainty column(s) for region(s): {missing_unc}")

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
            unc_col = f"{region_col}_unc"
            uncertainty = pd.to_numeric(tacs_sheet[unc_col], errors='coerce')
            if uncertainty.isna().any():
                raise ValueError(f"Uncertainty column '{unc_col}' must contain numeric values "
                                 "for every frame.")
            tacs_sheet[region_col] = activity.astype(np.float64)
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
            region_uncertainty = tacs_sheet[f'{region}_unc'].to_numpy()
            tac = TimeActivityCurve(times=frame_starts,
                                    activity=region_activity,
                                    uncertainty=region_uncertainty)
            tacs[region] = tac
        return tacs


    def load(self, tacs_path: str):
        self.load_tacs_sheet(tacs_path=tacs_path)
        return self.tacs_dictionary()


class ModelConfig:
    r"""
    Base class for config settings to apply to kinetic models
    """
    def __init__(self,
                 model_solver: Callable,
                 required_pars: list[str],
                 fitted_pars: list[str],
                 tacs_loader: Optional[RegionalTacsLoader] = None,
                 table_saver: Optional[TableSaver] = None):
        r"""
        Initialize a TCM model configuration.
        """
        self.model_solver = model_solver
        self.required_pars = required_pars
        self.fitted_pars = fitted_pars
        self.num_params = len(required_pars)
        self.model_pars = None
        self.table_saver = table_saver or TableSaver()
        self.tacs_loader = tacs_loader or RegionalTacsLoader()


    def set_required_pars(self, **pars):
        """Set each parameter in required_pars"""
        model_pars = namedtuple('Pars',self.required_pars)
        self.model_pars = model_pars(**pars)

    def null_result(self) -> np.ndarray:
        """Return a numpy array of NaNs indexed by the model's fitted_pars."""
        n_fitted_pars = len(self.fitted_pars)
        fits = np.full(n_fitted_pars,np.nan)
        return fits

    @abstractmethod
    def run_model(self, reference_tac, region_tac) -> Sequence[float]:
        ...

    @staticmethod
    def normalize_name(name: str) -> str:
        r"""
        Normalize a model name to a standard format.

        Converts the name to lowercase and replaces spaces and underscores with hyphens
        for consistent model name lookup.

        Args:
            name (str): The model name to normalize.

        Returns:
            str: The normalized model name.

        Example:
            .. code-block:: python

                from petpal.kinetic_modeling.tac_fitting import TcmModelConfig
                TcmModelConfig.normalize_name("Serial 2TCM")  # Returns "serial-2tcm"
                TcmModelConfig.normalize_name("1TCM")  # Returns "1tcm"
        """
        return name.lower().replace(' ', '_').replace('_', '-')

    def fit_regions(self) -> pd.DataFrame:
        """Run the kinetic model on all of the regions"""
        tacs = self.tacs
        fit_results = pd.DataFrame(index=tacs.keys(), columns=self.fitted_pars)
        for region,tac in tacs.items():
            try:
                region_fit = self.run_model(reference_tac=self.reference_tac,
                                            region_tac=tac)
            except Exception:
                region_fit = self.null_result()
            fit_results.loc[region,:] = region_fit
        return fit_results


class LoganRefConfig(ModelConfig):
    """Config settings for logan reference tissue"""
    def __init__(self):
        super().__init__(model_solver=graphical_analysis.logan_ref_region_analysis_with_rsquared,
                         required_pars=["t_star","k2_prime"],
                         fitted_pars=['DVR','Intercept','R-squared','BP'])


    def run_model(self,
                  reference_tac: TimeActivityCurve,
                  region_tac: TimeActivityCurve):
        """Run logan reference"""
        fits = self.model_solver(tac_times_in_minutes=reference_tac.times,
                        input_tac_values=reference_tac.activity,
                        region_tac_values=region_tac.activity,
                        t_thresh_in_minutes=self.model_pars.t_star,
                        k2_prime=self.model_pars.k2_prime)
        bp = fits[0] - 1
        fit_result = [*fits, bp]
        return fit_result

    def __call__(self,
                 reference_region,
                 regional_tacs_path,
                 save_path,
                 t_star: float,
                 k2_prime: float):
        self.tacs = self.tacs_loader.load(tacs_path=regional_tacs_path)
        self.reference_tac = self.tacs[reference_region]
        self.set_required_pars(t_star=t_star, k2_prime=k2_prime)
        fit_results = self.fit_regions()
        self.table_saver.save(fit_results, save_path)

class ParametricModel(ModelConfig):

    def run_parametric_model(self, pet_arr: np.ndarray):
        img_dims = pet_arr.shape

        result_arr = np.zeros((img_dims[0],img_dims[1], img_dims[2], len(self.fitted_pars)), float)

        for i in range(0, img_dims[0], 1):
            for j in range(0, img_dims[1], 1):
                for k in range(0, img_dims[2], 1):
                    voxel_tac = TimeActivityCurve(times=self.reference_tac.times,
                                                    activity=pet_arr[i,j,k,:])
                    result_arr[i,j,k,:] = self.run_model(reference_tac=self.reference_tac,
                                                    region_tac=voxel_tac)

        return result_arr

class LoganRefParametric(LoganRefConfig):

    def run_parametric_model(self, pet_arr: np.ndarray) -> np.ndarray:
        img_dims = pet_arr.shape

        result_arr = np.zeros((img_dims[0],img_dims[1], img_dims[2], len(self.fitted_pars)), float)

        for i in range(0, img_dims[0], 1):
            for j in range(0, img_dims[1], 1):
                for k in range(0, img_dims[2], 1):
                    voxel_tac = TimeActivityCurve(times=self.reference_tac.times,
                                                  activity=pet_arr[i,j,k,:])
                    result_arr[i,j,k,:] = self.run_model(reference_tac=self.reference_tac,
                                                         region_tac=voxel_tac)

        return result_arr

    def __call__(self, input_image_path: str, out_image_path: str, tacs_path: str, reference_region: str, t_star: float, k2_prime: float):
        input_img = ants.image_read(input_image_path)
        self.tacs = self.tacs_loader.load(tacs_path=tacs_path)
        self.reference_tac = self.tacs[reference_region]
        self.set_required_pars(t_star=t_star, k2_prime=k2_prime)
        pet_arr = input_img.numpy()
        result_arr = self.run_parametric_model(pet_arr=pet_arr)
        out_img = ants.from_numpy(result_arr,
                                  input_img.origin,
                                  input_img.spacing,
                                  input_img.direction)
        ants.image_write(out_img, out_image_path)
