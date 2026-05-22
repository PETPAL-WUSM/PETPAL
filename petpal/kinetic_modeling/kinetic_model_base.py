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
from petpal.io.table import TableSaver, RegionalTacsLoader
from petpal.utils.dimension import gen_3d_img_from_timeseries


class ModelConfig:
    r"""
    Base class for config settings to apply to kinetic models
    """
    def __init__(self,
                 model_name: str,
                 model_solver: Callable,
                 required_pars: list[str],
                 fitted_pars: list[str],
                 tacs_loader: Optional[RegionalTacsLoader] = None,
                 table_saver: Optional[TableSaver] = None):
        r"""
        Initialize a TCM model configuration.
        """
        self.model_name = model_name
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


class ParametricModel(ModelConfig):

    def model_parametric_img(self, pet_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        img_dims = pet_arr.shape

        result_arr = np.zeros((img_dims[0],img_dims[1], img_dims[2], len(self.fitted_pars)), float)

        for i in range(0, img_dims[0], 1):
            for j in range(0, img_dims[1], 1):
                for k in range(0, img_dims[2], 1):
                    if mask_arr[i,j,k]>0:
                        voxel_tac = TimeActivityCurve(times=self.reference_tac.times,
                                                      activity=pet_arr[i,j,k,:])
                        result_arr[i,j,k,:] = self.run_model(reference_tac=self.reference_tac,
                                                             region_tac=voxel_tac)

        return result_arr

    def run_save_parametric_model(self,
                                  input_image_path: str,
                                  out_image_prefix: str,
                                  mask_image_path: str,
                                  tacs_path: str,
                                  reference_region: str):
        """Set up, run, and save parametric kinetic model.

        Args:
            input_image_path (str): Path to dynamic PET image on which Logan ref is used to model
                activity on each voxel.
            out_image_prefix (str): Directory and filename prefix for output images. Ensure to
                include the destination folder as well as the prefix. One image is written for each
                fitted parameter in the model.
            mask_image_path (str): Path to 3D mask image aligned with PET image, where positive
                mask values represent voxels where the kinetic model is calculated.
            tacs_path (str): Path to TACS spreadsheet including the reference region TAC.
            reference_region (str): Label for the reference region in the TACs spreadsheet.
        """
        input_img = ants.image_read(input_image_path)
        mask_img = ants.image_read(mask_image_path)
        pet_arr = input_img.numpy()
        mask_arr = mask_img.numpy()

        out_img_template = gen_3d_img_from_timeseries(input_img=input_img)
        self.set_tacs_data(tacs_path=tacs_path,
                           reference_region=reference_region)
        result_arr = self.model_parametric_img(pet_arr=pet_arr, mask_arr=mask_arr)
        for i, par in enumerate(self.fitted_pars):
            out_img = ants.from_numpy_like(result_arr[:,:,:,i], out_img_template)
            ants.image_write(out_img,
                             f"{out_image_prefix}_model-{self.model_name}_{par}.nii.gz")
