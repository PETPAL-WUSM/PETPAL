"""Parametric kinetic modeling with Logan (reference region)"""
import pandas as pd
import ants
from petpal.kinetic_modeling import graphical_analysis
from petpal.utils.time_activity_curve import TimeActivityCurve
from petpal.utils.dimension import gen_3d_img_from_timeseries
from .kinetic_model_base import ParametricModel
from .logan_ref import LoganRefConfig
from ..meta.auto_cli import auto_cli


class LoganRefParametric(LoganRefConfig, ParametricModel):

    def __call__(self, input_image_path: str, out_image_prefix: str, mask_image_path: str, tacs_path: str, reference_region: str, t_star: float, k2_prime: float):
        """
        Fit all voxels in PET image with Logan reference kinetic model.

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
            t_star (str): Beginning model time for Logan reference.
            k2_prime (str): Average k2 value for the reference region, usually tracer-dependent.
        """
        self.set_required_pars(t_star=t_star, k2_prime=k2_prime)
        self.run_save_parametric_model(input_image_path=input_image_path,
                                       mask_image_path=mask_image_path,
                                       out_image_prefix=out_image_prefix,
                                       tacs_path=tacs_path,
                                       reference_region=reference_region)


def main():
    auto_cli(petpal_class=LoganRefParametric)

if __name__=='__main__':
    main()
