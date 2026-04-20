"""Parametric kinetic modeling with Logan (reference region)"""
import pandas as pd
import ants
from petpal.kinetic_modeling import graphical_analysis
from petpal.utils.time_activity_curve import TimeActivityCurve
from petpal.utils.dimension import gen_3d_img_from_timeseries
from ..kinetic_model_base import ParametricModel
from ..region_models.logan_ref import LoganRefConfig
from ...meta.auto_cli import auto_cli

class LoganRefParametric(LoganRefConfig, ParametricModel):

    def __call__(self, input_image_path: str, out_image_prefix: str, tacs_path: str, reference_region: str, t_star: float, k2_prime: float):
        """
        Fit all voxels in PET image with Logan reference kinetic model.

        Args:
            input_image_path (str): Path to dynamic PET image on which Logan ref is used to model
                activity on each voxel.
            out_image_prefix (str): Directory and filename prefix for output images. Ensure to
                include the destination folder as well as the prefix. One image is written for each
                fitted parameter in the model.
            tacs_path (str): Path to TACS spreadsheet including the reference region TAC.
            reference_region (str): Label for the reference region in the TACs spreadsheet.
            t_star (str): Beginning model time for Logan reference.
            k2_prime (str): Average k2 value for the reference region, usually tracer-dependent.
        """
        input_img = ants.image_read(input_image_path)
        self.tacs = self.tacs_loader.load(tacs_path=tacs_path)
        self.reference_tac = self.tacs[reference_region]
        self.set_required_pars(t_star=t_star, k2_prime=k2_prime)
        pet_arr = input_img.numpy()
        result_arr = self.run_parametric_model(pet_arr=pet_arr)
        out_img_template = gen_3d_img_from_timeseries(input_img=input_img)
        for i, par in enumerate(self.fitted_pars):
            out_img = ants.from_numpy_like(result_arr[:,:,:,i], out_img_template)
            ants.image_write(out_img, f"{out_image_prefix}_model-LoganRef_{par}.nii.gz")


def main():
    auto_cli(petpal_class=LoganRefParametric)

if __name__=='__main__':
    main()
