"""Solve MRTM2 voxelwise analysis."""
from .kinetic_model_base import ParametricModel
from ..meta.auto_cli import auto_cli
from .mrtm2 import Mrtm2Config

class Mrtm2Parametric(Mrtm2Config, ParametricModel):

    def __call__(self,
                 input_image_path: str,
                 out_image_prefix: str,
                 mask_image_path: str,
                 tacs_path: str,
                 reference_region: str,
                 k2_prime: float,
                 start_time: float,
                 end_time: float=600):
        """
        Fit all voxels in PET image with MRTM2 kinetic model.

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
            k2_prime (str): Average k2 value for the reference region, usually tracer-dependent.
            start_time (np.ndarray): Time point (in minutes) to begin logan model integration.
            end_time (np.ndarray): Time point (in minutes) to end logan model integration. Default
                600.
        """
        self.set_required_pars(k2_prime=k2_prime, start_time=start_time, end_time=end_time)
        self.run_save_parametric_model(input_image_path=input_image_path,
                                       mask_image_path=mask_image_path,
                                       out_image_prefix=out_image_prefix,
                                       tacs_path=tacs_path,
                                       reference_region=reference_region)

def main():
    auto_cli(petpal_class=Mrtm2Parametric)

if __name__=='__main__':
    main()
