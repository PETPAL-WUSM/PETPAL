"""
Provides functions for undo-ing decay correction and recalculating it.

"""
from typing import Optional
import math

import ants
import numpy as np
import pandas as pd

from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo
from ..io.image import ImageLoader
from ..utils.image_io import safe_copy_meta
from petpal.utils.constants import HALF_LIVES
from ..meta.auto_cli import auto_cli

def undo_decay_correction(input_image_path: str,
                          output_image_path: str,
                          metadata_dict: dict = None) -> ants.ANTsImage:
    """Uses decay factors from the metadata for an image to remove decay correction for each frame.

    This function expects to find decay factors in the .json sidecar file, or the metadata_dict, if given. If there are
    no decay factors (either under the key 'DecayFactor' or the BIDS-required 'DecayCorrectionFactor') listed, it may
    result in unexpected behavior. In addition to returning an ANTsImage containing the "decay uncorrected" data, the
    function writes an image to output_image_path, unless it is passed as 'None'.

    Args:
        input_image_path (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.
        output_image_path (str): Path to output (.nii.gz or .nii) output image. If None, no image will be written.
        metadata_dict (dict, optional): Metadata dictionary to use instead of corresponding .json sidecar. If not
            specified (default behavior), function will try to use sidecar .json in the same directory as
            input_image_path.

    Returns:
        ants.ANTsImage: ANTsImage with decay correction reversed."""

    decay_corrected_image = ants.image_read(filename=input_image_path)

    if metadata_dict is not None:
        json_data = metadata_dict
    else:
        json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)

    frame_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
    decay_factors = frame_info.decay

    uncorrected_image_numpy = decay_corrected_image.numpy()

    for frame_num, decay_factor in enumerate(decay_factors):
        uncorrected_image_numpy[..., frame_num] /= decay_factor

    uncorrected_image = ants.from_numpy_like(data=uncorrected_image_numpy,
                                             image=decay_corrected_image)

    if output_image_path is not None:
        ants.image_write(image=uncorrected_image,
                         filename=output_image_path)

        json_data['DecayFactor'] = list(np.ones_like(decay_factors))
        json_data['ImageDecayCorrected'] = "false"
        output_json_path = image_io.gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
        image_io.write_dict_to_json(meta_data_dict=json_data,
                                    out_path=output_json_path)

    return uncorrected_image


def decay_correct(input_image_path: str,
                  output_image_path: str) -> ants.ANTsImage:
    r"""Recalculate decay_correction for nifti image based on frame reference times.

    This function will compute frame reference times based on frame time starts and frame durations (both of which
    are required by BIDS. These reference times are used in the following equation to determine the decay factor for
    each frame. For more information, refer to Turku Pet Centre's materials at
    https://www.turkupetcentre.net/petanalysis/decay.html

    .. math::
        decay\_factor = \exp(\lambda*t)

    where :math:`\lambda=\log(2)/T_{1/2}` is the decay constant of the radio isotope and depends on its half-life and
    `t` is the frame's reference time with respect to TimeZero (ideally, injection time).

    Note: BIDS 'DecayCorrectionTime' is set to 0 (seconds w.r.t. TimeZero) for the image. If this assumption doesn't
        hold, be wary of downstream effects.

    Args:
        input_image_path (str): Path to input (.nii.gz or .nii) image. A .json sidecar file should exist in the same
             directory as the input image.
        output_image_path (str): Path to output (.nii.gz or .nii) output image.

    Returns:
        ants.ANTsImage: Decay-Corrected Image

    """
    half_life = image_io.get_half_life_from_nifti(image_path=input_image_path)

    json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)
    uncorrected_image = ants.image_read(filename=input_image_path)

    frame_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
    frame_reference_times = np.asarray(frame_info.start + frame_info.duration / 2.0, float).tolist()

    original_decay_factors = frame_info.decay
    if np.any(original_decay_factors != 1):
        raise ValueError(f'Decay Factors other than 1 found in metadata for {input_image_path}. This likely means the '
                         f'image has not had its previous decay correction undone. Try running undo_decay_correction '
                         f'before running this function to avoid decay correcting an image more than once.')

    corrected_data = uncorrected_image.numpy()
    new_decay_factors = []
    for frame_num, frame_reference_time in enumerate(frame_reference_times):
        new_decay_factor = math.exp(((math.log(2) / half_life) * frame_reference_time))
        corrected_data[..., frame_num] *= new_decay_factor
        new_decay_factors.append(new_decay_factor)

    corrected_image = ants.from_numpy_like(data=corrected_data,
                                           image=uncorrected_image)

    if output_image_path is not None:
        ants.image_write(image=corrected_image,
                         filename=output_image_path)
        output_json_path = image_io.gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
        json_data['DecayFactor'] = new_decay_factors
        json_data['ImageDecayCorrected'] = "true"
        json_data['ImageDecayCorrectionTime'] = 0
        json_data['FrameReferenceTime'] = frame_reference_times
        image_io.write_dict_to_json(meta_data_dict=json_data,
                                    out_path=output_json_path)

    return corrected_image


def calculate_frame_decay_factor(frame_reference_time: np.ndarray,
                                 half_life: float) -> np.ndarray:
    """Calculate decay correction factors for a scan given the frame reference time and half life.
    
    Important: 
        The frame reference time should be the time at which average activity occurs,
        not simply the midpoint. See
        :meth:`~petpal.utils.scan_timing.calculate_frame_reference_time` for more info.

    Args: 
        frame_reference_time (np.ndarray): Time at which the average activity occurs for the frame.
        half_life (float): Radionuclide half life.

    Returns: 
        np.ndarray: Decay Correction Factors for each frame in the scan.
    """
    decay_constant = np.log(2)/half_life
    frame_decay_factor = np.exp((decay_constant)*frame_reference_time)
    return frame_decay_factor


def scale_frames(input_img: ants.ANTsImage, scalar_arr: np.ndarray[float]):
    nframes = input_img.shape[-1]
    nscalar = len(scalar_arr)
    if nframes!=nscalar:
        raise ValueError(f"Length of correction factors ({nscalar}) does not "
                         f"match number of frames in dynamic PET ({nframes}).")
    modified_arr = ants.image_clone(input_img)
    for frame, scalar in enumerate(scalar_arr):
        modified_arr[:,:,:,frame] *= scalar
    return modified_arr


class DecayCorrect:
    """Decay correct or uncorrect each frame in a dynamic PET scan.
    
    :ivar image_loader: The image loader to use.
    :ivar modified_pet_img: The corrected PET image."""
    def __init__(self,
                 image_loader: Optional[ImageLoader] = None):
        self.image_loader = image_loader or ImageLoader()
        self.modified_pet_img: ants.ANTsImage = None

    def apply_factor(self,
                     input_image_path: str,
                     correction_factor: np.ndarray[float]):
        """Apply fix factor to image and write output.

        Args:
            input_image_path (str): Path to dynamic PET image.
            correction_factor_path (str): Path to file with correction factors, one per frame. 
                File must have exactly one column, with a header followed by one scalar per
                line."""

        input_img = self.image_loader.load(filename=input_image_path)
        self.modified_pet_img = scale_frames(input_img=input_img, scalar_arr=correction_factor)
    
    def save_modified_pet(self, input_image_path: str, output_image_path: str):
        """Save the modified PET image

        Args:
            input_image_path (str): Path to dynamic PET image.
            output_image_path (str): Path to where corrected image is saved.
        """
        ants.image_write(self.modified_pet_img, output_image_path)
        safe_copy_meta(input_image_path,out_image_path=output_image_path)

    def decay_correct_factor_from_file(self,
                                       input_image_path: str,
                                       output_image_path: str,
                                       correction_factor_path: str):
        """Apply a set of correction factors to each frame in a PET image.
        
        Provide path to dynamic PET image, path to where corrected image is saved, and path to a
        file containing factors for each frame. File containing correction factors must have
        exactly one column, with a header followed by one scalar per line to apply to the
        corresponding frame in the PET. Reads image and correction factors, applies them, and saves
        result.
        
        Args:
            input_image_path (str): Path to dynamic PET image.
            output_image_path (str): Path to where corrected image is saved.
            correction_factor_path (str): Path to file with correction factors, one per frame. 
                File must have exactly one column, with a header followed by one scalar per
                line.
        """
        correction_factor = pd.read_csv(correction_factor_path,index_col=False).iloc[:,0]
        self.apply_factor(input_image_path=input_image_path,
                          correction_factor=correction_factor)
        self.save_modified_pet(input_image_path=input_image_path,
                               output_image_path=output_image_path)


class DecayFix(DecayCorrect):
    """Special case for decay correction where the dynamic PET image was scaled with the wrong
    isotope but JSON metadata retains accurate decay correction factors."""
    def fix_factor(self,
                   input_image_path: str,
                   isotope_to_remove: str) -> np.ndarray[float]:
        """Get ratio of the expected decay correction factor to the factor that was incorrectly
        applied to PET image.
        
        Args:
            input_image_path (str): Path to dynamic PET image.
            isotope_to_remove (str): Name of the isotope that was incorrectly used to scale dynamic
                PET image, such as 'o15' or 'c11'.
        
        Returns:
            fix_factor (np.ndarray[float]): Scalar array with the correction factor to apply to
                each frame in dynamic PET.
        """
        scan_timing = ScanTimingInfo.from_nifti(input_image_path)
        decay_expected = scan_timing.decay
        decay_applied = calculate_frame_decay_factor(scan_timing.center,
                                                     HALF_LIVES[isotope_to_remove])
        return decay_expected/decay_applied

    def __call__(self,                 
                input_image_path: str,
                output_image_path: str,
                isotope_to_remove: str):
        """Calculate correction factor based on the accurate metadata and name of the isotope
        incorrectly used to scale image data.

        Args:
            input_image_path (str): Path to dynamic PET image.
            output_image_path (str): Path to where corrected image is saved.
            isotope_to_remove (str): Name of the isotope that was incorrectly used to scale dynamic
                PET image, such as 'o15' or 'c11'.
        """
        correction_factor = self.fix_factor(input_image_path=input_image_path,
                                            isotope_to_remove=isotope_to_remove)
        self.apply_factor(input_image_path=input_image_path,
                          correction_factor=correction_factor)
        self.save_modified_pet(input_image_path=input_image_path,
                               output_image_path=output_image_path)

def main():
    auto_cli(petpal_class=DecayFix)

if __name__=='__main__':
    main()
