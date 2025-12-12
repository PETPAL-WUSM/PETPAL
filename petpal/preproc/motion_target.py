
"""Module with function to get a motion target for motion correction and registration"""
import os
import tempfile
import ants

from ..utils.useful_functions import get_average_of_timeseries
from .standard_uptake_value import weighted_sum_for_suv


def determine_motion_target(motion_target_option: str | tuple | list,
                            input_image_path: str = None) -> str:
    """
    Produce a motion target given the ``motion_target_option`` from a method
    running registrations on PET, i.e. :meth:`motion_correction` or
    :meth:`register_pet`.

    The motion target option can be a string or a tuple. If it is a string,
    then if this string is a file, use the file as the motion target.

    If it is the option ``weighted_series_sum``, then run
    :meth:`weighted_series_sum` and return the output path.

    If it is the option ``mean_image``, then compute the time-average of the
    4D-PET image.

    If it is a tuple, run a weighted sum on the PET series on a range of
    frames. The elements of the tuple are treated as times in seconds, counted
    from the time of the first frame, i.e. (0,300) would average all frames
    from the first to the frame 300 seconds later. If the two elements are the
    same, returns the one frame closest to the entered time.

    Args:
        motion_target_option (str | tuple | list): Determines how the method behaves,
            according to the above description. Can be a file, a method
            ('weighted_series_sum' or 'mean_image'), or a tuple range e.g. (0,600).
        input_image_path (str): Path to the PET image. This is intended to
            be supplied by the parent method employing this function. Default
            value None.
    Returns:
        out_image_file (str): File to use as a target to compute
            transformations on.

    Raises:
        ValueError: If ``motion_target_option`` does not match an acceptable option, or if 
        ``half_life`` is not specifiedwhen ``motion_target_option`` is not 'mean_image'
        TypeError: If start and end time are incompatible with ``float`` type.
    """
    if isinstance(motion_target_option, str):
        if os.path.exists(motion_target_option):
            return motion_target_option

        if motion_target_option == 'weighted_series_sum':
            out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
            weighted_sum_for_suv(input_image_path=input_image_path,
                                output_image_path=out_image_file)
            return out_image_file

        if motion_target_option == 'mean_image':
            out_image_file = tempfile.mkstemp(suffix='_mean.nii.gz')[1]
            input_img = ants.image_read(input_image_path)
            mean_img = get_average_of_timeseries(input_image=input_img)
            ants.image_write(image=mean_img,filename=out_image_file)
            return out_image_file

        raise ValueError("motion_target_option did not match a file or 'weighted_series_sum'")

    if isinstance(motion_target_option, (list, tuple)):

        start_time = motion_target_option[0]
        end_time = motion_target_option[1]

        try:
            float(start_time)
            float(end_time)
        except Exception as exc:
            raise TypeError('Start time and end time of calculation must be '
                            'able to be cast into float! Provided values are '
                            f"{start_time} and {end_time}.") from exc

        out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
        weighted_sum_for_suv(input_image_path=input_image_path,
                                output_image_path=out_image_file,
                                start_time=start_time,
                                end_time=end_time)

        return out_image_file

    raise ValueError('motion_target_option did not match str or tuple type.')