"""Tools to apply brain mask to PET images.
"""
import ants
from .motion_target import determine_motion_target

def brain_mask_pet(input_image_path: str,
                   out_image_path: str,
                   atlas_image_path: str,
                   atlas_mask_path: str,
                   motion_target_option='mean_image'):
    """
    Create a brain mask for a PET image. Create target PET image, which is then warped to a
    provided anatomical atlas. The transformation to atlas space is then applied to transform a
    provided mask in atlas space into PET space. This mask can then by used in various operations.

    Args:
        input_image_path (str): Path to input 4D PET image.
        out_image_path (str): Path to which brain mask in PET space is written.
        atlas_image_path (str): Path to anatomical atlas image.
        atlas_mask_path (str): Path to brain mask in atlas space.
        motion_target_option: Used to determine 3D target in PET space. Default 'mean_image'.
    
    Note:
        Requires access to an anatomical atlas or scan with a corresponding brain mask on said
        anatomical data. FSL users can use the MNI152 atlas and mask available at 
        $FSLDIR/data/standard/.
    """
    atlas = ants.image_read(atlas_image_path)
    atlas_mask = ants.image_read(atlas_mask_path)
    motion_target = determine_motion_target(motion_target_option=motion_target_option,
                                            input_image_path=input_image_path)
    pet_ref = ants.image_read(motion_target)
    xfm = ants.registration(
        fixed=atlas,
        moving=pet_ref,
        type_of_transform='SyN'
    )
    mask_on_pet = ants.apply_transforms(
        fixed=pet_ref,
        moving=atlas_mask,
        transformlist=xfm['invtransforms'],
        interpolator='nearestNeighbor'
    )
    mask = mask_on_pet.get_mask()
    ants.image_write(image=mask,filename=out_image_path)
