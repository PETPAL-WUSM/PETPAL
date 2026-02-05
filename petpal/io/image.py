"""
Module for loading and saving images
"""
from typing import Optional
from collections.abc import Callable
import ants


class ImageLoader:
    """Class for reading image files, with extensions such as Nifti or MGZ.

    See also: :py:docs:`~ants.image_read`.

    Example:

        .. code-block:: python

            from petpal.io.image import ImageLoader

            image_loader = ImageLoader()
            my_img = image_loader.load('/path/to/img.nii.gz')

    :ivar _loader: Function that loads an image file as an ants.ANTsImage object.
    """
    def __init__(self, loader: Optional[Callable[[str], ants.ANTsImage]] = None):
        self._loader = loader or ants.image_read

    def load(self, filename: str) -> ants.ANTsImage:
        """Public read API that delegates to the configured reader.
        
        Args:
            filename (str): Path to file that will be loaded as ANTsImage.
    
        Returns:
            img (ants.ANTsImage): Image object loaded into Python."""
        return self._loader(filename)

    def __call__(self, filename: str) -> ants.ANTsImage:
        return self.load(filename=filename)
