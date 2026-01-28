"""
Module for reading and writing images
"""
from typing import Optional
from collections.abc import Callable
import ants


class ImageReader:
    """Class for reading image files, with extensions such as Nifti or MGZ.

    See also: :py:docs:`~ants.image_read`
    """
    def __init__(self, reader: Optional[Callable[[str], ants.ANTsImage]] = None):
        self._reader = reader or ants.image_read

    def load(self, filename: str) -> ants.ANTsImage:
        """Public read API that delegates to the configured reader"""
        return self._reader(filename)
