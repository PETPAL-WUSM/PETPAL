"""
Module for reading and writing images
"""
from typing import Optional
from collections.abc import Callable
import ants


class ImageLoader:
    """Class for reading image files, with extensions such as Nifti or MGZ.

    See also: :py:docs:`~ants.image_read`
    """
    def __init__(self, loader: Optional[Callable[[str], ants.ANTsImage]] = None):
        self._loader = loader or ants.image_read

    def load(self, filename: str) -> ants.ANTsImage:
        """Public read API that delegates to the configured reader"""
        return self._loader(filename)

    def __call__(self, filename: str) -> ants.ANTsImage:
        return self.load(filename=filename)
