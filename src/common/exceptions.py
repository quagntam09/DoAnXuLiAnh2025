class MosaicError(Exception):
    """Base exception for mosaic app."""


class InvalidInputError(MosaicError):
    """Raised when user input is invalid."""


class ImageLoadError(MosaicError):
    """Raised when image cannot be loaded."""


class TileDatasetError(MosaicError):
    """Raised when tile dataset is missing or invalid."""
