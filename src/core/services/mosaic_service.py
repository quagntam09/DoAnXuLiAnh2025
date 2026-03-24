from src.common.config import (
    DEFAULT_BLEND,
    DEFAULT_FRAME_EVERY,
    DEFAULT_LEVELS,
    DEFAULT_TILE_SIZE,
)
from src.common.exceptions import InvalidInputError
from src.core.algorithms.multiresolution import multi_resolution_mosaic


class MosaicService:
    def __init__(
        self,
        target_path,
        tiles_folder,
        tile_size=DEFAULT_TILE_SIZE,
        blend_factor=DEFAULT_BLEND,
        levels=DEFAULT_LEVELS,
        frame_every=DEFAULT_FRAME_EVERY,
    ):
        if not target_path:
            raise InvalidInputError("target_path không được để trống")
        if not tiles_folder:
            raise InvalidInputError("tiles_folder không được để trống")
        if int(tile_size) <= 0:
            raise InvalidInputError("tile_size phải > 0")
        if int(levels) <= 0:
            raise InvalidInputError("levels phải > 0")
        if int(frame_every) <= 0:
            raise InvalidInputError("frame_every phải > 0")

        blend = float(blend_factor)
        if blend < 0.0 or blend > 1.0:
            raise InvalidInputError("blend_factor phải nằm trong [0.0, 1.0]")

        self.target_path = target_path
        self.tiles_folder = tiles_folder
        self.tile_size = int(tile_size)
        self.blend_factor = blend
        self.levels = int(levels)
        self.frame_every = int(frame_every)

    def run(self, progress_callback, frame_callback=None):
        img, _ = multi_resolution_mosaic(
            target_path=self.target_path,
            tiles_folder=self.tiles_folder,
            base_tile=self.tile_size,
            levels=self.levels,
            blend_factor=self.blend_factor,
            progress_callback=progress_callback,
            frame_callback=frame_callback,
            frame_every=self.frame_every,
        )
        return img
