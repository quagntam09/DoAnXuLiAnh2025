from algorithms.multiresolution import multi_resolution_mosaic

class MosaicGenerator:
    def __init__(self, target_path, tiles_folder, tile_size, blend_factor,
                    levels=3,
                    frame_every=120):
        self.target_path = target_path
        self.tiles_folder = tiles_folder
        self.tile_size = tile_size
        self.blend_factor = blend_factor
        self.levels = levels
        self.frame_every = frame_every

    def run(self, progress_callback, frame_callback=None):
        img, _ = multi_resolution_mosaic(
            target_path=self.target_path,
            tiles_folder=self.tiles_folder,
            base_tile=self.tile_size,
            levels=self.levels,
            blend_factor=self.blend_factor,
            progress_callback=progress_callback,
            frame_callback=frame_callback,
            frame_every=self.frame_every
        )
        return img