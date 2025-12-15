import argparse
import os
import cv2

from mosaic_core import MosaicGenerator
from utils.progress import console_progress
from utils.uio import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Photomosaic Generator (avg color + KDTree + multi-resolution)")
    p.add_argument("--target", type=str, default="data/target/image.jpg", help="Đường dẫn ảnh gốc")
    p.add_argument("--tiles", type=str, default="data/tiles", help="Thư mục chứa ảnh tiles")
    p.add_argument("--tile", type=int, default=10, help="Base tile size (level nhỏ nhất), ví dụ 10")
    p.add_argument("--levels", type=int, default=3, help="Số levels multi-resolution (>=1)")
    p.add_argument("--blend", type=float, default=0.2, help="Blend factor 0..1")
    p.add_argument("--cache", action="store_true", help="Bật cache theo level")
    p.add_argument("--out", type=str, default="outputs/result_mosaic.jpg", help="Đường dẫn ảnh output")
    p.add_argument("--show", action="store_true", help="Hiển thị ảnh sau khi tạo xong")
    return p.parse_args()


def main():
    args = parse_args()

    gen = MosaicGenerator(
        target_path=args.target,
        tiles_folder=args.tiles,
        tile_size=args.tile,
        blend_factor=args.blend,
    )

    img = gen.run(console_progress)
    cv2.imwrite(args.out, img)
    print("Saved:", args.out)

    if args.show:
        cv2.imshow("Photomosaic", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
