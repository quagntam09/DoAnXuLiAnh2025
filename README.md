# KDTree Photo Mosaic

Ứng dụng desktop tạo ảnh mosaic từ ảnh gốc bằng cách ghép các tile ảnh nhỏ với thuật toán tìm láng giềng gần nhất (KD-Tree) và chia mức độ phân giải thích ứng (multi-resolution).

## Tính năng hiện có

- Giao diện desktop bằng `Tkinter`, thao tác trực tiếp không cần CLI phức tạp.
- Chọn ảnh gốc và thư mục tile ảnh qua hộp thoại hệ thống.
- Thuật toán adaptive multi-resolution: vùng chi tiết cao sẽ tiếp tục chia nhỏ.
- Dùng KD-Tree để tăng tốc tìm tile gần nhất theo đặc trưng màu trung bình.
- Hiển thị tiến độ theo phần trăm và preview frame trung gian trong quá trình render.
- Hỗ trợ lưu kết quả ra `.jpg` hoặc `.png`.
- Có cache tile/features trong `.cache/mosaic_tiles` để chạy lại nhanh hơn.

## Kiến trúc code hiện tại

Luồng chạy chính:

`main.py` -> `src.app.main.run()` -> `src.ui.main_window.App`

Các module chính:

```text
src/
├── app/
│   └── main.py                     # Bootstrap app
├── ui/
│   └── main_window.py              # Tkinter UI + preview + save
├── core/
│   ├── services/
│   │   └── mosaic_service.py       # Service orchestration + input validation
│   └── algorithms/
│       ├── multiresolution.py      # Adaptive mosaic pipeline + tile cache
│       ├── kdtree_module.py        # KDTree implementation
│       └── kdtree_nn.py            # Wrapper nearest-neighbor query
├── infra/
│   └── image_io.py                 # read/write/list image files
└── common/
    ├── config.py                   # Default parameters
    └── exceptions.py               # App-specific exceptions
```

Ngoài ra vẫn có các thư mục `algorithms/` và `ui/` ở root để giữ tương thích import cũ, nhưng implementation chính nằm trong `src/`.

## Cài đặt

Yêu cầu:

- Python 3.10+
- Pip

Dependencies hiện tại (`requirements.txt`):

- `numpy`
- `opencv-python`
- `Pillow`

Cài nhanh:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
python3 main.py
```

## Cách sử dụng

1. Bấm **Mở ảnh gốc...** để chọn ảnh chủ đề.
2. Bấm **Chọn thư mục ảnh nhỏ...** để chọn dataset tile.
3. Chỉnh các tham số:
   - `Kích thước ô nhỏ` (`tile_size`)
   - `Độ phân giải` (`levels`)
   - `Pha trộn ảnh gốc` (`blend_factor`)
4. Bấm **BẮT ĐẦU TẠO TRANH**.
5. Sau khi render xong, bấm **Lưu Kết Quả Về Máy**.

## Tham số mặc định (theo code)

Trong `src/common/config.py`:

- `DEFAULT_TILE_SIZE = 15`
- `DEFAULT_LEVELS = 3`
- `DEFAULT_BLEND = 0.2`
- `DEFAULT_FRAME_EVERY = 150`
- `DEFAULT_SPLIT_THRESHOLD = 20.0`

## Thuật toán (trạng thái hiện tại)

- Mỗi tile được resize về kích thước level tương ứng.
- Đặc trưng tile đang dùng: **mean color BGR 3 chiều**.
- Dựng KD-Tree trên tập vector đặc trưng tile ở từng level.
- Với mỗi block ảnh gốc:
  - Tính `mean/stddev` để quyết định có tách block tiếp không.
  - Nếu không tách, dùng `mean` của block để query KD-Tree và chọn tile gần nhất.
- Cuối pipeline có thể blend kết quả mosaic với ảnh gốc theo `blend_factor`.

## Cache

- Thư mục cache: `.cache/mosaic_tiles`
- Tên file cache gồm:
  - phiên bản cache (`CACHE_VERSION`)
  - fingerprint của dataset tile (dựa trên path/size/mtime)
  - tile size
- Nếu cache hợp lệ, app sẽ tái sử dụng tiles/features thay vì tính lại.

## Xử lý lỗi

Code dùng exception riêng trong `src/common/exceptions.py`:

- `InvalidInputError`
- `ImageLoadError`
- `TileDatasetError`

UI sẽ hiển thị cảnh báo/lỗi tương ứng bằng `messagebox`.

## Gợi ý dataset tile

- Nên dùng nhiều ảnh tile đa dạng màu sắc để kết quả phong phú hơn.
- Kích thước tile đầu vào không cần đồng nhất; app sẽ tự resize.
- Dataset càng lớn thì chất lượng khớp màu thường tốt hơn, nhưng thời gian build/cache cũng tăng.

