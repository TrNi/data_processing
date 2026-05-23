import argparse
import io
from pathlib import Path
from PIL import Image


SIZE_THRESHOLD = 1.5 * 1024 * 1024  # 1.5 MB
MAX_TARGET    = 2.0 * 1024 * 1024  # 2 MB


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=9)
    return buf.getvalue()


def compress_png(input_path: Path, output_path: Path) -> None:
    original_size = input_path.stat().st_size

    if original_size <= SIZE_THRESHOLD:
        print(f"  Skipped (< 1.5 MB): {input_path.name}  ({original_size/1024/1024:.2f} MB)")
        return

    target_size = min(original_size / 2, MAX_TARGET)

    img = Image.open(input_path)
    best_data = input_path.read_bytes()
    method = 'original'

    # Step 1: max zlib compression
    data = png_bytes(img)
    if len(data) < len(best_data):
        best_data, method = data, 'compress_level=9'
    if len(best_data) <= target_size:
        output_path.write_bytes(best_data)
        print(f"  [{method}]  {input_path.name}  "
              f"{original_size/1024/1024:.2f} MB -> {len(best_data)/1024/1024:.2f} MB")
        return

    # Step 2: strip alpha if fully opaque (RGBA -> RGB)
    if img.mode == 'RGBA':
        alpha = img.getchannel('A')
        if alpha.getextrema() == (255, 255):  # fully opaque
            rgb = img.convert('RGB')
            data = png_bytes(rgb)
            if len(data) < len(best_data):
                best_data, method = data, 'RGBA->RGB'
            if len(best_data) <= target_size:
                output_path.write_bytes(best_data)
                print(f"  [{method}]  {input_path.name}  "
                      f"{original_size/1024/1024:.2f} MB -> {len(best_data)/1024/1024:.2f} MB")
                return

    # Step 3: color quantization (progressively fewer colors)
    base = img.convert('RGB') if img.mode not in ('RGB', 'L') else img
    for n_colors in (256, 128, 64, 32):
        quantized = base.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        data = png_bytes(quantized)
        if len(data) < len(best_data):
            best_data, method = data, f'quantize({n_colors} colors)'
        if len(best_data) <= target_size:
            break

    output_path.write_bytes(best_data)
    hit = len(best_data) <= target_size
    status = 'OK' if hit else 'best effort'
    print(f"  [{method}] {status}  {input_path.name}  "
          f"{original_size/1024/1024:.2f} MB -> {len(best_data)/1024/1024:.2f} MB  "
          f"(target {target_size/1024/1024:.2f} MB)")


def process_folder(input_dir: str, output_dir: str | None) -> None:
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    png_files = sorted(dict.fromkeys(
        list(input_path.glob('*.png')) + list(input_path.glob('*.PNG'))
    ))

    if not png_files:
        print(f"No PNG files found in '{input_dir}'.")
        return

    print(f"Found {len(png_files)} PNG file(s). Threshold: 1.5 MB, "
          f"Target: min(size/2, 2 MB)\n")

    for f in png_files:
        dest = (out_path / f.name) if out_path else f
        compress_png(f, dest)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description='Compress PNG files >1.5 MB to min(size/2, 2 MB).'
    )
    parser.add_argument('input_dir', type=str,
                        help='Folder containing PNG files')
    parser.add_argument('output_dir', type=str, default=None,
                        help='Output folder (default: overwrite in-place)')
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
