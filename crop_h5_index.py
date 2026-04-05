import argparse
import h5py
import numpy as np
from pathlib import Path


def crop_h5_index(h5_path, dataset_key, index, x0, y0, x1, y1, output_h5_path):
    """
    Extract a single index from an H5 dataset, crop it, and save to a new H5 file.

    Parameters
    ----------
    h5_path : str
        Path to the input H5 file.
    dataset_key : str
        Dataset key within the H5 file.
    index : int
        Index along the first dimension of the dataset to extract.
    x0, y0 : int
        Top-left corner of the crop region (column, row).
    x1, y1 : int
        Bottom-right corner of the crop region (column, row).
    output_h5_path : str
        Path for the output H5 file.
    """
    with h5py.File(h5_path, 'r') as f:
        if dataset_key not in f:
            raise KeyError(
                f"Dataset key '{dataset_key}' not found. Available keys: {list(f.keys())}"
            )
        dataset = f[dataset_key]
        if index >= len(dataset):
            raise IndexError(
                f"Index {index} out of range. Dataset has {len(dataset)} entries."
            )
        data = dataset[index][()]

    # Apply spatial crop based on array layout
    # Supported: (C, H, W), (H, W, C), (H, W)
    ndim = data.ndim
    if ndim == 3 and data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
        # (C, H, W)
        cropped = data[:, y0:y1, x0:x1]
    elif ndim == 3:
        # (H, W, C)
        cropped = data[y0:y1, x0:x1, :]
    elif ndim == 2:
        # (H, W)
        cropped = data[y0:y1, x0:x1]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    print(f"Input  shape : {data.shape}  dtype: {data.dtype}")
    print(f"Cropped shape: {cropped.shape}")

    output_path = Path(output_h5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset(dataset_key, data=cropped)

    print(f"Saved to: {output_path}  (key: '{dataset_key}')")


def main():
    parser = argparse.ArgumentParser(
        description="Extract one index from an H5 dataset, crop it, and save to a new H5 file."
    )
    parser.add_argument("h5_path",          type=str, help="Path to input H5 file")
    parser.add_argument("dataset_key",      type=str, help="Dataset key within the H5 file")
    parser.add_argument("index",            type=int, help="Index to extract from the dataset")
    parser.add_argument("x0",              type=int, help="Top-left crop x (column)")
    parser.add_argument("y0",              type=int, help="Top-left crop y (row)")
    parser.add_argument("x1",              type=int, help="Bottom-right crop x (column)")
    parser.add_argument("y1",              type=int, help="Bottom-right crop y (row)")
    parser.add_argument("output_h5_path",   type=str, help="Path for the output H5 file")

    args = parser.parse_args()

    crop_h5_index(
        args.h5_path,
        args.dataset_key,
        args.index,
        args.x0, args.y0,
        args.x1, args.y1,
        args.output_h5_path,
    )


if __name__ == "__main__":
    main()
