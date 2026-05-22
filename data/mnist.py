import gzip
import os
import struct
import urllib.request

import numpy as np

_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    for fname in _FILES.values():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(_BASE_URL + fname, path)


def _parse_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return imgs.astype(np.float32) / 255.0


def _parse_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(8)  # magic + count
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist_data(data_dir: str = "data/mnist_files"):
    """Returns (train, test) where each is a list of (x, y) tuples.
    x: float32 array of shape (784,), normalised to [0, 1]
    y: int scalar label in [0, 9]
    """
    _download(data_dir)
    train_x = _parse_images(os.path.join(data_dir, _FILES["train_images"]))
    train_y = _parse_labels(os.path.join(data_dir, _FILES["train_labels"]))
    test_x  = _parse_images(os.path.join(data_dir, _FILES["test_images"]))
    test_y  = _parse_labels(os.path.join(data_dir, _FILES["test_labels"]))

    train = list(zip(train_x, train_y))
    test  = list(zip(test_x,  test_y))
    return train, test
