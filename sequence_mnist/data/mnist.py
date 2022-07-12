import codecs
import os
import sys
from urllib.error import URLError

import numpy as np

from sequence_mnist.data.file_utils import check_integrity, download_and_extract_archive

SN3_PASCALVINCENT_TYPEMAP = {
    8: np.uint8,
    9: np.int8,
    11: np.int16,
    12: np.int32,
    13: np.float32,
    14: np.float64,
}


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def read_sn3_pascalvincent(path: str, strict: bool = True) -> np.ndarray:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    dtype = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = np.iinfo(dtype).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with torch.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = np.frombuffer(bytearray(data), dtype=dtype, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.reshape(*s)


def read_image_file(path: str, transform: bool = True) -> np.ndarray:
    x = read_sn3_pascalvincent(path, strict=False)
    assert x.dtype == np.uint8
    assert x.ndim == 3
    return x


def read_label_file(path: str) -> np.ndarray:
    x = read_sn3_pascalvincent(path, strict=False)
    assert x.dtype == np.uint8
    assert x.ndim == 1
    return x


class MNIST:
    """
    DO NOT MODIFY
    Copies the MNIST class from torchvision.datasets.MNIST
    """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    def __init__(
        self,
        *,
        train: bool,
        root: str = "/tmp/data",
        download: bool = False,
    ):
        self.train = train  # train or test set
        self.root = root

        if download:
            self.download()

        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def _check_exists(self) -> bool:
        return all(
            check_integrity(
                os.path.join(
                    self.raw_folder, os.path.splitext(os.path.basename(url))[0]
                )
            )
            for url, _ in self.resources
        )

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def download(self):
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")
