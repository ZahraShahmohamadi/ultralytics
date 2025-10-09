# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.
    This version is modified to be compatible with modern Ultralytics architecture while robustly handling .npy files.
    """

    def __init__(
        self,
        img_path: str,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_im_files(self.img_path)
        self.labels = []
        self.ni = len(self.im_files)
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.mosaic = self.augment and hyp.mosaic > 0
        self.mosaic_border = [-imgsz // 2, -imgsz // 2]
        self.use_segments = hyp.get("segments", False)
        self.use_keypoints = hyp.get("keypoints", False)
        self.mode = "train" if augment else "val"
        self.use_labels = True

        # Cache attribute is essential for compatibility with augmentations like Mosaic
        self.cache = cache

        # Read classes
        if classes:
            self.names = dict(enumerate(classes))
        elif hyp and "names" in hyp:
            self.names = hyp["names"]
        else:
            self.names = {i: f"class_{i}" for i in range(999)}
        if self.single_cls:
            self.names = {0: "item"}

    def get_im_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # file
                    f.append(str(p))
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
            return im_files
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e

    def load_image(self, i) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Loads 1 image from dataset index 'i', returns (im, original hw, resized hw).
        This version is specifically designed to handle .npy files correctly.
        """
        f = self.im_files[i]
        
        # Load image
        if Path(f).suffix == ".npy":
            im = np.load(f)
            if im.ndim == 2:  # Convert grayscale to 3-channel
                im = np.stack([im] * 3, axis=-1)
        else:
            im = cv2.imread(f)  # BGR for standard formats
        
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")

        h0, w0 = im.shape[:2]  # orig hw
        r = self.imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
        
        return im, (h0, w0), im.shape[:2]


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.im_files)

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        label = self.labels[index].copy()
        
        # Load image
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        
        # Apply transforms
        if self.transforms:
            label = self.transforms(label)

        return label

    def build_transforms(self, hyp=None):
        """Users can customize augmentations here."""
        raise NotImplementedError

    def get_labels(self):
        """Users can customize their own format here."""
        raise NotImplementedError
