# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

# All necessary imports are included
from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.
    This version is rewritten to be compatible with the modern Ultralytics architecture
    while robustly handling .npy files.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.0,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initializes a BaseDataset."""
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp
        self.prefix = prefix
        self.fraction = fraction
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.single_cls = single_cls
        self.classes = classes
        self.im_files = self.get_im_files(self.img_path)
        self.labels = []
        self.ni = len(self.im_files)
        # self.buffer = []  # buffer for mosaic images of the same class
        self.bgr = self.hyp.get("bgr", 0.0) > 0.0

        # Essential for compatibility with augmentations
        self.cache = cache if augment else False
        self.mosaic = self.augment and hyp.mosaic > 0
        self.mosaic_border = [-imgsz // 2, -imgsz // 2]

        # Read classes
        if classes:
            self.names = dict(enumerate(classes))
        elif hyp and "names" in hyp:
            self.names = hyp["names"]
        else:
            self.names = {i: f"class_{i}" for i in range(999)}
        if self.single_cls:
            self.names = {0: "item"}
        
        # Other attributes from the original base.py
        self.ims = [None] * self.ni
        self.im_hw0 = [None] * self.ni
        self.im_hw = [None] * self.ni

    def get_im_files(self, img_path):
        """Get image files from a directory or file."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f.extend(glob.glob(str(p / "**" / "*.*"), recursive=True))
                elif p.is_file():
                    f.append(str(p))
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
            return im_files
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e

    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)."""
        im, f = self.ims[i], self.im_files[i]
        if im is None:  # not cached
            if Path(f).suffix == ".npy":
                im = np.load(f)
                if im.ndim == 2:
                    im = np.stack([im] * 3, axis=-1)
            else:
                im = cv2.imread(f)  # BGR
            
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # HW
            r = self.imgsz / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
            # Cache image
            if self.cache:
                self.ims[i] = im
                self.im_hw0[i] = (h0, w0)
                self.im_hw[i] = im.shape[:2]

            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def __len__(self):
        """Returns the number of images in the dataset."""
        return self.ni

    def __getitem__(self, index):
        """Returns one data sample (image and labels)."""
        # This method will be overridden by YOLODataset, but we include a basic structure
        hyp = self.hyp
        
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        
        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.imgsz  # final letterboxed shape
        img, ratio, pad = self.letterbox(img, shape, auto=False, scaleup=self.augment)
        
        # Create a label dictionary
        label = {
            "img": img,
            "ori_shape": (h0, w0),
        }
        
        # Apply transforms
        if self.transforms:
            label = self.transforms(label)

        return label

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    # The following methods are placeholders to be overridden by the child class (YOLODataset)
    def build_transforms(self, hyp=None):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError
