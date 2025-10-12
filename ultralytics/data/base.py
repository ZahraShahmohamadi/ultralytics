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

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, ops
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.instance import Instances
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.
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
        self.bgr = self.hyp.get("bgr", 0.0) > 0.0

        self.cache = cache if augment else False
        self.mosaic = self.augment and hyp.mosaic > 0
        self.mosaic_border = [-imgsz // 2, -imgsz // 2]

        if classes:
            self.names = dict(enumerate(classes))
        elif hyp and "names" in hyp:
            self.names = hyp["names"]
        else:
            self.names = {i: f"class_{i}" for i in range(999)}
        if self.single_cls:
            self.names = {0: "item"}

        self.ims = [None] * self.ni
        self.im_hw0 = [None] * self.ni
        self.im_hw = [None] * self.ni
        
        self.transforms = self.build_transforms(hyp=hyp)

    def get_im_files(self, img_path):
        """Get image files from a directory or file."""
        try:
            f = []
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
        if im is None:
            if Path(f).suffix.lower() == ".npy":
                im = np.load(f)
                if im.ndim == 2:
                    im = np.stack([im] * 3, axis=-1)
            else:
                im = cv2.imread(f)

            # This is the crucial fix.
            # Add these two lines to standardize the data type for ALL loaded images.
            if im is not None:
                im = im.astype(np.float32)

            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]
            r = self.imgsz / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)

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
        label = self.labels[index].copy()
        
        img, (h0, w0), (h, w) = self.load_image(index)

        # Check if the labels are OBB (8 points) or AABB (4 points)
        if label["bboxes"].shape[1] == 8:
            obb_polygons = label["bboxes"]
        else:
            # It's an AABB, convert it to an OBB polygon format
            # Assumes the AABB is in xywh format from the label file
            xyxy_from_xywh = ops.xywh2xyxy(label["bboxes"])
            obb_polygons = np.stack([
                xyxy_from_xywh[:, 0], xyxy_from_xywh[:, 1],  # top left
                xyxy_from_xywh[:, 2], xyxy_from_xywh[:, 1],  # top right
                xyxy_from_xywh[:, 2], xyxy_from_xywh[:, 3],  # bottom right
                xyxy_from_xywh[:, 0], xyxy_from_xywh[:, 3],  # bottom left
            ], axis=1)

        # Manually calculate the enclosing xyxy bboxes from the (now guaranteed) 8-point polygons
        x_coords = obb_polygons[:, 0::2]
        y_coords = obb_polygons[:, 1::2]
        xyxy_bboxes = np.stack([x_coords.min(axis=1), y_coords.min(axis=1), x_coords.max(axis=1), y_coords.max(axis=1)], axis=1)

        # Initialize Instances. This will no longer fail.
        instances = Instances(bboxes=xyxy_bboxes, segments=obb_polygons.reshape(-1, 4, 2))
        
        label_for_transform = {
            "img": img,
            "ori_shape": (h0, w0),
            "resized_shape": (h, w),
            "instances": instances,
            "cls": label["cls"],
            "im_file": self.im_files[index],
        }
        
        if self.transforms:
            label_for_transform = self.transforms(label_for_transform)

        return label_for_transform

    def build_transforms(self, hyp=None):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError
