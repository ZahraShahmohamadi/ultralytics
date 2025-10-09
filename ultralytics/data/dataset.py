# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
import numpy as np
import cv2

from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import (
    DATASET_CACHE_VERSION,
    HELP_URL,
    load_dataset_cache_file,
    save_dataset_cache_file,
    img2label_paths,
    get_hash,
)
from ultralytics.data.augment import v8_transforms
from ultralytics.utils import LOCAL_RANK, LOGGER, TQDM


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection labels in YOLO format.
    This version contains a robust 'get_labels' method for .npy files and background images.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with a specific task."""
        self.data = data
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        super().__init__(*args, **kwargs)

        if self.use_labels:
            self.labels = self.get_labels()

    def get_labels(self):
        """
        A new, robust method to get labels that correctly handles .npy files, caching, and background images.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")

        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError, IndexError):
            LOGGER.info(f"{self.prefix}Scanning {cache_path.parent / cache_path.stem}...")
            x = {"labels": []}
            pbar = TQDM(self.im_files, desc=f"Scanning images from {self.img_path}")

            for i, im_file in enumerate(pbar):
                try:
                    if Path(im_file).suffix.lower() == ".npy":
                        im = np.load(im_file)
                    else:
                        im = cv2.imread(im_file)

                    if im is None:
                        raise ValueError(f"Unable to read image {im_file}")
                    shape = im.shape[:2]  # height, width

                    label_file = self.label_files[i]
                    if Path(label_file).is_file():
                        with open(label_file) as f:
                            l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                            l = np.array(l, dtype=np.float32)
                    else:
                        l = np.zeros((0, 5), dtype=np.float32)  # For background images

                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=l[:, 0:1] if len(l) else np.zeros((0, 1)),
                            bboxes=l[:, 1:] if len(l) else np.zeros((0, 4)),
                            segments=[],
                            keypoints=None,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                except Exception as e:
                    LOGGER.warning(f"WARNING ⚠️ Ignoring corrupt image/label: {im_file}: {e}")

            x["hash"] = get_hash(self.label_files + self.im_files)
            x["version"] = DATASET_CACHE_VERSION
            if save_dataset_cache_file(self.prefix, cache_path, x):
                LOGGER.info(f"{self.prefix}New cache created at {cache_path}")
            cache = x

        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No valid labels found in {cache_path}. See {HELP_URL}")

        self.im_files = [lb["im_file"] for lb in labels]
        return labels

    def build_transforms(self, hyp=None):
        """Builds and returns data augmentation transforms for the dataset."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            from ultralytics.data.augment import LetterBox
            transforms = v8_transforms(self, self.imgsz, hyp)
        
        from ultralytics.data.augment import Format
        transforms.transforms.append(
            Format(bbox_format="xywh", normalize=True)
        )
        return transforms
