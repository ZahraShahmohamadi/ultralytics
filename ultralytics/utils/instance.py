# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List, Union

import numpy as np
import torch

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """Create a function that converts input to n-tuple by repeating singleton values."""

    def parse(x):
        """Parse input to return n-tuple by repeating singleton values n times."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes", "Instances")  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes in multiple formats.
    """

    def __init__(self, bboxes, format="xywh") -> None:
        """Initialize the Bboxes object with bounding box data and format."""
        assert format in ("xyxy", "xywh", "xywhn", "xyxyn", "xywhr"), f"unsupported format {format}"
        
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.numpy()
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32)

        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
            
        min_vals = 5 if format == "xywhr" else 4

        if bboxes.shape[0] > 0:
            assert bboxes.shape[1] >= min_vals, f"bboxes shape should be at least (N, {min_vals}), but got {bboxes.shape}"

        self.bboxes = bboxes.astype(np.float32)
        self.format = format

    
    def convert(self, format: str) -> None:
        """Convert bounding box format from one type to another."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes."""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]
        )

    def mul(self, scale: Union[int, tuple, list]) -> None:
        """Multiply bounding box coordinates by scale factor(s)."""
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: Union[int, tuple, list]) -> None:
        """Add offset to bounding box coordinates."""
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self) -> int:
        """Return the number of bounding boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis: int = 0) -> "Bboxes":
        """Concatenate a list of Bboxes objects into a single Bboxes object."""
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index: Union[int, np.ndarray, slice]) -> "Bboxes":
        """Retrieve a specific bounding box or a set of bounding boxes using indexing."""
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b)


class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.
    """

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: Union[np.ndarray, list] = None,
        keypoints: np.ndarray = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        """Initialize the Instances object."""
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments if segments is not None else []

    def convert_bbox(self, format: str) -> None:
        """Convert bounding box format."""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        """Scale coordinates by given factors."""
        self.bboxes[:, 0::2] *= scale_w
        self.bboxes[:, 1::2] *= scale_h
        if bbox_only:
            return
        
        if self.segments is not None and len(self.segments) > 0:
            if isinstance(self.segments, np.ndarray):
                self.segments[..., 0] *= scale_w
                self.segments[..., 1] *= scale_h
            else:
                for i in range(len(self.segments)):
                    self.segments[i][..., 0] *= scale_w
                    self.segments[i][..., 1] *= scale_h

        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h
    
    def denormalize(self, w: int, h: int) -> None:
        """Convert normalized coordinates to absolute coordinates."""
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))

        if self.segments is not None and len(self.segments) > 0:
            if isinstance(self.segments, np.ndarray):
                self.segments[..., 0] *= w
                self.segments[..., 1] *= h
            else:
                for i in range(len(self.segments)):
                    self.segments[i][..., 0] *= w
                    self.segments[i][..., 1] *= h

        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int) -> None:
        """Convert absolute coordinates to normalized coordinates."""
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        
        if self.segments is not None and len(self.segments) > 0:
            if isinstance(self.segments, np.ndarray):
                self.segments[..., 0] /= w
                self.segments[..., 1] /= h
            else:
                for i in range(len(self.segments)):
                    self.segments[i][..., 0] /= w
                    self.segments[i][..., 1] /= h

        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw: int, padh: int) -> None:
        """Add padding to coordinates."""
        assert not self.normalized, "you should add padding with absolute coordinates."
        self.bboxes[:, 0::2] += padw
        self.bboxes[:, 1::2] += padh
        
        if self.segments is not None and len(self.segments) > 0:
            if isinstance(self.segments, np.ndarray):
                self.segments[..., 0] += padw
                self.segments[..., 1] += padh
            else:
                for i in range(len(self.segments)):
                    self.segments[i][..., 0] += padw
                    self.segments[i][..., 1] += padh

        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh
    
    def __getitem__(self, index: Union[int, np.ndarray, slice]) -> "Instances":
        """Retrieve a specific instance or a set of instances using indexing."""
        segments = self.segments[index] if self.segments is not None and len(self.segments) > 0 else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h: int) -> None:
        """Flip coordinates vertically."""
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        
        if self.segments is not None and len(self.segments) > 0:
            if isinstance(self.segments, np.ndarray):
                self.segments[..., 1] = h - self.segments[..., 1]
            else:
                for i in range(len(self.segments)):
                    self.segments[i][..., 1] = h - self.segments[i][..., 1]

        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w: int) -> None:
        """Flip coordinates horizontally."""
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]

        if self.segments is not None and len(self.segments) > 0:
            if isinstance(self.segments, np.ndarray):
                self.segments[..., 0] = w - self.segments[..., 0]
            else:
                for i in range(len(self.segments)):
                    self.segments[i][..., 0] = w - self.segments[i][..., 0]
        
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w: int, h: int) -> None:
        """Clip coordinates to stay within image boundaries."""
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        
        if self.segments is not None and len(self.segments) > 0:
            if isinstance(self.segments, np.ndarray):
                self.segments[..., 0] = self.segments[..., 0].clip(0, w)
                self.segments[..., 1] = self.segments[..., 1].clip(0, h)
            else:
                for i in range(len(self.segments)):
                    self.segments[i][..., 0] = self.segments[i][..., 0].clip(0, w)
                    self.segments[i][..., 1] = self.segments[i][..., 1].clip(0, h)

        if self.keypoints is not None:
            self.keypoints[..., 2][
                (self.keypoints[..., 0] < 0)
                | (self.keypoints[..., 0] > w)
                | (self.keypoints[..., 1] < 0)
                | (self.keypoints[..., 1] > h)
            ] = 0.0
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)
            
    def remove_zero_area_boxes(self) -> np.ndarray:
        """Remove zero-area boxes."""
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if self.segments is not None and len(self.segments):
                # This needs to handle both list and numpy array
                if isinstance(self.segments, np.ndarray):
                    self.segments = self.segments[good]
                else:
                    self.segments = [s for i, s in enumerate(self.segments) if good[i]]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes: np.ndarray, segments: np.ndarray = None, keypoints: np.ndarray = None):
        """Update instance variables."""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self) -> int:
        """Return the number of instances."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        """Concatenate a list of Instances objects."""
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized
        
        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)

        # Handle concatenation for both list and numpy array segments
        is_list = any(isinstance(ins.segments, list) for ins in instances_list)
        if is_list:
             cat_segments = sum([ins.segments for ins in instances_list], [])
        else:
            seg_len = [b.segments.shape[1] for b in instances_list if b.segments is not None and len(b.segments) > 0]
            if len(frozenset(seg_len)) > 1:
                max_len = max(seg_len)
                cat_segments = np.concatenate(
                    [
                        resample_segments(list(b.segments), max_len)
                        if b.segments is not None and len(b.segments) > 0
                        else np.zeros((0, max_len, 2), dtype=np.float32)
                        for b in instances_list
                    ],
                    axis=axis,
                )
            else:
                cat_segments = np.concatenate([b.segments for b in instances_list if b.segments is not None and len(b.segments) > 0], axis=axis)

        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self) -> np.ndarray:
        """Return bounding boxes."""
        return self._bboxes.bboxes
