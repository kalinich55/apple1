"""Pure geometry helpers for segmentation detections."""

import numpy as np


def estimate_center(mask_center, bbox, confidence):
    """Return a robust object center from a mask centroid and bounding box.

    A mask centroid far from its bounding-box center usually indicates an
    occluded or malformed mask.  In that case the box center is safer; for a
    plausible mask, blend both estimates according to confidence.
    """
    if mask_center is None:
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    x1, y1, x2, y2 = bbox
    bbox_cx = (x1 + x2) / 2.0
    bbox_cy = (y1 + y2) / 2.0
    mask_cx, mask_cy = mask_center
    shift = np.hypot(mask_cx - bbox_cx, mask_cy - bbox_cy)
    box_diag = np.hypot(x2 - x1, y2 - y1)

    if box_diag <= 1.0:
        return float(mask_cx), float(mask_cy)
    if shift > max(18.0, 0.35 * box_diag):
        return float(bbox_cx), float(bbox_cy)

    if confidence >= 0.9:
        mask_weight = 0.7
    elif confidence >= 0.8:
        mask_weight = 0.6
    else:
        mask_weight = 0.5

    bbox_weight = 1.0 - mask_weight
    return (
        float(mask_cx * mask_weight + bbox_cx * bbox_weight),
        float(mask_cy * mask_weight + bbox_cy * bbox_weight),
    )
