import numpy as np
from lobster import Lobster

def center_distance(bb_test, bb_gt):
    """
    Computes distance between bounding box centers
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    gt_midpoint_x = bb_gt[..., 0] + 0.5 * (bb_gt[..., 2] - bb_gt[..., 0])
    gt_midpoint_y = bb_gt[..., 1] + 0.5 * (bb_gt[..., 3] - bb_gt[..., 1])
    test_midpoint_x = bb_test[..., 0] + 0.5 * (bb_test[..., 2] - bb_test[..., 0])
    test_midpoint_y = bb_test[..., 1] + 0.5 * (bb_test[..., 3] - bb_test[..., 1])

    dist = np.sqrt(
        (test_midpoint_x - gt_midpoint_x) ** 2 + (test_midpoint_y - gt_midpoint_y) ** 2
    )
    return dist

def convert_bounding_boxes(bboxes):
    result = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        result.append(np.array([x1, y1, w, h], dtype=bbox.dtype))
    return result

def only_bbs(l1: Lobster, l2: Lobster, window_size: int) -> np.array:
    l1_bbs = convert_bounding_boxes(l1.boundig_boxes[-window_size:])
    l2_bbs = convert_bounding_boxes(l2.boundig_boxes[-window_size:])
    return np.concatenate((np.array(l1_bbs), np.array(l2_bbs))).reshape(-1)
    