
from typing import List, Tuple
import numpy as np
import cv2
from lobster import Lobster

def add_trace_to_img_in_place(image, trace: list, color):
    frame_count = len(trace)
    for i, (x, y) in enumerate(trace):
        if i < frame_count / 3:
            size = 1
            thickness = -1
        elif i < 2 * frame_count / 3:
            size = 2
            thickness = -1
        else:
            size = 1
            thickness = 2

        pt = (int(x), int(y))
        cv2.circle(image, pt, size, color, thickness)

def draw_lob_info(
        frame,
        lobster: Lobster,
        frame_index: int,
        show_bb: bool = True,
        show_id: bool = True,
        id_str: str = None,
        show_trace: bool = True,
        show_kp: bool = False,
):  
    if show_kp:
        keypoints = lobster.get_keypoints_of_frame(frame_index)
    if id_str is None:     
        id_str = lobster.id
    bbox = lobster.get_bb_of_frame(frame_index)
    if show_id:
        cv2.putText(
            frame[0],
            str(id_str),
            (int(bbox[0]), int(bbox[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (250, 190, 0),
            1,
            cv2.LINE_AA,
        )
    if show_trace:
        ## Need to reverse - tracks from wrong index
        trace = Lobster.get_centers(lobster.boundig_boxes[max(frame_index - 75, 0) : frame_index + 1]) 
        if not np.any(np.isnan(trace)):
            add_trace_to_img_in_place(
                frame[0],
                trace,
                (170, 170, 170),
            )
    if show_bb:
        cv2.rectangle(
            frame[0],
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (255, 255, 255),
            1,
        )
    colors = [(0, 0, 255), 
            (0, 255, 0), 
            (255, 0, 0), 
            (255, 255, 0), 
            (255, 0, 255), 
            (0, 255, 255), 
            (255, 255, 255),
        ]
    if show_kp:
        for keypoint_index, koordinates in enumerate(keypoints):
                x, y, _ = koordinates
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(
                        frame[0], 
                        (int(x), int(y)), 
                        3, 
                        colors[keypoint_index],
                        -1)
    return frame