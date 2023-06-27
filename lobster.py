
from typing import Dict, List, Tuple
import json
import numpy as np
import math

### HELPERS 
def distance(pos_1, pos_2) -> float:
    return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

def min_dist(l1_pts: List[Tuple[float, float]], l2_pts: List[Tuple[float, float]]):
    dists = [
        distance(l1_point, l2_point) for (l1_point, l2_point) in product(l1_pts, l2_pts)
    ]
    return min(dists)

def coco_bbox_center_coords(bbox: List[float]):
    x1 = bbox[0]
    y1 = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return x1 + width / 2, y1 + height / 2


def min_dist_to_wireframe(
    l1_pts: List[Tuple[float, float]], l2_pts: List[Tuple[float, float]]
):
    dists = []
    for point in l1_pts:
        for i in range(len(l2_pts) - 1):
            lem = np.array(l2_pts[i + 1]) - np.array(l2_pts[i])
            if lem[0] == lem[1] == 0:
                continue
            point_to_lem = np.array(point) - np.array(l2_pts[i])

            r = np.dot(lem, point_to_lem)
            r /= np.dot(lem, lem)

            if r < 0:
                dists.append(np.sqrt(np.dot(point_to_lem, point_to_lem)))
            elif r > 1:
                point_to_lem2 = np.array(l2_pts[i + 1]) - np.array(point)
                dists.append(np.sqrt(np.dot(point_to_lem2, point_to_lem2)))
            else:
                dists.append(
                    np.sqrt(
                        np.dot(point_to_lem, point_to_lem) - r**2 * np.dot(lem, lem)
                    )
                )
    return min(dists)


class Lobster:
    def __init__(self, id: int, num_frames: int):
        self.id = id
        self.keypoints: List[List[float]] = []
        self.boundig_boxes: List[List[float]] = []
        self.interactions: List[Dict[int, float]] = [{} for _ in range(num_frames)]
        self.acc_aggression: List[float] = []
        self.dot_timestamps: List[int] = []
        self.is_interacting: bool = False

    def new_frame(self, bbox:List[List[float]] ,keypoints: List[List[float]] = None) -> None:
        self.keypoints.append(keypoints)
        self.boundig_boxes.append(bbox)

    def get_latest_keypoints(self) -> List[List[float]]:
        return self.keypoints[-1]
    
    def get_keypoints_of_frame(self, frame_index: int) -> List[List[float]]:
        return self.keypoints[frame_index]
    
    def get_latest_bounding_box(self) -> List[List[float]]:
        return self.boundig_boxes[-1]

    def get_bb_of_frame(self, frame_index: int) -> List[List[float]]:
        return self.boundig_boxes[frame_index]
    

    @staticmethod
    def from_json(file_name: str):
        lobsters: List[Lobster] = []
        with open(file_name, "r") as fil:
            dict_list = json.load(fil)
            for dict in dict_list:
                lobster = Lobster(dict["id"])
                lobster.bbs = dict["bbs"]
                lobster.keypoints = dict["keypoints"]
                # lobster.interactions = dict["interactions"]
                lobster.interactions = [
                    {int(k): v for (k, v) in ints.items()}
                    for ints in dict["interactions"]
                ]
                lobsters.append(lobster)
        return lobsters

    @staticmethod
    def get_x_data_kp(l1: "Lobster", l2: "Lobster", idx: int, window_size: int):
        start_idx = idx + 1 - window_size
        stop_idx = idx + 1

        l1_kps_raw_list = l1.keypoints[start_idx:stop_idx]
        l2_kps_raw_list = l2.keypoints[start_idx:stop_idx]

        def kps_raw_to_coords(kps):
            new_kp = kps.copy()
            new_kp[12:15] = kps[-3:]
            new_kp[-3:] = kps[12:15]

            kp_coords: List[Tuple[float, float]] = []
            for i in range(2, len(new_kp), 3):
                x = new_kp[i - 2]
                y = new_kp[i - 1]
                # visibility = coco_keypoints[i]
                pt = (x, y)
                kp_coords.append(pt)
            return kp_coords

        dists_l1_claws_l2: List[float] = []
        dists_l2_claws_l1: List[float] = []
        for l1_kps_raw, l2_kps_raw in zip(l1_kps_raw_list, l2_kps_raw_list):
            l1_kps_coords = kps_raw_to_coords(l1_kps_raw)
            (
                l1_left_claw,
                l1_left_elbow,
                l1_left_eye,
                l1_tail,
                l1_right_eye,
                l1_right_elbow,
                l1_right_claw,
            ) = l1_kps_coords

            l2_kps_coords = kps_raw_to_coords(l2_kps_raw)
            (
                l2_left_claw,
                l2_left_elbow,
                l2_left_eye,
                l2_tail,
                l2_right_eye,
                l2_right_elbow,
                l2_right_claw,
            ) = l2_kps_coords

            dists_l1_claws_l2.append(
                min_dist_to_wireframe([l1_left_claw, l1_right_claw], l2_kps_coords)
            )
            # dists_l1_claws_l2.append(
            #     min_dist([l1_left_claw, l1_right_claw], l2_kps_coords)
            # )
            # dists_l2_claws_l1.append(
            #     min_dist([l2_left_claw, l2_right_claw], l1_kps_coords)
            # )

        sample = [
            *dists_l1_claws_l2,
            # *dists_l2_claws_l1,
        ]
        return sample
    
    @staticmethod
    def get_centers(bbs: List[List[float]]):
        centers = [coco_bbox_center_coords(bb) for bb in bbs]
        return centers

