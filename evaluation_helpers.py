from TrackEval import trackeval

from ObjectDetector import YOLODetector
from keypointDetector import PoseEstimator
from pathlib import Path
from glob import glob
import pandas as pd
import os
from pathlib import Path
import numpy as np
import pandas as pd
import motmetrics as mm


import cv2
import numpy as np


def boxes_with_id(im, trackers, relative_color = True):
    '''
    Draws tracking boxes with ID-numbers.
    '''
    total_tracks = trackers.shape[0]
    for track in trackers:
        tl = (int(track[0]), int(track[1]))
        br = (int(track[2]),int(track[3]))
        box_id = int(track[4])
        color = (0,255*(1-box_id/total_tracks),255*(box_id/total_tracks)) if relative_color else (0,0,255)
        cv2.rectangle(im, tl, br, color, thickness = 3)
        cv2.putText(im, str(box_id), tl, 0, 1, (255,0,0), thickness = 3)


def path(im, points, id):
    '''
    Draws a path for a single individual.
    Args:
        im: Image on which to draw the path
        points (list[list[float]]): Coordinate sequence to draw. List of points [x, y]
        id (int): Id of the individual corresponding to the points.
    '''
    for coord in points:
        cv2.circle(im, tuple(coord), 2, color=(0,255*(1-id/21),255*(id/21)), thickness = -1)

def all_paths(im, coordinates, i):
    '''
    Draws all paths on an image.
    Args:
        im: Image on which to draw the paths
        coordinates (list[list[list[float]]]): The full coordinates for the whole sequence.
            Dim 0 contatins each individual path, dim 1 containes the time-steps [coord_0, ..., coord_n] and
            dim 2 contains [x_i, y_i]
        i (int): Current time-step
    '''
    nmbr_of_objects = len(coordinates)
    for j in range(nmbr_of_objects):
        path(im, coordinates[j,:1+i].astype(int), j+1)


def speed(im, point, speed, id):
    color =(0,0,255)#(0,255*(1-id/21),255*(id/21))
    cv2.arrowedLine(im, tuple(point), tuple(point + 20*speed),color=color, thickness = 2)

def all_speeds(im, coordinates, speeds, i):
    '''
    Draws all speeds on an image.
    Args:
        im: Image on which to draw the paths
        coordinates (list[list[list[float]]]): The full coordinates for the whole sequence.
            Dim 0 contatins each individual path, dim 1 containes the time-steps [coord_0, ..., coord_n] and
            dim 2 contains [x_i, y_i]
        speeds: the full speeds for the whole sequence
        i (int): Current time-step
    '''
    nmbr_of_objects = len(coordinates)
    for j in range(nmbr_of_objects):
        speed(im, coordinates[j][i].astype(int), speeds[j][i].astype(int), j+1)

def interactions(im, interactions):
    for box in interactions:
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]),int(box[3]))
        color = (255,0,0)
        cv2.rectangle(im, tl, br, color, thickness = 3)

def _background_box(im, tl = (0,0), br = (375, 700)):
    cv2.rectangle(im, tl, br,(0,0,0),thickness=-1)

def distances(im, cum_distances, i):
    """
    Display the current traveled distance for each individual.
    Args:
        im: Image on whcih to draw the paths
        cum_distances (np.ndarray): The cumulative distance for the whole sequence. Dimension (individ_numbers, max-frames)
                Dim 0 contatins each individual path from id n at index 0 to id 1 at max_idx,
                Dim 1 containes the time-steps [dist_0, ..., dist_n]
        i (int): Current time-step
    """
    _background_box(im)
    nmbrs_of_objects = len(cum_distances)
    cv2.putText(im, "Lobster ID: Distance",\
                (10,40), 0, 0.9, (0,0,255), thickness = 2)
    for x in range(nmbrs_of_objects):
        dist = cum_distances[x,i]
        cv2.putText(im, "Lobster " + str(x+1).zfill(2) +": " + str(int(dist)).zfill(6),\
                (10,40 + 30*(x+1)), 0, 0.9, (0,0,255), thickness = 2)


def num_interactions(im, interaction_dict, offset = 0):
    '''
    Args:
        im: image
        interaction_dict (dict): Dict of total interaction iou
    '''
    _background_box(im)
    cv2.putText(im, "Lobster " + "ID" +": " + "interactions",\
                (10,40 + offset), 0, 0.9, (0,0,255), thickness = 2)
    for lob_id, val in interaction_dict.items():
        cv2.putText(im, "Lobster " + str(lob_id).zfill(2) +": " + str(int(val)).zfill(6),\
                (10,40 + offset+ 30*(lob_id)), 0, 0.9, (0,0,255), thickness = 2)

def interactions_and_distance(im, interaction_dict, cum_distances, i, color = (0,0,255)):
    '''
    For each lobster, display its total traversed distance, and total number of unique interactions
    Args:
        im: Image
        interaction_dict (dict): Dictionary with {lobster_id : number of interactions}
        cum_distances (np.ndarray): The cumulative distance for the whole sequence. Dimension (individ_numbers, max-frames)
                Dim 0 contatins each individual path from id n at index 0 to id 1 at max_idx,
                Dim 1 containes the time-steps [dist_0, ..., dist_n]
        i (int): current frame
    '''
    _background_box(im)
    cv2.putText(im, "ID: Distance, Interactions",\
                (10,40), 0, 0.9, color, thickness = 2)
    for lob_id, val in interaction_dict.items():
        dist = cum_distances[lob_id-1,i]
        cv2.putText(im, str(lob_id).zfill(2) +": " + str(int(dist)).zfill(6)\
         + ", " + str(int(val)).zfill(6),(10,40 + 30*(lob_id)), 0, 0.9, color, thickness = 2)


def what_interactions(im, interactions, color = (0,0,255)):
    '''
    Displays what interactions are currently going on in the frame.
    Displayed on the form "ID1 | ID2 | Time since start"
    '''
    _background_box(im)
    cv2.putText(im, "ID1 | ID2 | time",\
                (10,40), 0, 0.9, color, thickness = 2)
    for i, interaction in enumerate(interactions):
        cv2.putText(im, str(interaction[-3]).zfill(2) + "  | " + str(interaction[-2]).zfill(2) + "  | " + str(interaction[-1]).zfill(4),
        (10,40 + 30*(i+1)), 0, 0.9, color, thickness=2)


def key_points(im, keypoints, metadata):
    '''
    Draws all keypoints in a frame.
    '''
    keypoint_names = metadata.get("keypoint_names")
    for individual_keypoints in keypoints:
        visible = {}
        for idx, keypoint in enumerate(individual_keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > 0.05:
                cv2.circle(img=im, center=(x, y), radius=4, color=(255,0,0), thickness=-1)
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)
        if metadata.get("keypoint_connection_rules"):
            for kp0, kp1, color in metadata.keypoint_connection_rules:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = tuple(x for x in color)
                    cv2.line(im, (x0, y0), (x1, y1), color=color, thickness=1)


def final_display(im, interaction_dict, cum_distances):
    '''
    Args:
        im (np.ndarray): image
        interaction_dict (dict): interactions
        cum_distances (np.ndarray): The cumulative distance for the whole sequence. Dimension (individ_numbers, max-frames)
                Dim 0 contatins each individual path from id n at index 0 to id 1 at max_idx,
                Dim 1 containes the time-steps [dist_0, ..., dist_n]
        color (tuple): color
    '''
    num_individs = cum_distances.shape[0]
    _background_box(im, tl = (1000,0), br=(1920, 60 + 30*num_individs))
    cv2.putText(im, "ID: Distance, Interactions",\
                (1050,40), 0, 0.9, (0,0,255), thickness = 2)

    

    #Calculate scores for each index
    distance_id    = [[cum_distances[i,-1], i+1] for i in range(num_individs)]
    interaction_id = [[interaction_dict[i+1], i+1] for i in range(num_individs)]
    
    distance_id.sort()
    interaction_id.sort()

    score_dict = {i+1:0 for i in range(num_individs)}
    interaction_score = 0
    max_interaction   = min(interaction_id[:][0])

    #The score is the sum of the each lobsters rank in distance and number of interactions
    #If two lobsters have equal number of interactions they get the same interaction score
    for i, (dist, inter) in enumerate(zip(distance_id, interaction_id)):
        if(inter[0] > max_interaction):
            max_interaction = inter[0]
            interaction_score += 1
        score_dict[dist[1]] += i
        score_dict[inter[1]] += interaction_score

    #Sort indexes on scores
    score_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1])}

    #Draw output
    for i,  lob_id in enumerate(score_dict.keys()):
        color = (0,0,255)
        if (i<3):
            color = (0,255,0)
        elif(i < num_individs - 3):
            color = (51, 255, 255)
        cv2.putText(im, str(int(lob_id)).zfill(2) +": " + str(int(cum_distances[lob_id-1,-1])).zfill(6)\
         + ", " + str(int(interaction_dict[lob_id])).zfill(6),(1050,40 + 30*(i+1)), 0, 0.9, color, thickness = 2)



def evaluate_trackers(metrics, ground_truth_folder, trackers_folder, split_to_evaluate, dataset_to_evaluate):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config["USE_PARALLEL"] = True
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = ground_truth_folder
    dataset_config["TRACKERS_FOLDER"] = trackers_folder
    dataset_config["SPLIT_TO_EVAL"] = split_to_evaluate
    dataset_config["BENCHMARK"] = dataset_to_evaluate

    # Run evaluation code
    evaluator = trackeval.Evaluator(eval_config)
    print(trackeval.datasets.MotChallenge2DBox(dataset_config))
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics['METRICS']:
            metrics_list.append(metric(metrics))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)


from time import strftime, time
from typing import List, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path
import cv2
from argparse import ArgumentParser, Namespace
import torch
from Sort.sort import Sort


class VideoAnnotator:
    # General class for using an object detector to
    # annotate a given lobster video.
    def __init__(
        self, model_type, tracker="SORT", weights=None
    ) -> None:
        self.model_type = model_type
        self.weights = weights
        self.tracker = tracker
        self.det_thresh = 0.7
        if tracker == "BoT-SORT":
            self.tracker_config = "botsort.yaml"
        elif tracker == "ByteTrack":
            self.tracker_config = "bytetrack.yaml"
            
        self.predictor = self._instantiate_predictor(self.model_type, self.weights)

    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        return images

    def annotate_video(
        self,
        input_path: str,
        num_lobsters: Optional[int] = None,
        write_video: bool = False,
        output_video: Optional[str] = None,
        write_tracker_output: bool = False,
        tracker_output_file: Optional[str] = None,
        keypoints_output_file: Optional[str] = None,
    ):
        """
        Annotates a video using the predictor, and optionally
        writes the resulting video and tracker outputs to file.
        """
        # Instantiate the frame provider
        print(f'Loading video from {input_path}...')
        # Load input video
        frames = self.load_images_from_folder(input_path)
        print(f'Loaded {len(frames)} frames from video.')

        # Infer how many lobsters are present in frame (based on 100 first frames in FrameProvider)
        if num_lobsters is None:
            number_of_objects = self._infer_num_objects(frames)
        else:
            number_of_objects = num_lobsters
        print(f"Found {number_of_objects} lobsters in the video. Continuing...")
        # Set up timing variables
        predict_time = 0
        detect_time = 0
        track_time = 0
        # initialized = False

        # Set up data structures for storing output
        tracker_list = []
        keypoints_list = []

        print(f"Annotating data from {input_path} using model {self.model_type}")
        print(f"Total frames: {len(frames)}")

        # Iterate over whole sequence
        inference_start_time = time()
        for frame in tqdm(
            frames,
            desc="Running tracking",
            total=len(frames),
            unit=" frames",
        ):
            # Assume YOLO tracker
            frame_inference_start = time()
            tracker_output = self.predictor.model.track(source=frame, conf=0.3, iou=0.5, tracker=self.tracker_config)
            a = np.array([obj.boxes.xyxy.cpu().numpy() for obj in tracker_output][0])
            b = np.array([obj.boxes.id.cpu().numpy() for obj in tracker_output][0])
            b = b.reshape(-1 , 1)
            tracker_output = np.concatenate((a, b), axis=1)
            tracker_list.append(tracker_output)
            frame_inference_end = time()
            tracking_start_time = time()
            tracking_end_time = time()
            predict_time += tracking_end_time - frame_inference_start
            track_time += tracking_end_time - tracking_start_time
            detect_time += frame_inference_end - frame_inference_start

        inference_time = time() - inference_start_time
        print(f"Inference complete for {input_path}")
        print(f"Predict time: {predict_time:.3f} s")
        print(f"Total detect time: {detect_time:.3f} s")
        print(f"Total track and data production time: {track_time:.3f} s")
        print(f"Total inference time: {inference_time:.3f} s")
        avg_fps = len(frames) / inference_time
        print(f"Average fps: {avg_fps:.3f}")
        if write_video:
            if output_video is None:
                vname = strftime("data/annotated_video_%Y-%m-%d_%H%M%S.mp4")
                print(f"WARNING: output_video not set, writing video to file: {vname}")
            else:
                vname = output_video
            # self.make_video(
            #     vname,
            #     frames,
            #     tracker_list,
            #     keypoints_list if keypoints_list else None,
            # )
            # Write tracker output to file if desired
        if write_tracker_output:
            if tracker_output_file is None:
                tname = strftime("data/annotation_output_%Y-%m-%d_%H%M%S.csv")
                print(f"WARNING: tracker_output_file not set!")
                print("Writing tracker output to file: {tname}")
            else:
                tname = tracker_output_file
            dump_keypoints = self.tracker == "KeypointSort"
            if keypoints_output_file is None:
                kpname = strftime("data/keypoint_output_%Y-%m-%d_%H%M%S.csv")
                if dump_keypoints:
                    print(f"WARNING: keypoints_output_file not set!")
                    print("Writing keypoint output to file: {kpname}")
            else:
                kpname = keypoints_output_file
            self.dump_mot_detections(tname, tracker_list, dump_keypoints, kpname, 0)

        return tracker_list, avg_fps

    def _infer_num_objects(self, images):
        """Automatically infers how many lobsters are in frame, based on the first 100
        frames delivered by the frame provider."""
        num_objects = []
        model = YOLODetector("yolomodels/yolo8l/weights/best.pt")
        instances = 0
        for i, frame in tqdm(
            enumerate(images), desc="Inferring number of lobsters", total=100, unit="frames"
        ):
            if i >= 25:
                break

            out = model.model.track(source=frame, conf=0.3, iou=0.5)
            instances = max(max([obj.boxes.id.cpu().numpy() for obj in out][0]), instances) 
        return int(instances)

    def make_video(
        self,
        output_path: str,
        images,
        track_list: List[np.ndarray],
        keypoints_list: List[np.ndarray] = None,
    ):
        """
        Adds bounding boxes and optionally keypoints to the frames,
        and writes them to a video file.
        """
        video_out = self._setup_video_output(output_path, images)
        video_making_start_time = time()

        for i, frame in tqdm(
            enumerate(images),
            desc="Producing videos",
            unit="frames",
            total=len(images),
        ):
            # Get the track corresponding to the current frame
            track = track_list[i]

            # Draw bounding boxes (and keypoints if they're provided),
            # and write the edited frame to file
            boxes_with_id(frame, track, relative_color=False)
            if keypoints_list is not None:
                key_points(
                    frame,
                    keypoints_list[i],
                    self.predictor.get_keypoint_metadata(),
                )
            video_out.write(frame)

        video_out.release()
        print(f"Finished making video {output_path}")
        print(f"Total time for making video: {time()-video_making_start_time:.3f} s")

    def dump_mot_detections(self, output_path, tracker_list, dump_keypoints, kp_output_path, frame_offset):
        """
        Dumps the tracker output on MOT format to file. NOTE: Does not write keypoints to file.
        """
        self._check_target_dir(output_path)
        with open(output_path, "w") as f:
            for i, track in enumerate(tracker_list):
                for instance in track:
                    f.write(
                        f"{i + frame_offset + 1},{int(instance[4])},{instance[0]},{instance[1]},{instance[2]-instance[0]},{instance[3]-instance[1]},-1,-1,-1,-1\n"
                    )
        if dump_keypoints:
            self._check_target_dir(kp_output_path)
            with open(kp_output_path, "w") as f:
                for i, (track) in enumerate(tracker_list):
                    for instance in track:
                        keypoints = instance[5:].reshape(-1, 3)[..., :2].reshape(-1)
                        s = f"{i + frame_offset + 1},{instance[4]},"
                        s += ",".join([f"{kp}" for kp in keypoints])
                        s += "\n"
                        f.write(s)

    def _setup_video_output(self, output_path, images):
        self._check_target_dir(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        sample_frame = images[0]
        video_shape = (sample_frame.shape[1], sample_frame.shape[0])
        video_out = cv2.VideoWriter(output_path, fourcc, 30.0, video_shape, True)

        return video_out

    def _check_target_dir(self, output_path):
        target_dir = Path(output_path).parent
        if not target_dir.exists():
            target_dir.mkdir(parents=True)

    def _instantiate_predictor(self, model_type, weights):
        if "Yolo" in model_type:
            return YOLODetector(weights)
        elif "SLEAP" in model_type:
            return PoseEstimator(weights)
        else:
            raise NotImplementedError(f"Model {model_type} is not implemented.")


def get_BYTE_args():
    args = Namespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False)
    return args


def create_metric_overview(dataset="LOBSTER"):
    tracker_results_dir = f"data/tracker_results/{dataset}-test"
    tracker_dirs = [Path(p) for p in glob(tracker_results_dir + "/*") if Path(p).is_dir()]
    tracker_names = [p.name for p in tracker_dirs]

    summary_df = pd.DataFrame(
        columns=[
            "HOTA___AUC", 
            "MOTA", 
            "MOTP", 
            "CLR_Re", 
            "CLR_Pr", 
            "FP_per_frame", 
            "CLR_FN", 
            "CLR_FP", 
            "IDSW", 
            "IDs",
            "GT_IDs", 
            "MT", 
            "PT", 
            "ML", 
            "Frag",
            "IDF1", 
            "IDP", 
            "IDR", 
            "FPS"
        ],
        index = pd.Index(tracker_names, name="tracker")
    )
    for tracker in tracker_dirs:
        tracker_df = pd.read_csv(tracker.joinpath("pedestrian_detailed.csv"))
        
        # Extract only relevant row and cols
        tracker_df = tracker_df.iloc[-1, :].loc[
            ["HOTA___AUC", "MOTA", "MOTP", "CLR_Re", "CLR_Pr", "FP_per_frame", 
             "CLR_FN", "CLR_FP", "IDSW", "IDs", "GT_IDs", "MT", "PT", "ML", "Frag",
             "IDF1", "IDP", "IDR"]
        ]

        tracker_df["FPS"] = pd.read_csv(tracker.joinpath("fps_summary.txt")).set_index("sequence").loc["average", "fps"]

        summary_df.loc[tracker.name] = tracker_df

    summary_df = summary_df.sort_index()
    summary_df.index = summary_df.index.rename("Model")
    summary_df = summary_df.rename(columns={
        "HOTA___AUC": "HOTA",
        "CLR_Re": "Recall",
        "CLR_Pr": "Precision",
        "FP_per_frame": "FP per frame",
        "CLR_FN": "FN",
        "CLR_FP": "FP",
        "IDSW": "IDsw",
        "GT_IDs": "GT IDs"
    })   
    summary_df.to_csv(str(Path(tracker_results_dir).joinpath("model_summary.csv")))
    dump_page_wide_latex_tables_to_file(summary_df, str(Path(tracker_results_dir).joinpath("model_summary.tex")))


def create_latex_table(df, cols=None):
    if cols is None:
        cols = list(df.columns)
    subset_df = df[cols]
    # Create the string used for formatting the columns in the final table
    column_format_str = "|l|" + "".join(["r" for _ in range(len(subset_df.columns)-1)]) + "|"
    
    # Cast the Frag column to int if it is present
    if "Frag" in cols:
        subset_df.loc[:, "Frag"] = subset_df["Frag"].astype("int")
    
    # Multiply the percentage attributes by 100 for better readability
    subset_df = subset_df.apply(lambda x: x * 100 if x.name in ["HOTA", "MOTA", "MOTP", "Recall", "Precision", "IDF1", "IDP", "IDR"] and max(x.values) <= 1 else x)
    
    styler = subset_df.style
    # Hide the index and format the df with precision 2
    styler = styler.hide_index().format(precision=2)
    
    # Escape special characters in the model column if present
    if "Model" in cols:
        styler = styler.format(subset="Model", escape="latex")
        
    # Highlight max values in max columns
    subset_cols = [col for col in cols if col in ["HOTA", "MOTA", "MOTP", "Recall", "Precision", "MT", "IDF1", "IDP", "IDR", "FPS"]]
    styler = styler.highlight_max(subset=subset_cols, props="textbf:--rwrap")
    
    # Highlight min values in min columns
    subset_cols = [col for col in cols if col in ["FP per frame", "FN", "FP", "IDsw", "PT", "ML", "Frag"]]
    styler = styler.highlight_min(subset=subset_cols, props="textbf:--rwrap")
    
    # Find the ID of the first IDs value that is closest to GT IDs 
    # (i.e. the model with the closest amount of tracks to the ground truth)
    if "IDs" in cols and "GT IDs" in cols:
        closest_ids = (subset_df["IDs"] - subset_df["GT IDs"]).abs().argsort()[0]
        styler = styler.highlight_between(subset=["IDs"], left=subset_df.at[closest_ids, "IDs"], right=subset_df.at[closest_ids, "IDs"], props="textbf:--rwrap")
        
    return styler.to_latex(hrules=True, column_format=column_format_str)


def dump_page_wide_latex_tables_to_file(df, outfile):
    reset_df = df.reset_index()
    if not Path(outfile).parent.exists():
        Path(outfile).parent.mkdir(parents=True)
    with open(outfile, "w") as f:
        f.write(create_latex_table(reset_df, ["Model", "HOTA", "MOTA", "MOTP", "Recall", "Precision", "FP per frame", "FN", "FP"]))
        f.write("\n")
        f.write(create_latex_table(reset_df, ["Model", "IDsw", "IDs", "GT IDs", "MT", "PT", "ML", "Frag", "IDF1", "IDP", "IDR", "FPS"]))
    print(f"Successfully wrote results table to {outfile}!")




def evaluate_kp_trackers(gt_dir, bb_gt_dir, preds_dir, criterion="PCK", thresh=0.1):
    gt_sequences = [gt_file for gt_file in os.listdir(gt_dir) if "sequence" in gt_file]
    gt_sequences = sorted(gt_sequences, key=lambda s: int(s.split("sequence")[-1]))
    trackers = [d for d in sorted(os.listdir(preds_dir)) if Path(f"{preds_dir}/{d}").is_dir()]

    groups = ["claws", "elbows", "eyes", "tail", "total"]
    mean_metrics = ['idf1', 'idp', 'idr', 'recall', 'precision', 'mota', 'motp']
    count_metrics = ['num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations']
    all_metric_names = mean_metrics+count_metrics

    metrics = [f"{m}_{g}" for m in mean_metrics+count_metrics for g in groups]
    n_joints = 7
    joint_names = ["l_claw", "l_elbow", "l_eye", "tail", "r_claw", "r_elbow", "r_eye"]
    head_idx = 2
    tail_idx = 3
    lob_kp_sigmas = [0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05]

    final_result_df = pd.DataFrame(
        columns=metrics,
        index=pd.Index(trackers, name="Model")
    )

    for tracker in trackers:
        tracker_result_df = pd.DataFrame(
            columns=metrics,
            index=pd.Index(gt_sequences, name="seq")
        )
        print(f">>> Evaluating {tracker}")
        for sequence in gt_sequences:
            # Calculate sequence-wise metrics and add to dataframe
            # Load gt and preds
            gt_file = f"{gt_dir}/{sequence}/gt.txt"
            gt = pd.read_csv(gt_file, header=None).values
            bb_gt_file = f"{bb_gt_dir}/{sequence}/gt/gt.txt"
            bb_gt = pd.read_csv(bb_gt_file, header=None).values
            pred_file = f"{preds_dir}/{tracker}/data/{sequence}.txt"
            preds = pd.read_csv(pred_file, header=None).values
            pred_frames = np.unique(preds[..., 0])
            gt_frames = np.unique(gt[..., 0])
            common_frames = np.intersect1d(gt_frames, pred_frames)
            joint_accs = []
            # Calculate MOTA and IDsw for each joint
            for joint in range(n_joints):
                acc = mm.MOTAccumulator(auto_id=True)
                joint_accs.append(acc)
                # Go through each frame, and accumulate all objects, hypotheses
                # and distances (bounded by lobster length times a threshold value)
                for frame in common_frames:
                    frame_gts = gt[np.where(gt[..., 0] == frame)]
                    frame_preds = preds[np.where(preds[..., 0] == frame)]
                    frame_bbs = bb_gt[np.where(bb_gt[..., 0] == frame), 2:6].reshape(-1, 4)
                    
                    # Get all hypotheses IDs and object IDs
                    gt_ids = frame_gts[..., 1]
                    pred_ids = frame_preds[..., 1]
                    
                    # Get all keypoints in the frame
                    gt_frame_kps = frame_gts[..., 2:].reshape(-1, 7, 2)
                    pred_frame_kps = frame_preds[..., 2:].reshape(-1, 7, 2)

                    # Extract relevant joints
                    gt_joints = gt_frame_kps[:, joint]
                    pred_joints = pred_frame_kps[:, joint]                 
                                        
                    # Calculate keypoint distance matrix from all objects to all hypotheses
                    dist_matrix = np.sqrt(mm.distances.norm2squared_matrix(gt_joints, pred_joints))
                    
                    dist = np.full(dist_matrix.shape, np.nan)
                    if criterion == "PCK":
                        # Find all lobster lengths
                        gt_lob_lens = np.sqrt(np.sum((gt_frame_kps[:, head_idx] - gt_frame_kps[:, tail_idx])**2, axis=1))

                        # Discard all keypoints that are farther away than thresh times gt lobster length
                        for gt_idx in range(len(gt_ids)):
                            dist[gt_idx] = np.where(dist_matrix[gt_idx] < thresh * gt_lob_lens[gt_idx], dist_matrix[gt_idx], dist[gt_idx])
                
                    elif criterion == "KS":
                        # Find all lobster areas
                        gt_lob_scales = frame_bbs[..., 2] * frame_bbs[..., 3]
                        
                        # Find keypoint similarity matrix
                        ks_matrix = np.full(dist_matrix.shape, np.nan)
                        for gt_idx in range(len(gt_ids)):
                            ks_matrix[gt_idx] = np.exp(-dist_matrix[gt_idx]**2 / ((2 * lob_kp_sigmas[joint])**2) / gt_lob_scales[gt_idx] / 2.)
                            dist[gt_idx] = np.where(ks_matrix[gt_idx] > thresh, ks_matrix[gt_idx], dist[gt_idx])
                        dist = 1. - dist
                    else: raise ValueError(f"Unsupported MOTA criterion: {criterion}")                     

                    # Update the MOT accumulator
                    acc.update(
                        oids=gt_ids,
                        hids=pred_ids,
                        dists=dist
                    )
                    
            # Calculate MOT metrics for the sequence
            mh = mm.metrics.create()
            summary = mh.compute_many(joint_accs, metrics=all_metric_names, names=joint_names)
            for group in groups:
                if group == "total":
                    summary.loc[group, :] = [*summary.loc[joint_names, mean_metrics].mean(axis=0), *summary.loc[joint_names, count_metrics].sum(axis=0)]
                elif group == "tail":
                    continue
                else:
                    summary.loc[group, :] = [*summary.loc[[f"l_{group[:-1]}", f"r_{group[:-1]}"], mean_metrics].mean(axis=0), *summary.loc[[f"l_{group[:-1]}", f"r_{group[:-1]}"], count_metrics].sum(axis=0)]
            sequence_results = []
            for m in summary.columns:
                sequence_results += summary.loc[groups, m].tolist()
            tracker_result_df.loc[sequence, :] = sequence_results        

        # Calculate "total" row, and dump df to file
        tracker_result_df.loc["total", :] = [*tracker_result_df.loc[:, [f"{m}_{g}" for m in mean_metrics for g in groups]].mean(axis=0), *tracker_result_df.loc[:, [f"{m}_{g}" for m in count_metrics for g in groups]].sum(axis=0)]
        print(f"MOTA: {tracker_result_df.loc['total', 'mota_total']}")
        tracker_result_df.to_csv(f"{preds_dir}/{tracker}/results_{criterion}.csv")
        print(f"Tracker results dumped to: {preds_dir}/{tracker}/results_{criterion}.csv")

        # Add total row to final result df
        final_result_df.loc[tracker, :] = tracker_result_df.loc["total", :]

    # Dump final_result_df to file
    final_result_df.to_csv(f"{preds_dir}/results_{criterion}.csv")
    print(f"> Full overview dumped to: {preds_dir}/results_{criterion}.csv")      

