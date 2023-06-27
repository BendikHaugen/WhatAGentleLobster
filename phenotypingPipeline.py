from typing import Dict, List, Tuple
from ObjectDetector import YOLODetector
from draw_utils import draw_lob_info
from keypointDetector import PoseEstimator
import cv2
import numpy as np
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from helpers import center_distance, only_bbs as mapping_func_bbox
from Sort.sort import Sort
import pickle
from bisect import bisect
from lobster import Lobster
from time import perf_counter, strftime
import pandas as pd


class PheotypingPipeline:
    def __init__(
        self,
        modelFolder="yolomodels/yolo8l/weights/best.pt",
        #modelFolder="tensorflowModels/leapTd/",
        videoPath="videos/channel1.mp4",
        ineteraction_detector_weights="models/MLP(1000,)_bbonly_ws5.pkl",  #MLP(100,100)_bbonly_ws1.pkl",
        disable_tensrflow_feedback=True,
        SLEAP=False
    ) -> None:
        # Keypoint / BB detector
        if SLEAP: 
            self.predictor = PoseEstimator(
                modelFolder,
                disable_tensrflow_feedback=disable_tensrflow_feedback
                )
        else:
            self.predictor = YOLODetector(modelFolder)


        # Interaction detector
        with open(ineteraction_detector_weights, 'rb') as f:
            self.interaction_detector = pickle.load(f)
        self.video_path = videoPath
        self.SLEAP = SLEAP
        # Min distance between the center of two bounding boxes around a lobster
        self.min_distance = 200
        # How many frames to look back to find a match for a lobster
        self.window_size = 5
        # When to start labelling agressive lobsters with dots
        self.aggression_detection = 20

        #dots:
        self.dot_cooldown = 90
        self.dot_window_size = 20
        self.dot_threshold = 0.6
        
    def load_video_as_images(self):
        print(f'Loading video from {self.video_path}...')
        # Load input video
        cap = cv2.VideoCapture(self.video_path)
        # Define list to store images
        images = []
        # Loop through video frames and convert to images
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Preprocess input image, Resize to input shape [height, width]
                if self.SLEAP:
                    image = cv2.resize(
                        frame, 
                        (self.predictor.input_tensor_shape[2], self.predictor.input_tensor_shape[1])
                    )  
                else:
                    image = frame
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                # Convert to uint8 data type
                image = image.astype(np.uint8)
                # Add batch dimension
                image = np.expand_dims(image, axis=0)
                images.append(image)
            else:
                break
        cap.release()
        self.images = images[:10]
        print(f'Loaded {len(self.images)} frames from video.')
        return images
    
    def _setup_video_output(self, output_path, frame_provider, frames=None):
        self._check_target_dir(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if frame_provider.is_cam:
            sample_frame = frames[0]
        else:
            sample_frame = next(iter(frame_provider))
        video_shape = (sample_frame.shape[1], sample_frame.shape[0])
        video_out = cv2.VideoWriter(output_path, fourcc, 30.0, video_shape, True)

        return video_out

    def _check_target_dir(self, output_path):
        target_dir = Path(output_path).parent
        if not target_dir.exists():
            target_dir.mkdir(parents=True)

    def make_video(
            self,
            lobsters: List[Lobster],
            output_path: str = "test.mp4",       
    ):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            30.0, 
            (1920, 1080), 
            True
        )
        for frame_index, frame in tqdm(
            enumerate(self.images),
            desc="Producing video",
            unit=" frames",
            total=len(self.images)
        ): 
            """
            Need to ensure that indexed match the frame index
            Might be a delay from when interaction and aggrassion detection start
            """
            max_aggression = max([lobster.acc_aggression[frame_index] for lobster in lobsters]) / (frame_index + 1)
            leaderboard: Dict[int, Tuple[float, int]] = {}
            for lobster in lobsters:
                aggression = lobster.acc_aggression[-frame_index] / (frame_index + 1)
                dots = bisect(lobster.dot_timestamps, frame_index)
                leaderboard[lobster.id] = (aggression, dots)
                if np.any(np.isnan(lobster.boundig_boxes[frame_index])):
                    # Might also be a good idea to check if the keypoints are Nan
                    continue
                id_str = f"{lobster.id} {aggression:.2f} "
                id_str += "*" * dots if dots > 0 else ""
                draw_lob_info(
                    frame=frame,
                    lobster=lobster,
                    frame_index=frame_index,
                    show_bb=True,
                    id_str=id_str,
                    show_id=True,
                    show_trace=True,   
                )
            video_out.write(frame[0])

        video_out.release()
            

    def __call__(
            self, 
            num_lobsters: int,
            num_keypoints: int = 7,
            annotate_video: bool = True,
            output_path: str = "test.mp4", 
    ):
        """
        Adds Keypoints to the frames and saves a video.
        """
        if not self.images:
            # Could call it instead of raising error
            ValueError("No video loaded. Please run load_video_as_images() first with path to video before calling make_video().")
        lobsters = [Lobster(i, len(self.images)) for i in range(1, num_lobsters + 1)]
        tracker = Sort(num_lobsters)
        
        printed_keypoint_warning = False

        # Setup timers
        predict_times = []
        detect_times = []
        track_times = []
        frame_times = []
        feature_eng_times = []
        mapping_func_times = []
        cb_pred_times = []
        lob_prep_times = []
        badness_times = []
        dots_times = []
                
        
        inference_start_time = perf_counter()
        for frame_index, frame in tqdm(
            enumerate(self.images), 
            desc="Perform tracking and phenotying", 
            unit=" frames", 
            total=len(self.images)
        ):
            frame_time_start = perf_counter()
            frame_inference_start = perf_counter()
            if self.SLEAP:
                keypoints = self.predictor(frame)
                frame_inference_end = perf_counter()

                """
                    This is non-ideal way of ensuring correct shape. The pedrictor has a tendency of returning 
                    keypoint with almost exclusively nan values. I chose to remove these keypoints, but this
                    should be investigated further
                    TODO: refactor and make cleaner
                """
                nan_mask = np.isnan(keypoints)
                # Check which elements contain `nan` values along the second axis
                #nan_in_row = np.any(np.any(nan_mask, axis=2)[0], axis=1)  
                nan_in_row = np.any(nan_mask, axis=2)[0]
                nan_in_row = np.any(nan_in_row, axis=1)
                orignal_shape = keypoints.shape
                nan_in_row = nan_in_row.reshape((orignal_shape[0], len(nan_in_row), 1, 1))
                # Drop the rows that contain `nan` values
                keypoints = keypoints[~nan_in_row[:, :, 0, 0]]
                try:
                    keypoints = keypoints.reshape((1, num_lobsters, num_keypoints, 2))
                except:
                    # This is a superstupid temp solution; use som cv2 shortest distance to remove the one that
                    # is furthest away instead
                    keypoints = keypoints[:7, :, :].reshape((1, num_lobsters, num_keypoints, 2))
            else:
                tracking_start_time = perf_counter()
                res = self.predictor.model.track(source=frame[0], conf=0.3, iou=0.5)
                tracking_end_time = perf_counter()
                frame_inference_end = perf_counter()
                # Extract each box as xywh element and ID
                bb_cordinates = [obj.boxes.xyxy.cpu().numpy()  for obj in res][0]
                ids = [obj.boxes.id.cpu().numpy() for obj in res][0]
            if not printed_keypoint_warning and self.SLEAP:
                print("\nNumber of lobsters predicted does not match number of lobsters in the video, removing predicted frames with nan values.")
                printed_keypoint_warning = True

            # Loop over each instance and calculate the bounding box
            # Used by the tracker
            if self.SLEAP:
                bb_cordinates = []
                for i in range(num_lobsters):
                    # Get the keypoints for the current instance
                    instance_keypoints = keypoints[0][i]
                    # Calculate the bounding box coordinates
                    x, y, w, h = cv2.boundingRect(instance_keypoints)
                    # Add the bounding box coordinates to the list
                    # Append 1 ad placeholder for confidence
                    bb_cordinates.append([(x - ( (w*1.5 - w) //2)), y - ((h*1.5-h) // 2), x + (w * 1.5), y + (h * 1.5), 1])
                # Add list as column to the own array
                bb_cordinates = np.array(bb_cordinates)
                # Append 1 to each keypoint as placeholder for confidence
                tracker_input = np.concatenate([keypoints, np.ones((1, num_lobsters, num_keypoints, 1))], axis=-1)
                # Flatten the matrix over instances and combine with bbox for tracking
                tracker_input = tracker_input.squeeze()
                tracker_input = tracker_input.reshape((num_lobsters, -1))
                tracker_input = np.hstack((bb_cordinates, tracker_input))
                # Update tracker
                tracking_start_time = perf_counter()
                tracker_out = tracker.update(tracker_input)
                tracking_end_time = perf_counter()
                # Parse tracker data
                tracker_bbox = tracker_out[:, :4]
                tracker_keypoints = tracker_out[:, 5:].reshape((-1,21))
                tracker_ids = tracker_out[:, 4].astype(int)
                # Add information to lobsters
                lob_update_start = perf_counter()
                for id, bbox, tracker_kp in zip(tracker_ids, tracker_bbox, tracker_keypoints):
                    # Could be smart to format to coco format here
                    lobster = lobsters[id - 1]
                    lobster.new_frame(bbox, tracker_kp.reshape((num_keypoints, 3)))
            else:
                lob_update_start = perf_counter()
                for idx, id in enumerate(ids):
                    lobster = lobsters[int(id) - 1]
                    lobster.new_frame(bb_cordinates[idx])
            lob_update_end = perf_counter()
            """
            Interaction detection
            We use the is_interacting property to determine if a lobster is interacting with another lobster
            The idea is that to cope with fps demands, we may only want to track keypoints on lobsters that are interacting
            """
            interaction_detection_start_index = self.window_size
            if frame_index >= interaction_detection_start_index:
                feature_engineering_start = perf_counter()
                bbs = [lobster.boundig_boxes[-1] for lobster in lobsters]
                #Create matrix with distances between all pairs of lobsters
                distances = center_distance(bbs, bbs)
                # Extract permutations of lobsters that are closer than minimum distance
                permutations = np.array(list(zip(*np.where(np.logical_and(float(0) < distances, distances <= self.min_distance)))))
                # Create interaction detection model input for each permutation
                mapping_func_start = perf_counter()
                detector_input = np.array(
                    [
                        mapping_func_bbox(
                            lobsters[l1], lobsters[l2], self.window_size
                        )
                        for l1, l2 in permutations
                    ]
                )
                mapping_func_end = perf_counter()
                # Filter out rows with NaNs
                if len(detector_input) != 0:
                    non_nans = ~np.any(np.isnan(detector_input), axis=1)
                    detector_input = detector_input[non_nans]
                    permutations = permutations[non_nans]
                    # Perform model prediction
                    feature_engineering_end = perf_counter()

                    probabilities = self.interaction_detector.predict_proba(detector_input)[:, 1]
                    for (l1, l2), prob in zip(permutations, probabilities):
                        lobsters[l1].interactions[frame_index][lobsters[l2].id] = prob
                        # Idea was to perform pose estimation on interacting models //TODO
                        lobsters[l1].is_interacting = True
                    det_model_end = perf_counter()
                else:
                    feature_engineering_end = perf_counter()
                    det_model_end = perf_counter()
            else:
                feature_engineering_start = perf_counter()
                mapping_func_start = perf_counter()
                mapping_func_end = perf_counter()
                feature_engineering_end = perf_counter()
                det_model_end = perf_counter()

            """
                Perform Phenotyping / aggression detection
                This part is supposed to detect when a lobster is deemed aggressive.
                Each time, the lobster is rewarded with a dot. When the number of dots received for a single 
                Lobster exceeds a threshold, the lobster should be eliminated. 
                Noe funky her
            """
            agg_start = perf_counter()
            for lobster in lobsters:
                aggression = 0
                if (len(lobster.interactions) > 0):
                    interactions = lobster.interactions[-1]
                    max_interaction_probability = max(interactions.values(), default=0)
                    aggression = max_interaction_probability if max_interaction_probability > 0.3 else 0
                    lobster.acc_aggression.append(aggression + lobster.acc_aggression[-1] if lobster.acc_aggression else 0)
            badness_end = perf_counter()



            dots_start = perf_counter()
            if frame_index > self.aggression_detection:
                for lobster in lobsters:
                    badness = 0
                    window = range(max(frame_index - self.dot_window_size, 0), frame_index)
                    for i in window:
                        badness += lobster.acc_aggression[i] - (lobster.acc_aggression[i - 1] if i > 0 else 0)
                    badness /= len(window)

                    if badness >= self.dot_threshold and (frame_index - lobster.dot_timestamps[-1] > self.dot_cooldown if len(lobster.dot_timestamps) > 0 else True):
                        lobster.dot_timestamps.append(frame_index)
            dots_end = perf_counter()

            # Timing
            predict_times.append(frame_inference_end - frame_inference_start)
            track_times.append(tracking_end_time - tracking_start_time)
            detect_times.append(agg_start - tracking_end_time)
            lob_prep_times.append(lob_update_end - lob_update_start)
            badness_times.append(badness_end - agg_start)
            dots_times.append(dots_end - dots_start)
            feature_eng_times.append(
                feature_engineering_end - feature_engineering_start
            )
            mapping_func_times.append(mapping_func_end - mapping_func_start)
            cb_pred_times.append(det_model_end - feature_engineering_end)
            frame_time_end = perf_counter()
            frame_times.append(frame_time_end - frame_time_start)

        inference_time = perf_counter() - inference_start_time


        print(f"Inference complete for {self.video_path}")
        print(f"Total inference time: {inference_time:.3f} s")
        fpss = 1.0 / np.array(frame_times)
        print(f"Average fps: {sum(fpss)/len(fpss):.3f}")
        print(f"Minimum fps: {min(fpss):.3f}")
        print(f"Maximum fps: {max(fpss):.3f}")


        # Save FPS values and timings for further analysis later
        timing_df = pd.DataFrame(
            data=[
                frame_times,
                predict_times,
                track_times,
                lob_prep_times,
                badness_times,
                dots_times,
                feature_eng_times,
                mapping_func_times,
                cb_pred_times,
                fpss,
            ],
            index=[
                "Frame times",
                "Obj/kp det model prediction times",
                "Tracking times",
                "Lobster update times",
                "Badness calculation times",
                "Strike system times",
                "AID feature creation times",
                "AID feature mapping times",
                "AID model prediction times",
                "FPS",
            ],
        ).transpose()
        timing_df.to_csv(
            f"{'.'.join(output_path.split('.')[:-1])}_timings.csv", index=False
        )

        if annotate_video:
            print("Annotating video...")
            self.make_video(lobsters, output_path=output_path)

        print(f"Tracking performed on {self.video_path}.")
        
        return 0
            


if __name__ == "__main__":
    pipeline = PheotypingPipeline(SLEAP=False)
    pipeline.load_video_as_images()
    pipeline(7)