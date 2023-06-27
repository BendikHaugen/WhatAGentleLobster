from argparse import ArgumentParser
from evaluation_helpers import evaluate_kp_trackers, create_metric_overview, VideoAnnotator, evaluate_trackers
from pathlib import Path


def main(dataset_to_evaluate="LOBSTER", just_evaluate=False):
    models = [  
        "YoloV8l+BoTSort",
        "YoloV8m+BoTSort",
        "YoloV8n+BoTSort",
        "YoloV8s+BoTSort",
        "YoloV8x+BoTSort",
        "YoloV8+BotSort",
        "YoloV8l+ByteTrack",
        "YoloV8m+ByteTrack",
        "YoloV8n+ByteTrack",
        "YoloV8s+ByteTrack",
        "YoloV8x+ByteTrack",
        "YoloV8+ByteTrack",
    ]
    weights = [  
        "yolomodels/yolo8l/weights/best.pt",
        "yolomodels/yolo8m/weights/best.pt",
        "yolomodels/yolo8n/weights/best.pt",
        "yolomodels/yolo8s/weights/best.pt",
        "yolomodels/yolo8x/weights/best.pt",
        "yolomodels/yolo8/weights/best.pt",
        "yolomodels/yolo8l/weights/best.pt",
        "yolomodels/yolo8m/weights/best.pt",
        "yolomodels/yolo8n/weights/best.pt",
        "yolomodels/yolo8s/weights/best.pt",
        "yolomodels/yolo8x/weights/best.pt",
        "yolomodels/yolo8/weights/best.pt",
    ]
    names = [ 
        "YOLOv8-L + BoT-SORT",
        "YOLOv8-M + BoT-SORT",
        "YOLOv8-N + BoT-SORT",
        "YOLOv8-S + BoT-SORT",
        "YOLOv8-X + BoT-SORT",
        "YOLOv8 + BoT-SORT",
        "YOLOv8-L + ByteTrack",
        "YOLOv8-M + ByteTrack",
        "YOLOv8-N + ByteTrack",
        "YOLOv8-S + ByteTrack",
        "YOLOv8-X + ByteTrack",
        "YOLOv8 + ByteTrack",
    ]
    trackers = [
        "BoT-SORT",
        "BoT-SORT",
        "BoT-SORT",
        "BoT-SORT",
        "BoT-SORT",
        "BoT-SORT",
        "ByteTrack",
        "ByteTrack",
        "ByteTrack",
        "ByteTrack",
        "ByteTrack",
        "ByteTrack",
    ]

    # Evaluation configuration
    metrics = {
        "METRICS": [
            "HOTA",  # Higher Order Tracking Accuracy and associated metrics
            "CLEAR",  # CLEAR MOT Metrics, e.g. MOTA, MOTP, ID sw.
            "Identity",  # ID Metrics, i.e. IDF1, IDP, IDR
        ],
        "THRESHOLD": 0.5,
    }
    ground_truth_folder = "data/test_data"
    kp_ground_truth_folder = "data/test_data_keypoints"
    trackers_folder = f"data/tracker_results"
    trackers_kps_folder = f"data/tracker_results_keypoints"
    split_to_evaluate = "test"

    

    if not just_evaluate:
        # Create tracking data for all models
        for i, (model, weight, name, tracker) in enumerate(zip(models, weights, names, trackers)):
            print(f">>> Evaluating model {i+1}/{len(models)}: {name}")
            output_folder = (
                f"{trackers_folder}/{dataset_to_evaluate}-{split_to_evaluate}/{name}"
            )
            kp_output_folder = (
                f"{trackers_kps_folder}/{dataset_to_evaluate}-{split_to_evaluate}/{name}"
            )

            with open(
                Path(ground_truth_folder).joinpath(
                    f"seqmaps/{dataset_to_evaluate}-{split_to_evaluate}.txt"
                ),
                "r",
            ) as f:
                sequences = [s.strip() for s in f.readlines()[1:]]

            
            annotator = VideoAnnotator(
                model_type=model, weights=weight, tracker=tracker
            )
            avg_fps_list = []

            for sequence in sequences:
                # Annotate images using VideoAnnotator:
                _, seq_avg_fps = annotator.annotate_video(
                    input_path=f"{ground_truth_folder}/{dataset_to_evaluate}-{split_to_evaluate}/{sequence}/images",
                    write_video=True,
                    output_video=f"{output_folder}/video/{sequence}.mp4",
                    write_tracker_output=True,
                    tracker_output_file=f"{output_folder}/data/{sequence}.txt",
                    keypoints_output_file=f"{kp_output_folder}/data/{sequence}.txt"
                )

                avg_fps_list.append(seq_avg_fps)
            with open(Path(output_folder).joinpath("fps_summary.txt"), "w") as f:
                f.write("sequence,fps\n")
                f.writelines(
                    [f"{seq},{fps}\n" for seq, fps in zip(sequences, avg_fps_list)]
                )
                f.write(f"average,{sum(avg_fps_list) / len(avg_fps_list)}\n")


    evaluate_trackers(
        metrics,
        ground_truth_folder,
        trackers_folder,
        split_to_evaluate,
        dataset_to_evaluate,
    )

    create_metric_overview(dataset_to_evaluate)

if __name__ == "__main__":
    parser = ArgumentParser(description="Interface for annotating lobster videos")
    parser.add_argument(
        "--just_evaluate",
        default=True,
        action="store_true",
        help="Set flag if model prediction step should be skipped",
    )
    parser.add_argument(
        "--dataset",
        default="LOBSTER",
        help="Which dataset to use",
        type=str,
        choices=["LOBSTER", "ROBOARM"],
    )
    args = parser.parse_args()
    main(dataset_to_evaluate=args.dataset, just_evaluate=args.just_evaluate)
