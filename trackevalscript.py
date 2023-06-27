from queue import Queue
from threading import Thread
from ultralytics import YOLO
import cv2
import os
import glob
import pickle
import numpy as np


def create_video_from_images(folder_path, video_name='output.mp4', fps=30):
    # Get all the image paths from the folder
    img_paths = glob.glob(os.path.join(folder_path, '*'))
    # Read the first image to get the shape
    frame = cv2.imread(img_paths[0])
    height, width, _ = frame.shape

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID' for .avi files
    
    # Get the current working directory
    cwd = os.getcwd()
    video_name = os.path.join(cwd, video_name)
    
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Loop through the images and write them to the video
    for img_path in sorted(img_paths):
        frame = cv2.imread(img_path)
        video.write(frame)

    # Finally, release the video writer
    video.release()

def check_for_mp4(directory_path):
    # Get all directories in the specified folder
    directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    
    for directory in directories:
        # Construct the mp4 file name using the directory name
        mp4_file_name = f"{directory}.mp4"
        
        # Use glob to check if the file exists in the directory
        matches = glob.glob(os.path.join(directory_path, directory, mp4_file_name))
        
        # Print the results
        if matches:
            print(f"Found {mp4_file_name} in {directory}")
        else:
            print(f"Did not find {mp4_file_name} in {directory}, creating viedo.")
            create_video_from_images(f"{directory_path}/{directory}/images", video_name=f"{directory}.mp4", fps=30)

# Replace with your directory path
directory_path = 'data/test_data/LOBSTER-test'
check_for_mp4(directory_path)


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



for i, model in enumerate(models):
    print(f"Evaluating {models[i]}")
    model = YOLO(weights[i])
    print("Model loaded")
    # Select tracker
    if trackers[0] == "BoT-SORT":
        tracker_config = "botsort.yaml"
    elif trackers[0] == "ByteTrack":
        tracker_config = "bytetrack.yaml"
    else:
        raise NotImplementedError("No tracker yaml file exists for " + model[i] + " and " + trackers[i])
    
    root_dir = 'data/test_data/LOBSTER-test'
    sequences = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.mp4'):
                full_file_path = os.path.join(dirpath, file)
                sequences.append([full_file_path, file])    
    output_dir = "data/tracker_results/LOBSTER-test"
    print(f"Starting inference for {names[i]}")
    for sequence in sequences:
        output_path = f"data/tracker_results/LOBSTER-test/{names[i]}/data/{sequence[1].replace('.mp4', '.txt')}"
        print(f"Performing inference and tracking on {sequence[1]}...")
        results = model.track(source=sequence[0], conf=0.3, iou=0.5, tracker=tracker_config)
        print(f"Inference finished. Writing to file {output_path}")
        with open(output_path, "w") as f:
            for index, result in enumerate(results):
                a = np.array([obj.boxes.xyxy.cpu().numpy() for obj in result])
                b = np.array([obj.boxes.id.cpu().numpy() for obj in result])
                b = b.reshape(-1 , 1, 1) 
                tracker_output = np.concatenate((a, b), axis=2)
                for instance in tracker_output:
                    instance = instance[0]
                    f.write(
                                f"{index + 1},{int(instance[4])},{instance[0]},{instance[1]},{instance[2]-instance[0]},{instance[3]-instance[1]},-1,-1,-1,-1\n"
                            )
    
