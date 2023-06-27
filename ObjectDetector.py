import cv2
import torch
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class YOLODetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        model = YOLO(model_path)
        model.fuse()
        self.model = model
        self.CLASS_NAMES_DICT = self.model.model.names

    def __call__(self, img, max_det):
        # Only predicts, access self.model if performing tracking
        results = self.model.predict(img, max_det=max_det)
        return results
    

