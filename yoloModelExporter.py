from ultralytics import YOLO
import os

def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

directory='yolomodels'
dirs = list_directories(directory)

print("All available yolomodels: \n")
print(dirs)

for dir in dirs:
    model = YOLO(f"{directory}/{dir}/weights/best.pt")
    model.export(format="pb", imgsz=(1920, 1088))