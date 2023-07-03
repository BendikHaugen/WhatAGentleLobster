from  sleap.io.format.coco import LabelsCocoAdaptor
from sleap.io.format.filehandle import FileHandle
from sleap.io.format.main import write
import json

"""
Script used for converting local cooc datasets to SLP format
"""

adaptor = LabelsCocoAdaptor()

"""
categories had to be altered, and included:
skeleton=[[0,1], [1,2], [2,3], [3,6], [6,3], [4,5], [5,6]]
keypoints=["left_claw", "leftElbow","leftEye","tail","rightClaw","rightElbow","rightEye"]
assume this is the case for other coco-files as well
"""

train_file = FileHandle(filename="training_data.json")
#train_file = FileHandle(filename="hundredSamples.json")
#val_file = FileHandle(filename="training_data\coco_training\\val_with_track_id.json")
"""
with open("training_data/coco_training/train_with_track_id.json") as f1, open("training_data\coco_training\\val_with_track_id.json") as f2:
    first_list = json.load(f1)
    second_list = json.load(f2)

first_list.update(second_list)

with open("training_data.json", "w") as f3:
    json.dump(first_list, f3)

"""
# Not sure if needed
skeleton_file=FileHandle(filename="skeleton_lobster_eyes.json") 

filehandle_train_json = adaptor.read(train_file, img_dir="trainImages", skeleton=skeleton_file)
#filehandle_val_json = adaptor.read(val_file, img_dir="training_data\coco_training\\val", skeleton=skeleton_file)

write("full_train.slp", filehandle_train_json)
#write("val_with_track_id.slp", filehandle_val_json)


