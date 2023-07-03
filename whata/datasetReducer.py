import json

file = open("lobster_coco_data/lobster_coco_data/annotations/train_rounded.json")

data_full = json.load(file)

images = data_full["images"][:100]
image_ids = list(set([image['id'] for image in images]))

annotations = [element for element in data_full["annotations"] if element["image_id"] in image_ids]

data_formatted = {
    "annotations": annotations,
    "images": images,
    "categories": data_full["categories"],
    "licences": data_full["licences"],
    "info": data_full["info"]
}

with open("hundredSamples.json", "w") as f:
    json.dump(data_formatted, f)
    f.close()

import shutil

for jpg in images:
    shutil.copy(f"lobster_coco_data/lobster_coco_data/images/train/{jpg['file_name']}", "hundredImages")

