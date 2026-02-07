import json
import random
import os
import shutil

# ✅ FULL COCO LOCATION (matches your screenshot)
SOURCE_ROOT = r"C:\Users\Aditya Pandey\Downloads\coco_dataset\coco2017"

# ✅ DESTINATION inside RF-DETR repo
DEST_ROOT = r"data\coco_subset"

NUM_IMAGES = 200  # start small

# Create output folders
os.makedirs(os.path.join(DEST_ROOT, "images"), exist_ok=True)
os.makedirs(os.path.join(DEST_ROOT, "annotations"), exist_ok=True)

# Load COCO annotations
ann_file = os.path.join(
    SOURCE_ROOT, "annotations", "instances_val2017.json"
)

with open(ann_file, "r") as f:
    coco = json.load(f)

# Randomly sample images
selected_images = random.sample(coco["images"], NUM_IMAGES)
selected_ids = {img["id"] for img in selected_images}

# Filter annotations
selected_annotations = [
    ann for ann in coco["annotations"]
    if ann["image_id"] in selected_ids
]

# Save subset annotation file
subset = {
    "images": selected_images,
    "annotations": selected_annotations,
    "categories": coco["categories"]
}

subset_ann_path = os.path.join(
    DEST_ROOT, "annotations", "instances_subset.json"
)

with open(subset_ann_path, "w") as f:
    json.dump(subset, f)

# Copy images
for img in selected_images:
    src = os.path.join(SOURCE_ROOT, "val2017", img["file_name"])
    dst = os.path.join(DEST_ROOT, "images", img["file_name"])
    shutil.copy(src, dst)

print(f"✅ COCO subset created with {NUM_IMAGES} images")
