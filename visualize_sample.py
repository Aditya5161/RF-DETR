import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np  # noqa: F401

from datasets.coco_dataset import COCODataset
from datasets.transforms import make_coco_transforms

# --------------------------------------------------
# COCO category ID → name mapping (official)
# --------------------------------------------------
COCO_ID_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

# --------------------------------------------------
# Unnormalize helper
# --------------------------------------------------
def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean

# --------------------------------------------------
# Dataset
# --------------------------------------------------
dataset = COCODataset(
    img_dir="data/coco_subset/images",
    ann_file="data/coco_subset/annotations/instances_subset.json",
    transforms=make_coco_transforms()
)

print(f"Dataset size: {len(dataset)} images")

# --------------------------------------------------
# Visualization config
# --------------------------------------------------
NUM_SAMPLES = 4
fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(20, 6))

if NUM_SAMPLES == 1:
    axes = [axes]

# --------------------------------------------------
# Plot samples
# --------------------------------------------------
for i in range(NUM_SAMPLES):
    image, target = dataset[i]

    image = unnormalize(image).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    ax = axes[i]
    ax.imshow(image)
    ax.set_title(f"Sample {i}")
    ax.axis("off")

    boxes = target["boxes"]
    labels = target["labels"]

    for box, label in zip(boxes, labels):
        x, y, w, h = box.tolist()

        class_name = COCO_ID_TO_NAME.get(label.item(), "unknown")

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x, y - 5,
            class_name,
            fontsize=10,
            color="white",
            bbox=dict(facecolor="red", alpha=0.7, pad=2)
        )

plt.tight_layout()
plt.show()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

# import random
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# from datasets.coco_dataset import COCODataset
# from datasets.transforms import make_coco_transforms
# from datasets.coco_classes import COCO_CLASSES


# # -----------------------------
# # Helper: denormalize image
# # -----------------------------
# def denormalize(img):
#     """
#     img: Tensor [3, H, W] normalized with ImageNet stats
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#     img = img * std + mean
#     return img.clamp(0, 1)


# # -----------------------------
# # Helper: random color per box
# # -----------------------------
# def random_color():
#     return (
#         random.random(),
#         random.random(),
#         random.random()
#     )


# # -----------------------------
# # Visualization
# # -----------------------------
# def visualize_dataset(dataset, num_images=3):
#     print(f"Dataset size: {len(dataset)}")

#     indices = random.sample(range(len(dataset)), num_images)

#     for idx, dataset_idx in enumerate(indices):
#         image, target = dataset[dataset_idx]

#         image = denormalize(image)
#         image = image.permute(1, 2, 0).numpy()

#         boxes = target["boxes"]
#         labels = target["labels"]

#         fig, ax = plt.subplots(1, figsize=(10, 8))
#         ax.imshow(image)
#         ax.set_title(f"Sample {idx + 1}")
#         ax.axis("off")

#         for box, label in zip(boxes, labels):
#             x1, y1, x2, y2 = box.tolist()
#             width = x2 - x1
#             height = y2 - y1

#             color = random_color()

#             rect = patches.Rectangle(
#                 (x1, y1),
#                 width,
#                 height,
#                 linewidth=2,
#                 edgecolor=color,
#                 facecolor="none"
#             )
#             ax.add_patch(rect)

#             class_name = COCO_CLASSES[label.item()]

#             ax.text(
#                 x1,
#                 y1 - 5,
#                 class_name,
#                 color=color,
#                 fontsize=10,
#                 weight="bold",
#                 bbox=dict(facecolor="black", alpha=0.6, pad=2)
#             )

#         plt.show()


# # -----------------------------
# # Main
# # -----------------------------
# if __name__ == "__main__":
#     dataset = COCODataset(
#         img_dir="data/coco_subset/images",
#         ann_file="data/coco_subset/annotations/instances_subset.json",
#         transforms=make_coco_transforms()
#     )

#     visualize_dataset(dataset, num_images=3)
