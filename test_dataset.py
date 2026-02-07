from datasets.coco_dataset import COCODataset
from datasets.transforms import make_coco_transforms

dataset = COCODataset(
    img_dir="data/coco_subset/images",
    ann_file="data/coco_subset/annotations/instances_subset.json",
    transforms=make_coco_transforms()
)

image, target = dataset[0]

print("Image shape:", image.shape)
print("Boxes:", target["boxes"].shape)
print("Labels:", target["labels"][:5])
