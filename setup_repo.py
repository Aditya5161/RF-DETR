import os
from pathlib import Path

BASE_DIR = Path(".")

folders = [
    "configs",
    "data/coco/images/train2017",
    "data/coco/images/val2017",
    "data/coco/annotations",
    "data/videos",
    "models",
    "tracking",
    "datasets",
    "utils"
]

files = {
    "README.md": "# RF-DETR Object Detection & Tracking\n\nResearch-grade local setup.",
    "requirements.txt": """torch>=2.1.0
torchvision>=0.16.0
numpy
opencv-python
pyyaml
scipy
matplotlib
tqdm
filterpy
scikit-image
""",
    ".env": """DEVICE=cuda
NUM_CLASSES=80
IMG_SIZE=640
""",
    "configs/rf_detr.yaml": """model:
  name: rf_detr
  hidden_dim: 256
  num_queries: 100

training:
  batch_size: 4
  epochs: 50
  lr: 1e-4
""",
    "configs/tracking.yaml": """tracker:
  type: SORT
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
""",
    "models/__init__.py": "",
    "models/rf_detr.py": """# RF-DETR model placeholder""",
    "tracking/__init__.py": "",
    "tracking/sort.py": """# SORT tracker placeholder""",
    "datasets/__init__.py": "",
    "datasets/coco_dataset.py": """# COCO dataset loader""",
    "utils/__init__.py": "",
    "utils/visualization.py": """# Visualization utilities""",
    "train.py": """# Training script""",
    "infer_video.py": """# Video inference script"""
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    for file, content in files.items():
        path = BASE_DIR / file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    print("✅ RF-DETR repo structure created successfully.")

if __name__ == "__main__":
    create_structure()
