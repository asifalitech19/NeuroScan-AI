import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from pathlib import Path

class NEUDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.samples = []
        for xml_path in sorted(self.annotation_dir.glob("*.xml")):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename_node = root.find("filename")
            if filename_node is None:
                continue
            filename = filename_node.text.strip()
            stem = Path(filename).stem
            # Find matching image using filename stem
            candidates = [
                p for p in self.image_dir.rglob("*")
                if p.is_file() and p.stem.lower() == stem.lower()
            ]
            if len(candidates) == 0:
                continue
            self.samples.append((candidates[0], xml_path))
        print(f"Loaded {len(self.samples)} image/XML pairs") #

    def __len__(self):
        return len(self.samples) #[cite: 1]

    def __getitem__(self, idx):
        image_path, xml_path = self.samples[idx]
        # Read real image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}") #[cite: 1]
        height, width = image.shape
        # Normalize image
        image = image.astype(np.float32) / 255.0 #[cite: 1]

        # Convert XML boxes to pseudo-mask
        mask = np.zeros((height, width), dtype=np.float32)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text) #[cite: 1]
            
            xmin = max(0, min(xmin, width - 1))
            xmax = max(0, min(xmax, width - 1))
            ymin = max(0, min(ymin, height - 1))
            ymax = max(0, min(ymax, height - 1)) #[cite: 1]
            
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin #[cite: 1]
                
            mask[ymin:ymax + 1, xmin:xmax + 1] = 1.0 #[cite: 1]

        # Convert to PyTorch format
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask #[cite: 1]

def get_dataloaders(base_dir_path, batch_size=16):
    BASE_DIR = Path(base_dir_path)
    TRAIN_IMG_DIR = BASE_DIR / "train" / "images"
    TRAIN_XML_DIR = BASE_DIR / "train" / "annotations"
    VAL_IMG_DIR = BASE_DIR / "validation" / "images"
    VAL_XML_DIR = BASE_DIR / "validation" / "annotations" #[cite: 1]

    train_dataset = NEUDataset(TRAIN_IMG_DIR, TRAIN_XML_DIR)
    val_dataset = NEUDataset(VAL_IMG_DIR, VAL_XML_DIR) #[cite: 1]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) #[cite: 1]

    return train_loader, val_loader
