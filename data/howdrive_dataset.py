import os
from PIL import Image
from torch.utils.data import Dataset

class HowDriveDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Dataset that loads pre-specified image paths and labels.
        Used after subject-wise split has already been performed.

        Args:
            image_paths (list[str]): Full paths to images
            labels (list[int]): Corresponding class labels (0–9)
            transform (callable, optional): Transform to apply to images
        """
        assert len(image_paths) == len(labels), "Number of paths and labels must match"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

    # Optional: useful for logging or inference
    @property
    def class_to_idx(self):
        return {i: i for i in range(10)}  # c0→0, ..., c9→9