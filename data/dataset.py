"""
Custom dataset classes
"""
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Callable

class ImagePathDataset(Dataset):
    """
    Dataset that loads images from file paths
    
    Args:
        paths: List of image file paths
        labels: List of labels corresponding to images
        transform: Optional transform to apply to images
    """
    def __init__(
        self, 
        paths: List[str], 
        labels: List[int], 
        transform: Optional[Callable] = None
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
        assert len(paths) == len(labels), "Paths and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
