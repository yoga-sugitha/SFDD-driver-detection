"""
Lightning DataModule for SFDD dataset
"""
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from sklearn.model_selection import train_test_split
from .dataset import ImagePathDataset

class SFDDDataModule(L.LightningDataModule):
    """
    Lightning DataModule for State Farm Distracted Driver Detection
    
    Args:
        data_dir: Path to training data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        val_split: Validation split ratio
        img_size: Image size for resizing
        seed: Random seed for reproducibility
    """
    def __init__(
        self, 
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
        img_size: int = 224,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.img_size = img_size
        
        # Define transforms
        self.test_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            # Lighting augmentations
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.RandomApply([T.RandomGrayscale(p=1.0)], p=0.1),
            # Geometric augmentations (mild to preserve driver pose)
            T.RandomAffine(
                degrees=3,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=None
            ),
            T.RandomPerspective(distortion_scale=0.05, p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize dataset placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Setup datasets for different stages
        
        Args:
            stage: Stage ('fit', 'test', or None for all)
        """
        # Only do the expensive split once
        if not hasattr(self, '_splits_created'):
            self.full_dataset = datasets.ImageFolder(self.data_dir)
            self.paths = [sample[0] for sample in self.full_dataset.imgs]
            self.labels = [sample[1] for sample in self.full_dataset.imgs]

            # Stratified split - 60% train, 20% val, 20% test
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                self.paths, self.labels, 
                test_size=0.2, 
                stratify=self.labels, 
                random_state=self.seed
            )
            
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels, 
                test_size=0.25,  # 0.25 * 0.8 = 0.2 of total
                stratify=train_val_labels, 
                random_state=self.seed
            )
            
            # Store splits
            self._train_paths = train_paths
            self._train_labels = train_labels
            self._val_paths = val_paths
            self._val_labels = val_labels
            self._test_paths = test_paths
            self._test_labels = test_labels
            self._splits_created = True
        
        # Create datasets based on stage
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                self.train_dataset = ImagePathDataset(
                    self._train_paths, 
                    self._train_labels, 
                    transform=self.train_transform
                )
            if self.val_dataset is None:
                self.val_dataset = ImagePathDataset(
                    self._val_paths, 
                    self._val_labels, 
                    transform=self.test_transform
                )
        
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                self.test_dataset = ImagePathDataset(
                    self._test_paths, 
                    self._test_labels, 
                    transform=self.test_transform
                )
    
    def train_dataloader(self):
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
