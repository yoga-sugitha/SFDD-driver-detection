"""
Lightning DataModule for SFDD dataset with binary/multi-class support
"""
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from sklearn.model_selection import train_test_split
from .dataset import ImagePathDataset

class SFDDDataModule(L.LightningDataModule):
    """
    Lightning DataModule for State Farm Distracted Driver Detection
    Supports both binary (normal vs distracted) and multi-class (10 classes) classification
    
    Args:
        data_dir: Path to training data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        val_split: Validation split ratio
        img_size: Image size for resizing
        seed: Random seed for reproducibility
        task_type: Classification type - 'binary' or 'multiclass'
        binary_mapping: How to map classes for binary classification
                       'c0_vs_rest': c0 (normal) = 0, c1-c9 (distracted) = 1 (default)
                       'custom': provide custom mapping via binary_class_map
        binary_class_map: Custom dict mapping original class -> binary class (0 or 1)
    """
    def __init__(
        self, 
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
        img_size: int = 224,
        seed: int = 42,
        task_type: str = 'multiclass',  # 'binary' or 'multiclass'
        binary_mapping: str = 'c0_vs_rest',
        binary_class_map: dict = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.img_size = img_size
        self.task_type = task_type.lower()
        self.binary_mapping = binary_mapping
        self.binary_class_map = binary_class_map

        # Immediately compute and store
        self.num_classes = 2 if self.task_type == 'binary' else 10
        self.class_names = (
            ['Normal Driving', 'Distracted Driving'] if self.task_type == 'binary' else
            ["normal driver", "texting-right", "talking on the phone-right",
            "texting-left", "talking on the phone-left", "operating the radio",
            "drinking", "reaching behind", "hair and makeup", "talking to passenger"]
        )
        
        # Validate task type
        if self.task_type not in ['binary', 'multiclass']:
            raise ValueError(f"task_type must be 'binary' or 'multiclass', got '{task_type}'")
        
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
        Setup datasets for different stages.
        Handles binary vs multiclass label conversion.
        """
        # Only do the expensive split once
        if not hasattr(self, '_splits_created'):
            self.full_dataset = datasets.ImageFolder(self.data_dir)
            self.paths = [sample[0] for sample in self.full_dataset.imgs]
            self.original_labels = [sample[1] for sample in self.full_dataset.imgs]
            
            # Convert labels based on task type
            if self.task_type == 'binary':
                self.labels = self._convert_to_binary(self.original_labels)
                print(f"\n✓ Converted to binary classification:")
                print(f"  Class 0 (normal): {self.labels.count(0)} samples")
                print(f"  Class 1 (distracted): {self.labels.count(1)} samples")
            else:
                self.labels = self.original_labels
                print(f"\n✓ Using multi-class classification (10 classes)")

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

    
    def get_test_sample(self, idx: int):
        """
        Return (image: Tensor, label: int) for test set index `idx`.
        
        The image is returned **as transformed by test_transform** (normalized, resized).
        This matches exactly what your test_dataloader returns.
        
        Args:
            idx: Index in the test split (0 to len(test_dataset)-1)
            
        Returns:
            (image: torch.Tensor in [C, H, W], label: int)
        """
        if not hasattr(self, '_test_paths') or not hasattr(self, '_test_labels'):
            # Ensure test split exists
            self.setup(stage='test')
        
        if idx < 0 or idx >= len(self._test_paths):
            raise IndexError(f"Test set index {idx} out of range (0–{len(self._test_paths)-1})")
        
        path = self._test_paths[idx]
        label = self._test_labels[idx]
        
        # Load and transform using the same logic as ImagePathDataset
        from PIL import Image
        image = Image.open(path).convert('RGB')
        image = self.test_transform(image)  # Apply test-time transform (normalize, etc.)
        
        return image, label
    
    def _convert_to_binary(self, labels):
        """
        Convert multi-class labels to binary labels
        
        Args:
            labels: List of original class labels (0-9)
            
        Returns:
            List of binary labels (0 or 1)
        """
        if self.binary_mapping == 'c0_vs_rest':
            # Class 0 (normal driving) vs rest (distracted)
            return [0 if label == 0 else 1 for label in labels]
        
        elif self.binary_mapping == 'custom':
            if self.binary_class_map is None:
                raise ValueError("binary_mapping='custom' requires binary_class_map to be provided")
            return [self.binary_class_map.get(label, label) for label in labels]
        
        else:
            raise ValueError(
                f"Unknown binary_mapping: {self.binary_mapping}. "
                f"Use 'c0_vs_rest' or 'custom'"
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