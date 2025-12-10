"""
Lightning DataModule for SFDD dataset with subject-wise split (person-disjoint)
Supports multi-class only in this pipeline; binary args retained for compatibility.
"""
import os
import random
from PIL import Image
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T
from typing import List, Tuple
from .dataset import ImagePathDataset

class HowDriveDataModule(L.LightningDataModule):
    """
    Lightning DataModule with **subject-wise (person-disjoint) splitting**.
    Designed for fair evaluation on driver state recognition.
    
    The dataset is expected to have:
        data_dir/
          ├── subject1/
          │   ├── c0/ ... c9/
          ├── subject2/
          │   ├── c0/ ... c9/
          ...
    
    Args remain unchanged for compatibility, but:
      - `val_split` is ignored (fixed subject split used)
      - `task_type` must be 'multiclass' (binary not supported in this split)
    """
    def __init__(
        self, 
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,  # ignored
        img_size: int = 224,
        seed: int = 42,
        task_type: str = 'multiclass',
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

        # Enforce multi-class for this pipeline (subject-wise split assumes 10 classes)
        if self.task_type != 'multiclass':
            raise ValueError("This pipeline only supports 'multiclass' (subject-wise split). "
                             "Binary classification not implemented for person-disjoint evaluation.")

        self.num_classes = 10
        self.class_names = [
            "safe driver", "texting-right", "talking on the phone-right",
            "texting-left", "talking on the phone-left", "operating the radio",
            "drinking", "reaching behind", "hair and makeup", "talking to passenger"
        ]
        
        # Define transforms
        self.test_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.RandomApply([T.RandomGrayscale(p=1.0)], p=0.1),
            T.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandomPerspective(distortion_scale=0.05, p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_subject_class_paths(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Walk the dataset and return (image_paths, labels, subjects)
        """
        image_paths = []
        labels = []
        subjects = []
        class_dirs = [f"c{i}" for i in range(10)]

        for subject in sorted(os.listdir(self.data_dir)):
            subject_path = os.path.join(self.data_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            for cls_dir in class_dirs:
                cls_path = os.path.join(subject_path, cls_dir)
                if not os.path.exists(cls_path):
                    continue
                for img_name in os.listdir(cls_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(cls_path, img_name))
                        labels.append(int(cls_dir[1:]))  # 'c0' → 0
                        subjects.append(subject)
        return image_paths, labels, subjects

    def setup(self, stage=None):
        if hasattr(self, '_datasets_prepared'):
            return

        # Get all data
        all_paths, all_labels, all_subjects = self._get_subject_class_paths()
        if not all_paths:
            raise ValueError(f"No images found in {self.data_dir}. Check directory structure.")

        # Get unique subjects and set split (deterministic with seed)
        unique_subjects = sorted(set(all_subjects))
        random.seed(self.seed)
        shuffled = random.sample(unique_subjects, len(unique_subjects))

        # Fixed: 5 train, 2 val, 2 test (for 9 subjects)
        # Adapt if subject count differs
        n = len(shuffled)
        if n < 3:
            raise ValueError("Need at least 3 subjects for train/val/test split")
        n_train = max(1, int(0.6 * n))
        n_val = max(1, int(0.2 * n))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test

        train_subjects = set(shuffled[:n_train])
        val_subjects = set(shuffled[n_train:n_train + n_val])
        test_subjects = set(shuffled[n_train + n_val:])

        # Assign samples to splits
        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        test_paths, test_labels = [], []

        for path, label, subject in zip(all_paths, all_labels, all_subjects):
            if subject in train_subjects:
                train_paths.append(path)
                train_labels.append(label)
            elif subject in val_subjects:
                val_paths.append(path)
                val_labels.append(label)
            elif subject in test_subjects:
                test_paths.append(path)
                test_labels.append(label)

        # Store for dataloaders and get_test_sample
        self._train_paths, self._train_labels = train_paths, train_labels
        self._val_paths, self._val_labels = val_paths, val_labels
        self._test_paths, self._test_labels = test_paths, test_labels
        self._datasets_prepared = True

        print(f"\n✅ Subject-wise split (seed={self.seed}):")
        print(f"   Train: {len(train_subjects)} subjects → {len(train_paths)} images")
        print(f"   Val:   {len(val_subjects)} subjects → {len(val_paths)} images")
        print(f"   Test:  {len(test_subjects)} subjects → {len(test_paths)} images")

        # Create datasets if needed
        if stage == "fit" or stage is None:
            self.train_dataset = ImagePathDataset(train_paths, train_labels, self.train_transform)
            self.val_dataset = ImagePathDataset(val_paths, val_labels, self.test_transform)
        if stage == "test" or stage is None:
            self.test_dataset = ImagePathDataset(test_paths, test_labels, self.test_transform)

    def get_test_sample(self, idx: int):
        """Get a single test sample (image tensor, label) by index."""
        if not hasattr(self, '_test_paths'):
            self.setup(stage='test')
        if idx < 0 or idx >= len(self._test_paths):
            raise IndexError(f"Test index {idx} out of range")
        image = Image.open(self._test_paths[idx]).convert('RGB')
        image = self.test_transform(image)
        return image, self._test_labels[idx]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )