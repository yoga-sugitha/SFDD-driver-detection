import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HowDriveDataset(Dataset):
    def __init__(self, root_dir, subjects, transform=None):
        """
        Args:
            root_dir (str): Path to '/kaggle/input/howdir-safedriving-benchmarking-sample/HowDir'
            subjects (list): List of subject names to include (e.g., ['zakaria', 'hajar'])
            transform (callable, optional): Optional transform to apply to images
        """
        self.root_dir = root_dir
        self.subjects = subjects
        self.transform = transform
        self.samples = []  # list of (image_path, class_label)
        self.class_to_idx = self._get_class_to_idx()

        for subject in subjects:
            subject_path = os.path.join(root_dir, subject)
            for class_dir in sorted(os.listdir(subject_path)):
                if not class_dir.startswith('c') or not class_dir[1:].isdigit():
                    continue
                class_idx = int(class_dir[1:])  # 'c0' → 0, 'c9' → 9
                class_path = os.path.join(subject_path, class_dir)
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_path, img_name), class_idx))

    def _get_class_to_idx(self):
        # c0 to c9 → 0 to 9
        return {f'c{i}': i for i in range(10)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label