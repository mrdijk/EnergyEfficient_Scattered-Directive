import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SVHN_Model(nn.Module):
    def __init__(self):
        super(SVHN_Model, self).__init__()
        self.fc3 = nn.Linear(3072, 512)  # 32x32x3
        self.fc5 = nn.Linear(512, 10)
        self.size = float(6.1)  # Mb

    def forward(self, xb):
        out = xb.view(-1, 3072)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc5(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size


class SVHN_Dataset(Dataset):
    def __init__(
        self,
        file_path,
        transform=None,
        row_ids: list[int] = [],
        num_classes=10,
        random_seed=42,
    ):
        """
        Args:
            mat_file: Path to extracted images directory
            transform: Optional torchvision transforms
            row_ids: list of indices to load (e.g., [0, 1, 2, ..., 4999])
            num_classes: Number of classes to use -> i.i.d. level (3, 5, 7, 10)
        """
        # Load .mat file
        data = sio.loadmat(file_path)

        # Extract only the specified rows
        # Original shape: (32, 32, 3, num_total_samples)
        self.images = data["X"][:, :, :, row_ids]  # (32, 32, 3, len(row_indices))
        self.labels = data["y"].flatten()[row_ids]  # (len(row_indices),)

        # Convert label 10 to 0
        self.labels[self.labels == 10] = 0

        # Randomly select classes if num_classes < 10
        if num_classes < 10:
            np.random.seed(random_seed)
            all_classes = np.arange(10)
            selected_classes = np.sort(
                np.random.choice(all_classes, num_classes, replace=False)
            )

            mask = np.isin(self.labels, selected_classes)
            valid_indices = np.where(mask)[0]
            self.images = self.images[:, :, :, valid_indices]
            self.labels = self.labels[valid_indices]

            # No remapping — labels stay as original digits 0-9
            self.class_mapping = {i: i for i in range(10)}
            self.reverse_mapping = {i: i for i in range(10)}
            self.selected_classes = selected_classes

        self.transform = transform
        self.num_classes = num_classes
        self.random_seed = random_seed

    def __len__(self):
        return self.images.shape[3]

    def __getitem__(self, idx):
        # Get image: (32, 32, 3, idx) -> (32, 32, 3)
        image = self.images[:, :, :, idx]
        label = self.labels[idx]

        # Convert to PIL Image if using transforms
        if self.transform:
            image = Image.fromarray(image.astype("uint8"))
            image = self.transform(image)
        else:
            # Convert to tensor: (32, 32, 3) -> (3, 32, 32)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def get_num_classes(self):
        return self.num_classes

    def get_selected_classes(self):
        """Return the original digit labels that were selected."""
        return self.selected_classes

    def get_class_mapping(self):
        """Return mapping from original labels to model labels."""
        return self.class_mapping

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_original_class_distribution(self):
        """Get distribution using original digit labels."""
        distribution = self.get_class_distribution()
        return {self.reverse_mapping[k]: v for k, v in distribution.items()}


def get_svhn_transforms():
    """Get standard SVHN transforms"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
