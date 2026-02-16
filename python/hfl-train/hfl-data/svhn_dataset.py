import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SVHNDataset(Dataset):
    def __init__(self, file_path, transform=None, row_ids: list[int] = []):
        """
        Args:
            mat_file: Path to extracted images directory
            transform: Optional torchvision transforms
            row_ids: list of indices to load (e.g., [0, 1, 2, ..., 4999])
        """
        # Load .mat file
        data = sio.loadmat(file_path)

        # Extract only the specified rows
        # Original shape: (32, 32, 3, num_total_samples)
        self.images = data["X"][:, :, :, row_ids]  # (32, 32, 3, len(row_indices))
        self.labels = data["y"].flatten()[row_ids]  # (len(row_indices),)

        # Convert label 10 to 0
        self.labels[self.labels == 10] = 0

        self.transform = transform

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


def get_svhn_transforms():
    """Get standard SVHN transforms"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
