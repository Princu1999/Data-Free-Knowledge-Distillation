
from torchvision import datasets, transforms
import torch, numpy as np

def load_data_cifar100(data_root: str, batch_size: int, test_split: float = 0.4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
    total = len(test_dataset)
    split_size = int(total * test_split)
    rng = np.random.default_rng(42)
    indices = rng.choice(range(total), split_size, replace=False)
    subset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=True)
    return test_loader
