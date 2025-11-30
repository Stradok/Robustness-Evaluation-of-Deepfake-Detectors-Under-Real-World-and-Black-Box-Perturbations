import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config

def get_transforms():
    """
    Returns preprocessing transforms.
    All images are resized to 299x299 (Xception size).
    """
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])


def create_dataloaders():
    """
    Creates PyTorch DataLoaders for train, val, and test sets.
    Uses paths from Config.
    """

    transform = get_transforms()

    # ---------------------------
    # Load datasets
    # ---------------------------
    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(Config.VAL_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(Config.TEST_DIR, transform=transform)

    # ---------------------------
    # Dataloaders
    # ---------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader
