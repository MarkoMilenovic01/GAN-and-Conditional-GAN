import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size=128, root="data"):
    """
    Prepares the MNIST training and test dataloaders.
    
    - Loads MNIST
    - Converts images to tensors
    - Scales images to [0, 1]
    - Outputs DataLoaders with shape (N, 1, 28, 28)
    """

    # Transform: convert to tensor (automatically scales to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=root, train=False, download=True, transform=transform)


    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Example: run this file directly to test
if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()

    # Check one batch
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)      # → torch.Size([128, 1, 28, 28])
    print("Pixel range:", (images.min().item(), images.max().item()))
