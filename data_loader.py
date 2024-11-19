import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    """
    Create data loaders for training and testing Fashion MNIST dataset.
    
    Args:
        batch_size (int): Number of images per batch (default: 64)
            - Larger batch size = faster training but more memory
            - Smaller batch size = better generalization but slower training
            
    Returns:
        tuple: (train_loader, test_loader)
            - train_loader: DataLoader for training data with augmentation
            - test_loader: DataLoader for test data with only normalization
    """
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    # Define training data transformations
    train_transform = transforms.Compose([
        # Random rotation up to 10 degrees
        transforms.RandomRotation(10),
        
        # Random affine transformations
        transforms.RandomAffine(
            degrees=0,              # No additional rotation
            translate=(0.1, 0.1),   # Shift by up to 10% in x and y
            scale=(0.9, 1.1)       # Scale by ±10%
        ),
        
        # Convert PIL Image to tensor (scales pixels to [0-1])
        transforms.ToTensor(),
        
        # Normalize pixel values to [-1, 1]
        # Formula: (x - mean) / std
        transforms.Normalize(
            mean=(0.5,),  # Single channel (grayscale)
            std=(0.5,)    # After this: (x - 0.5) / 0.5 = 2x - 1
        )
    ])
    
    # Define test data transformations (only essential preprocessing)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load training dataset with augmentation
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,        # Data directory
        train=True,           # Training set
        download=True,        # Download if not present
        transform=train_transform
    )
    
    # Load test dataset with basic transforms
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,          # Test set, therefore False
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data for better generalization
        num_workers=2          # Parallel data loading
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # No need to shuffle test data
        num_workers=2
    )
    
    return train_loader, test_loader