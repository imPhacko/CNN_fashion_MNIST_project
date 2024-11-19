import torch
import torch.nn as nn

class FashionCNN(nn.Module):
    """
    Convolutional Neural Network for Fashion MNIST classification.
    
    Architecture:
    - 3 Convolutional blocks with increasing channels (32 -> 64 -> 128)
    - Each block includes BatchNorm, ReLU, MaxPool, and Dropout
    - Fully connected layers with dropout for final classification
    
    Input: (batch_size, 1, 28, 28) - Single channel 28x28 images
    Output: (batch_size, 10) - Logits for 10 fashion classes
    """
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            # First conv layer: (1, 28, 28) -> (32, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # More explicit inputs for first example
            nn.BatchNorm2d(32),    # Normalize activations for stable training
            nn.ReLU(),             # Rectified Linear Unit activation function
            # Second conv layer: (32, 28, 28) -> (32, 28, 28)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),       # Reduce spatial dimensions: (32, 28, 28) -> (32, 14, 14)
            nn.Dropout(0.25)       # Prevent overfitting
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            # Increase channels: (32, 14, 14) -> (64, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),       # (64, 14, 14) -> (64, 7, 7)
            nn.Dropout(0.25)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            # Further increase channels: (64, 7, 7) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),       # (128, 7, 7) -> (128, 3, 3)
            nn.Dropout(0.25)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),          # Flatten: (128, 3, 3) -> (1152)
            # First FC layer: 1152 -> 512
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),       # Higher dropout for FC layers
            # Output layer: 512 -> 10 classes
            nn.Linear(512, 10)     # No activation (used with CrossEntropyLoss)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        Forward pass defines how data flows through the network layers.
        This is where the prediction happens.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
                - batch_size: number of images
                - 1: number of channels (grayscale)
                - 28, 28: image dimension
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10)
                - 10: probability scores for each class
        
        Note: Comments show tensor shapes throughout the network
        """
        x = self.conv1(x)  # Output: (batch_size, 32, 14, 14)
        x = self.conv2(x)  # Output: (batch_size, 64, 7, 7)
        x = self.conv3(x)  # Output: (batch_size, 128, 3, 3)
        x = self.fc(x)     # Output: (batch_size, 10)
        return x

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization, they are created in the convolutional layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            tuple: Feature maps after each convolutional block
                - features1: (batch_size, 32, 14, 14)
                - features2: (batch_size, 64, 7, 7)
                - features3: (batch_size, 128, 3, 3)
        """
        features1 = self.conv1(x)
        features2 = self.conv2(features1)
        features3 = self.conv3(features2)
        return features1, features2, features3

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
        
    Note: Uses generator expression with sum() for memory efficiency
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test the model architecture
if __name__ == "__main__":
    # Create a sample input
    batch_size = 4
    sample_input = torch.randn(batch_size, 1, 28, 28)
    
    # Initialize the model
    model = FashionCNN()
    
    # Print model summary
    print("\nFashion CNN Architecture:")
    print(model)
    
    # Test forward pass
    output = model(sample_input)
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print parameter count
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    # Test feature map extraction
    features1, features2, features3 = model.get_feature_maps(sample_input)
    print("\nFeature map shapes:")
    print(f"First conv block: {features1.shape}")
    print(f"Second conv block: {features2.shape}")
    print(f"Third conv block: {features3.shape}")