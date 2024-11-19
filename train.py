import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Progress bar for loops
from model import FashionCNN
from data_loader import get_data_loaders
import utils
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Main training function that handles:
    1. Model initialization
    2. Training loop
    3. Evaluation
    4. Visualization
    5. Model saving
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda == 'GPU'
    print(f"Using device: {device}")
    
    model = FashionCNN().to(device) # Initialize model and move to GPU
    train_loader, test_loader = get_data_loaders()
    
    criterion = nn.CrossEntropyLoss() # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Parameters to optimize and learning rate
    # Learning rate scheduler to lower lr when it plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2) # minimize loss, patience of 2 epochs before reducing lr
    
    # Lists to store metrics
    train_losses = [] # Training loss per epoch
    test_losses = [] # Test loss per epoch
    train_accs = [] # Training accuracy per epoch
    test_accs = [] # Test accuracy per epoch
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = utils.train_epoch(
            model, train_loader, criterion, optimizer, device)
        # Evaluate on test set
        test_loss, test_acc = utils.evaluate(
            model, test_loader, criterion, device)
        
        # Save metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        scheduler.step(test_loss) # Adjust learning rate based on test loss
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Save model and model weights
    torch.save(model.state_dict(), 'fashion_cnn.pth')
    print("\nTraining completed! Model saved as 'fashion_cnn.pth'") # Model saved as 'fashion_cnn.pth'
    print(f"Final test accuracy: {test_acc:.2f}%") # Final test accuracy

if __name__ == '__main__':
    main()