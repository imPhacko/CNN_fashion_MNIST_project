from tqdm import tqdm
import torch

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run the training on
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0 # Total error across all batches
    correct = 0 # Number of correct predictions
    total = 0 # Total number of predictions
    
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device) # Move our images and labels to GPU
        
        optimizer.zero_grad() # Reset gradients to zero
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        running_loss += loss.item() # Update running loss
        _, predicted = outputs.max(1) # Get predicted class
        total += labels.size(0) # Update total number of predictions
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run the evaluation on
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Not training don't need gradients
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device) # Move our images and labels to GPU
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            
            running_loss += loss.item() # Update running loss
            _, predicted = outputs.max(1) # Get predicted class
            total += labels.size(0) # Update total number of predictions
            correct += predicted.eq(labels).sum().item() # Update number of correct predictions

    # Calculate final test performance
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total  # Convert to percentage
    
    return avg_loss, accuracy
