"""
Training utilities for multimodal learning models.

This module provides training loops, evaluation functions, and utilities
for training multimodal neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


class Trainer:
    """Trainer class for multimodal learning models."""
    
    def __init__(self, model, device='cuda', learning_rate=0.001, criterion=None):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            criterion: Loss function (defaults to CrossEntropyLoss if None)
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in tqdm(dataloader, desc="Training"):
            # Move data to device
            if isinstance(batch_data, list):
                batch_data = [d.to(self.device) for d in batch_data]
            else:
                batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader, desc="Validation"):
                # Move data to device
                if isinstance(batch_data, list):
                    batch_data = [d.to(self.device) for d in batch_data]
                else:
                    batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='../checkpoints'):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save model checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        return self.history


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc="Testing"):
            if isinstance(batch_data, list):
                batch_data = [d.to(device) for d in batch_data]
            else:
                batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
    
    accuracy = 100. * correct / total
    return {'accuracy': accuracy, 'correct': correct, 'total': total}
