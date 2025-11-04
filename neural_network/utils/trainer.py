"""
Training utilities for the neural network.

This module provides the Trainer class for managing the training loop,
including loss computation, optimization, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json


class Trainer:
    """
    Trainer class for managing the neural network training process.
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader, optional): DataLoader for validation data
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cpu' or 'cuda')
        checkpoint_dir (str): Directory to save model checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function: Mean Squared Error for audio regression
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam with default parameters
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                # Move data to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """
        Validate the model on the validation set.
        
        Returns:
            float: Average validation loss
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch in pbar:
                    # Move data to device
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    # Forward pass
                    outputs, _ = self.model(inputs)
                    
                    # Compute loss
                    loss = self.criterion(outputs, targets)
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                print(f"Validation Loss: {val_loss:.6f}")
                
                # Update learning rate based on validation loss
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"Saved best model with validation loss: {val_loss:.6f}")
            else:
                # Update learning rate based on training loss if no validation
                self.scheduler.step(train_loss)
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save final model
        self.save_checkpoint(num_epochs - 1, is_best=False, name='final_model.pt')
        self.save_history()
        print("\nTraining completed!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, name: str = None):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
            name (str, optional): Custom name for the checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        if name is None:
            name = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch + 1}.pt'
        
        checkpoint_path = self.checkpoint_dir / name
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
