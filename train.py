import numpy as np
from transformer import Transformer
from utils import create_padding_mask, create_look_ahead_mask
import time
import os
import json

class Trainer:
    def __init__(self, model, learning_rate=0.0001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
    def __init__(self, config):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Configuration object containing model and training parameters
        """
        self.config = config
        self.model = Transformer(config)
        self.optimizer = AdamOptimizer(
            learning_rate=config.learning_rate,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-9
        )
        
        # Initialize learning rate scheduler
        self.lr_scheduler = WarmupScheduler(
            d_model=config.d_model,
            warmup_steps=config.warmup_steps
        )
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def step(self, loss):
        """Perform one optimization step"""
        # Backward pass through the model
        self.model.backward(loss)
        
        # Update parameters using optimizer
        self._update_parameters(self.model.params, self.model.grads)
    
    def _update_parameters(self, params, grads, prefix=''):
        """Recursively update parameters in nested dictionaries"""
        for name, param in params.items():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(param, dict):
                # Recursively handle nested parameters
                self._update_parameters(param, grads[name], full_name)
            else:
                # Update leaf parameters
                param_dict = {full_name: param}
                grad_dict = {full_name: grads[name]}
                updated_params = self.optimizer.update(param_dict, grad_dict)
                params[name] = updated_params[full_name]
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.optimizer.learning_rate
    
    def set_learning_rate(self, lr):
        """Set learning rate"""
        self.learning_rate = lr
        self.optimizer.learning_rate = lr

    def train_step(self, src_batch, tgt_batch, src_mask, tgt_mask):
        """
        Perform a single training step.
        
        Args:
            src_batch: Source sequence batch
            tgt_batch: Target sequence batch
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask
            
        Returns:
            loss: Training loss for the step
        """
        # Forward pass
        predictions = self.model(src_batch, tgt_batch[:, :-1], src_mask, tgt_mask)
        
        # Calculate loss
        loss = self.compute_loss(predictions, tgt_batch[:, 1:])
        
        # Backward pass
        self.step(loss)
        
        return loss

    def compute_loss(self, predictions, targets):
        """
        Compute the loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Target sequences
            
        Returns:
            loss: Computed loss value
        """
        # Reshape predictions and targets for loss computation
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1)
        
        # Compute cross-entropy loss
        loss = -np.mean(np.log(pred_flat[np.arange(len(target_flat)), target_flat] + 1e-9))
        return loss

    def train(self, train_dataset, val_dataset, num_epochs):
        """
        Train the model for specified number of epochs.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs to train
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            train_losses = []
            for batch_idx, (src_batch, tgt_batch) in enumerate(train_dataset):
                # Create masks
                src_mask = create_padding_mask(src_batch)
                tgt_mask = create_look_ahead_mask(tgt_batch.shape[1])
                
                # Update learning rate
                lr = self.lr_scheduler.get_lr(self.optimizer.t)
                self.set_learning_rate(lr)
                
                # Training step
                loss = self.train_step(src_batch, tgt_batch, src_mask, tgt_mask)
                train_losses.append(loss)
                
                # Log progress
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}')
            
            # Validation phase
            val_loss = self.validate(val_dataset)
            
            # Update metrics
            self.metrics['train_loss'].append(np.mean(train_losses))
            self.metrics['val_loss'].append(val_loss)
            self.metrics['learning_rates'].append(lr)
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            # Log epoch results
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
            print(f'Training Loss: {np.mean(train_losses):.4f}, Validation Loss: {val_loss:.4f}')
            print(f'Learning Rate: {lr:.6f}\n')
            
            # Save metrics
            self.save_metrics()

    def validate(self, val_dataset):
        """
        Validate the model on the validation dataset.
        
        Args:
            val_dataset: Validation dataset
            
        Returns:
            val_loss: Average validation loss
        """
        val_losses = []
        
        for src_batch, tgt_batch in val_dataset:
            # Create masks
            src_mask = create_padding_mask(src_batch)
            tgt_mask = create_look_ahead_mask(tgt_batch.shape[1])
            
            # Forward pass
            predictions = self.model(src_batch, tgt_batch[:, :-1], src_mask, tgt_mask)
            
            # Compute loss
            loss = self.compute_loss(predictions, tgt_batch[:, 1:])
            val_losses.append(loss)
        
        return np.mean(val_losses)

    def save_checkpoint(self, epoch, val_loss):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.get_state(),
            'optimizer_state': self.optimizer.get_state(),
            'val_loss': val_loss
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.npz')
        np.savez(checkpoint_path, **checkpoint)
        print(f'Checkpoint saved: {checkpoint_path}')

    def save_metrics(self):
        """Save training metrics to a JSON file."""
        metrics_path = os.path.join(self.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)

class AdamOptimizer:
    """
    Adam optimizer implementation.
    """
    def __init__(self, learning_rate=0.0001, beta1=0.9, beta2=0.98, epsilon=1e-9):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step

    def update(self, params, grads):
        """Update parameters using Adam optimization"""
        if not self.m:  # Initialize moments if empty
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        
        # Compute bias-corrected learning rate
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        for key in params:
            if key in grads:  # Only update if we have gradients
                # Update moments
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])
                
                # Update parameters
                params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.epsilon)
        
        return params

    def get_state(self):
        """Get optimizer state for checkpointing."""
        return {
            't': self.t,
            'm': self.m,
            'v': self.v
        }

class WarmupScheduler:
    """
    Learning rate scheduler with warmup as described in the paper.
    """
    def __init__(self, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_lr(self, step):
        """
        Get learning rate for current step.
        
        Args:
            step: Current training step
            
        Returns:
            learning_rate: Learning rate for the step
        """
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

if __name__ == '__main__':
    # Example usage
    from config import Config
    
    # Load configuration
    config = Config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load datasets (implement your dataset loading logic)
    train_dataset = None  # Replace with your dataset
    val_dataset = None    # Replace with your dataset
    
    # Train model
    trainer.train(train_dataset, val_dataset, num_epochs=config.num_epochs) 