import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import json
import os
import h5py
from pathlib import Path
from model import EnergyBasedModel

class DebugLogger:
    """Utility class for logging debug information during training"""
    
    def __init__(self, enabled=False, log_dir='train_debug_logs'):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        if enabled:
            self.log_dir.mkdir(exist_ok=True)
        self.logs = {}
        self.epoch = 0
        self.batch = 0
    
    def enable(self):
        """Enable debug logging"""
        self.enabled = True
        self.log_dir.mkdir(exist_ok=True)
        print(f"Training debug logging enabled. Logs will be saved to {self.log_dir.absolute()}")
    
    def disable(self):
        """Disable debug logging"""
        self.enabled = False
        print("Training debug logging disabled")
    
    def log(self, key, value, step=None):
        """Log a value"""
        if not self.enabled:
            return
            
        if step is None:
            step = self.epoch
            
        if key not in self.logs:
            self.logs[key] = {}
            
        # Convert numpy/tensor values to Python types
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            
        if isinstance(value, np.ndarray):
            if value.size <= 10:
                value = value.tolist()
            else:
                value = {
                    'mean': float(np.mean(value)),
                    'std': float(np.std(value)),
                    'min': float(np.min(value)),
                    'max': float(np.max(value)),
                    'shape': list(value.shape)
                }
        
        self.logs[key][step] = value
    
    def start_epoch(self, epoch):
        """Start a new epoch"""
        self.epoch = epoch
        self.batch = 0
        if self.enabled:
            print(f"\n====== Debug: Starting Epoch {epoch} ======")
    
    def next_batch(self):
        """Move to next batch"""
        self.batch += 1
        
    def log_batch(self, data, target_onehot, output, loss):
        """Log batch information"""
        if not self.enabled:
            return
            
        if self.batch % 10 == 0:  # Only log every 10 batches to avoid too much output
            print(f"\n--- Debug: Epoch {self.epoch}, Batch {self.batch} ---")
            print(f"Input shape: {data.shape}, Batch size: {data.shape[0]}")
            print(f"Output shape: {output.shape}")
            print(f"Loss: {loss:.6f}")
            
            self.log(f"batch_{self.epoch}_{self.batch}/input_stats", {
                'mean': float(data.mean().item()),
                'std': float(data.std().item()),
            })
            self.log(f"batch_{self.epoch}_{self.batch}/output_stats", {
                'mean': float(output.mean().item()),
                'std': float(output.std().item()),
            })
            self.log(f"batch_{self.epoch}_{self.batch}/loss", loss)
    
    def log_epoch_results(self, train_loss, val_loss, train_acc, val_acc, lr):
        """Log epoch results"""
        if not self.enabled:
            return
            
        print(f"\n--- Debug: Epoch {self.epoch} Results ---")
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}")
        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}")
        print(f"Learning Rate: {lr:.6f}")
        
        self.log(f"epoch_{self.epoch}/train_loss", train_loss)
        self.log(f"epoch_{self.epoch}/val_loss", val_loss)
        self.log(f"epoch_{self.epoch}/train_acc", train_acc)
        self.log(f"epoch_{self.epoch}/val_acc", val_acc)
        self.log(f"epoch_{self.epoch}/lr", lr)
    
    def log_model_updates(self, model):
        """Log model weight and bias updates"""
        if not self.enabled:
            return
            
        for i, w in enumerate(model.weights):
            self.log(f"model/weight_{i}", {
                'mean': float(w.mean().item()),
                'std': float(w.std().item()),
                'min': float(w.min().item()),
                'max': float(w.max().item()),
            })
        
        for i, b in enumerate(model.biases):
            self.log(f"model/bias_{i}", {
                'mean': float(b.mean().item()),
                'std': float(b.std().item()),
                'min': float(b.min().item()),
                'max': float(b.max().item()),
            })
    
    def save(self):
        """Save logs to file"""
        if not self.enabled or not self.logs:
            return
            
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"train_debug_{timestamp}_epoch_{self.epoch}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
            
        print(f"Debug logs saved to {log_file}")
        
        # Clear logs after saving to avoid memory issues
        self.logs = {}

def generate_circles_dataset(n_samples=1000, noise=0.1, debug=False):
    """
    Generate a dataset of two perpendicular circles in 3D space.
    
    Args:
        n_samples (int): Number of samples per circle
        noise (float): Amount of noise to add to the points
        debug (bool): Whether to print debug information
        
    Returns:
        tuple: (X, y) where X is the 3D points and y is the labels
    """
    if debug:
        print(f"Generating circles dataset with {n_samples} samples per circle, noise={noise}")
    
    # Generate points for first circle (in xy plane)
    theta1 = np.linspace(0, 2*np.pi, n_samples)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)
    z1 = np.zeros_like(x1)
    
    # Generate points for second circle (in xz plane)
    theta2 = np.linspace(0, 2*np.pi, n_samples)
    x2 = np.cos(theta2)
    y2 = np.zeros_like(x2)
    z2 = np.sin(theta2)
    
    # Combine points and add noise
    X = np.vstack([
        np.column_stack([x1, y1, z1]),
        np.column_stack([x2, y2, z2])
    ])
    
    if debug:
        print(f"Clean data shape: {X.shape}")
        print(f"First circle (xy-plane) point example: {X[0]}")
        print(f"Second circle (xz-plane) point example: {X[n_samples]}")
    
    # Add noise
    np.random.seed(42)  # For reproducibility when debugging
    noise_vec = np.random.normal(0, noise, X.shape)
    X += noise_vec
    
    if debug:
        print(f"Noise stats - Mean: {noise_vec.mean():.4f}, Std: {noise_vec.std():.4f}")
        print(f"Noisy data - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    
    # Create labels (0 for first circle, 1 for second circle)
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    
    # Create one-hot encoded labels for output layer format
    y_onehot = torch.zeros(len(y), 2)
    y_onehot.scatter_(1, torch.LongTensor(y).unsqueeze(1), 1.0)
    
    if debug:
        print(f"Labels distribution: Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")
        print(f"One-hot encoded labels shape: {y_onehot.shape}")
    
    return torch.FloatTensor(X), y_onehot, torch.LongTensor(y)

def train_ebm(model, train_loader, val_loader, test_loader=None, epochs=50, 
              scheduler_type='plateau', patience=10, early_stopping=True, debug=False):
    """
    Train an Energy-Based Model using equilibrium propagation with PyTorch features.
    
    Args:
        model (EnergyBasedModel): The energy-based model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        test_loader (DataLoader, optional): DataLoader for test data
        epochs (int): Number of epochs to train for
        scheduler_type (str): Type of learning rate scheduler ('plateau', 'cosine', or None)
        patience (int): Patience for early stopping and plateau scheduler
        early_stopping (bool): Whether to use early stopping
        debug (bool): Whether to enable debug mode
    
    Returns:
        dict: Dictionary containing training history
        model: Trained model
    """
    # Initialize debug logger
    logger = DebugLogger(enabled=debug)
    
    # Enable debug mode in model if requested
    if debug and not model.debug:
        model.enable_debug()
    
    if debug:
        print("\n========== Starting Training ==========")
        print(f"Model Architecture: {[model.layer_sizes]}")
        print(f"Hyperparameters: beta={model.beta}, dt={model.dt}, n_steps={model.n_steps}")
        print(f"Training config: epochs={epochs}, scheduler={scheduler_type}, patience={patience}")
        print(f"Optimizer: {type(model.optimizer).__name__}, "
              f"LR: {model.optimizer.param_groups[0]['lr']}")
        print(f"Train data: {len(train_loader.dataset)} samples, "
              f"Validation data: {len(val_loader.dataset)} samples")
        if test_loader:
            print(f"Test data: {len(test_loader.dataset)} samples")
        print("===================================\n")
    
    # Initialize scheduler based on optimizer in model
    if model.optimizer is None:
        raise ValueError("Model must have an optimizer. Initialize the model with an optimizer.")
    
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(model.optimizer, mode='min', factor=0.5, 
                                     patience=patience//2, verbose=True)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(model.optimizer, T_max=epochs)
    else:
        scheduler = None
    
    device = next(model.parameters()).device
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': [],
        'time_per_epoch': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0

    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Update debug logger
        logger.start_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (data, target_onehot, target_idx) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            logger.next_batch()
            
            data, target_onehot = data.to(device), target_onehot.to(device)
            
            # Debug: Log input data stats
            if debug and batch_idx == 0:
                print(f"\nBatch sample - Input mean: {data.mean().item():.4f}, std: {data.std().item():.4f}")
                print(f"Target distribution: {torch.bincount(target_idx)}")
            
            # Forward pass with target for equilibrium propagation
            output = model(data, target_onehot)
            
            # Compute loss for tracking
            loss = model.cost(output, target_onehot, beta=0, grad=False)
            train_loss += loss
            
            # Debug: Log batch info
            logger.log_batch(data, target_onehot, output, loss)
            
            # Store predictions and targets for accuracy
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            train_preds.extend(predictions)
            train_targets.extend(target_idx.numpy())
            
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        
        # Debug: Log model parameters after training phase
        if debug:
            logger.log_model_updates(model)
            print(f"\nTraining phase complete: Loss={train_loss:.6f}, Accuracy={train_acc:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        if debug:
            print("\nStarting validation phase...")
        
        with torch.no_grad():
            for data, target_onehot, target_idx in val_loader:
                data, target_onehot = data.to(device), target_onehot.to(device)
                
                output = model(data)
                loss = model.cost(output, target_onehot, beta=0, grad=False)
                val_loss += loss
                
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                val_preds.extend(predictions)
                val_targets.extend(target_idx.numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Record current learning rate
        current_lr = model.optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Update scheduler
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Calculate time taken for epoch
        epoch_time = time.time() - epoch_start_time
        history['time_per_epoch'].append(epoch_time)
        
        # Debug: Log epoch results
        logger.log_epoch_results(train_loss, val_loss, train_acc, val_acc, current_lr)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'LR: {current_lr:.6f} | Time: {epoch_time:.2f}s')
        
        # Check for improvement for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
            
            if debug:
                print(f"New best validation loss: {best_val_loss:.6f}")
        else:
            no_improve_count += 1
            
            if debug:
                print(f"No improvement for {no_improve_count} epochs. Best val loss: {best_val_loss:.6f}")
            
        if early_stopping and no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            if debug:
                print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Save debug logs periodically
        if debug and (epoch % 5 == 0 or epoch == epochs - 1):
            logger.save()
    
    # Restore best model
    if best_model_state is not None:
        if debug:
            print("\nRestoring best model from checkpoint")
        model.load_state_dict(best_model_state)
    
    # Final test evaluation if test_loader provided
    if test_loader is not None:
        if debug:
            print("\n========== Final Test Evaluation ==========")
        
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for data, target_onehot, target_idx in test_loader:
                data, target_onehot = data.to(device), target_onehot.to(device)
                
                output = model(data)
                loss = model.cost(output, target_onehot, beta=0, grad=False)
                test_loss += loss
                
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                test_preds.extend(predictions)
                test_targets.extend(target_idx.numpy())
                
        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_targets, test_preds)
        conf_matrix = confusion_matrix(test_targets, test_preds)
        
        print(f'Final Test Results: Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)
        
        if debug:
            print("\nDetailed test metrics:")
            class_names = [f"Circle {i}" for i in range(len(conf_matrix))]
            print_confusion_matrix(conf_matrix, class_names)
            
            # Log test results
            logger.log("test/loss", test_loss)
            logger.log("test/accuracy", test_acc)
            logger.log("test/conf_matrix", conf_matrix.tolist())
            logger.save()
        
        history['test_loss'] = test_loss
        history['test_accuracy'] = test_acc
        history['confusion_matrix'] = conf_matrix
    
    return history, model

def print_confusion_matrix(cm, class_names):
    """
    Print a formatted confusion matrix with class names.
    """
    print("\nConfusion Matrix:")
    # Header
    header = "      "
    for name in class_names:
        header += f"{name:>10}"
    print(header)
    
    # Rows
    for i, name in enumerate(class_names):
        row = f"{name:<6}"
        for j in range(len(class_names)):
            row += f"{cm[i, j]:>10}"
        print(row)
    
    # Calculate metrics
    print("\nPer-Class Metrics:")
    for i, name in enumerate(class_names):
        true_pos = cm[i, i]
        false_pos = cm[:, i].sum() - true_pos
        false_neg = cm[i, :].sum() - true_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

def plot_training_history(history, save_path=None, show=True):
    """
    Plot the training history.
    
    Args:
        history (dict): Training history dictionary
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_dataset(X, y, title="3D Circles Dataset"):
    """
    Visualize the 3D circles dataset.
    
    Args:
        X (torch.Tensor): 3D data points
        y (torch.Tensor): Labels
        title (str): Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to numpy for plotting
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    
    # Plot each class with a different color
    classes = np.unique(y_np)
    for cls in classes:
        idx = y_np == cls
        ax.scatter(X_np[idx, 0], X_np[idx, 1], X_np[idx, 2], 
                  label=f'Circle {int(cls)}', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def run_training_experiment(device='cuda' if torch.cuda.is_available() else 'cpu', debug=False):
    """
    Run a complete training experiment with the 3D circles dataset.
    
    Args:
        device (str): Device to run on ('cuda' or 'cpu')
        debug (bool): Whether to enable debug mode
    
    Returns:
        tuple: (trained_model, history)
    """
    if debug:
        print("\n========== Starting Experiment ==========")
        print(f"Running on device: {device}")
    
    # Generate dataset
    X, y_onehot, y_idx = generate_circles_dataset(n_samples=1000, noise=0.1, debug=debug)
    
    # Visualize the dataset if in debug mode
    if debug:
        visualize_dataset(X, y_idx)
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    test_size = len(X) - train_size - val_size
    
    if debug:
        print(f"\nSplitting data - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Use fixed seed for reproducible splits when debugging
    if debug:
        torch.manual_seed(42)
        
    indices = torch.randperm(len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    X_train, y_train_onehot, y_train_idx = X[train_indices], y_onehot[train_indices], y_idx[train_indices]
    X_val, y_val_onehot, y_val_idx = X[val_indices], y_onehot[val_indices], y_idx[val_indices]
    X_test, y_test_onehot, y_test_idx = X[test_indices], y_onehot[test_indices], y_idx[test_indices]
    
    if debug:
        print(f"\nData split complete:")
        print(f"Train - X: {X_train.shape}, y: {y_train_idx.shape}, Class dist: {torch.bincount(y_train_idx)}")
        print(f"Val   - X: {X_val.shape}, y: {y_val_idx.shape}, Class dist: {torch.bincount(y_val_idx)}")
        print(f"Test  - X: {X_test.shape}, y: {y_test_idx.shape}, Class dist: {torch.bincount(y_test_idx)}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train_onehot, y_train_idx)
    val_dataset = TensorDataset(X_val, y_val_onehot, y_val_idx)
    test_dataset = TensorDataset(X_test, y_test_onehot, y_test_idx)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if debug:
        print(f"\nDataLoaders created with batch size {batch_size}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Set up the device
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Initialize model and optimizer
    learning_rate = 0.001
    model_config = {
        'input_size': 3,  # 3D input
        'hidden_sizes': [64, 32, 2],  # Output size 2 for binary classification
        'beta': 0.1,
        'dt': 0.1,
        'n_steps': 20,
        'debug': debug
    }
    
    if debug:
        print(f"\nCreating model with config: {model_config}")
        print(f"Using learning rate: {learning_rate}")
    
    # Create model
    model = EnergyBasedModel(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        optimizer=None,  # Will set it below
        beta=model_config['beta'],
        dt=model_config['dt'],
        n_steps=model_config['n_steps'],
        debug=model_config['debug']
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer and assign to model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.optimizer = optimizer
    
    if debug:
        print("\nModel created and moved to device")
        print(f"Optimizer: {type(optimizer).__name__}")
        print("\nStarting training...")
    
    # Train model
    history, trained_model = train_ebm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=50,
        scheduler_type='cosine',
        patience=15,
        early_stopping=True,
        debug=debug
    )
    
    # Plot results
    if debug:
        print("\nTraining complete. Plotting results...")
        
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Save and display the plot
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_path = plots_dir / f"training_history_{timestamp}.png"
    plot_training_history(history, save_path=plot_path if debug else None)
    
    if debug:
        print("\n========== Experiment Complete ==========")
    
    return trained_model, history

def extract_hidden_representations(model, data_loader, device):
    """
    Extract hidden layer representations from the model for all samples in the data loader.
    
    Args:
        model (EnergyBasedModel): Trained model
        data_loader (DataLoader): DataLoader containing the dataset
        device (torch.device): Device to run on
        
    Returns:
        dict: Dictionary containing images, labels, and hidden representations
    """
    model.eval()
    hidden_reps = {f'layer_{i}': [] for i in range(len(model.layer_sizes))}
    images = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Extracting representations'):
            data, target = data.to(device), target.to(device)
            
            # Get hidden representations
            states = model.negative(data)
            
            # Store representations
            for i, state in enumerate(states):
                hidden_reps[f'layer_{i}'].append(state.cpu().numpy())
            
            # Store images and labels
            images.append(data.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    # Concatenate all batches
    for layer in hidden_reps:
        hidden_reps[layer] = np.concatenate(hidden_reps[layer], axis=0)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return {
        'images': images,
        'labels': labels,
        'hidden_representations': hidden_reps
    }

def save_to_hdf5(data_dict, filename):
    """
    Save the extracted representations to an HDF5 file.
    
    Args:
        data_dict (dict): Dictionary containing the data to save
        filename (str): Path to save the HDF5 file
    """
    with h5py.File(filename, 'w') as f:
        # Save images
        f.create_dataset('images', data=data_dict['images'])
        
        # Save labels
        f.create_dataset('labels', data=data_dict['labels'])
        
        # Save hidden representations
        hidden_group = f.create_group('hidden_representations')
        for layer_name, data in data_dict['hidden_representations'].items():
            hidden_group.create_dataset(layer_name, data=data)

def run_mnist_experiment(device='cuda' if torch.cuda.is_available() else 'cpu', debug=False):
    """
    Run a complete training experiment with MNIST dataset.
    
    Args:
        device (str): Device to run on ('cuda' or 'cpu')
        debug (bool): Whether to enable debug mode
    
    Returns:
        tuple: (trained_model, history)
    """
    if debug:
        print("\n========== Starting MNIST Experiment ==========")
        print(f"Running on device: {device}")
    
    # Set up data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if debug:
        print(f"\nDataset loaded:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model and optimizer
    model_config = {
        'input_size': 784,  # 28x28 MNIST images
        'hidden_sizes': [512, 256, 128, 10],  # Output size 10 for 10 digits
        'beta': 0.1,
        'dt': 0.1,
        'n_steps': 20,
        'debug': debug
    }
    
    learning_rate = 0.001
    
    if debug:
        print(f"\nCreating model with config: {model_config}")
        print(f"Using learning rate: {learning_rate}")
    
    # Create model
    model = EnergyBasedModel(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        optimizer=None,  # Will set it below
        beta=model_config['beta'],
        dt=model_config['dt'],
        n_steps=model_config['n_steps'],
        debug=model_config['debug']
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer and assign to model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.optimizer = optimizer
    
    if debug:
        print("\nModel created and moved to device")
        print(f"Optimizer: {type(optimizer).__name__}")
        print("\nStarting training...")
    
    # Train model
    history, trained_model = train_ebm(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for simplicity
        test_loader=None,
        epochs=10,
        scheduler_type='cosine',
        patience=5,
        early_stopping=True,
        debug=debug
    )
    
    # Extract and save hidden representations
    if debug:
        print("\nExtracting hidden representations...")
    
    # Create a new data loader for the entire dataset
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    # Extract representations
    representations = extract_hidden_representations(trained_model, full_loader, device)
    
    # Save to HDF5
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    hdf5_path = f'mnist_representations_{timestamp}.h5'
    save_to_hdf5(representations, hdf5_path)
    
    if debug:
        print(f"\nHidden representations saved to {hdf5_path}")
        print("\n========== Experiment Complete ==========")
    
    return trained_model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EBM training experiment on MNIST')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = run_mnist_experiment(device=device, debug=args.debug)
