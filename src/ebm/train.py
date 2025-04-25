import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from ebm.dataset import get_dataset
from ebm.model import EnergyBasedModel
from ebm.util.model_extract import extract_hidden_representations, save_to_hdf5


def train_ebm(model, train_loader, val_loader=None, test_loader=None, epochs=10, 
              scheduler_type=None, patience=5, early_stopping=False, debug=False):
    """
    Train an Energy-Based Model using equilibrium propagation.
    
    Args:
        model (EnergyBasedModel): The energy-based model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader, optional): DataLoader for validation data
        test_loader (DataLoader, optional): DataLoader for test data
        epochs (int): Number of epochs to train for
        scheduler_type (str): Type of learning rate scheduler ('plateau', 'cosine', or None)
        patience (int): Patience for early stopping and plateau scheduler
        early_stopping (bool): Whether to use early stopping
        debug (bool): Whether to enable debug mode for the model
    
    Returns:
        dict: Dictionary containing training history
        model: Trained model
    """
    # Initialize scheduler based on optimizer in model
    if model.optimizer is None:
        raise ValueError("Model must have an optimizer. Initialize the model with an optimizer.")
    
    if scheduler_type == 'plateau' and val_loader is not None:
        scheduler = ReduceLROnPlateau(model.optimizer, mode='min', factor=0.5, 
                                     patience=patience//2, verbose=True)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(model.optimizer, T_max=epochs)
    else:
        scheduler = None
    
    device = next(model.parameters()).device
    if debug:
        model.debug = True
    
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
        
        # Training phase
        model.train()
        train_preds = []
        train_targets = []
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            # Handle different formats of data
            if len(batch_data) == 2:
                data, target = batch_data
                data, target = data.to(device), target.to(device)
                
                # Forward pass with target for equilibrium propagation
                output = model(data, target)
                
                # Store predictions and targets for accuracy
                predictions = torch.argmax(output, dim=1) if output.dim() > 1 else output
                train_preds.append(predictions)
                train_targets.append(target)
        
        # Calculate accuracy
        if train_preds and train_targets:
            train_preds = torch.cat(train_preds)
            train_targets = torch.cat(train_targets)
            if train_preds.dim() > 1:
                train_preds = train_preds.argmax(dim=1)
            train_acc = (train_preds == train_targets).float().mean().item()
            history['train_accuracy'].append(train_acc)
            
            current_lr = model.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Calculate time taken for epoch
            epoch_time = time.time() - epoch_start_time
            history['time_per_epoch'].append(epoch_time)
            
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Acc: {train_acc:.4f} | '
                  f'LR: {current_lr:.6f} | Time: {epoch_time:.2f}s')
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    
                    output = model(data)
                    val_preds.append(output)
                    val_targets.append(target)
            
            if val_preds and val_targets:
                val_preds = torch.cat(val_preds)
                val_targets = torch.cat(val_targets)
                if val_preds.dim() > 1:
                    val_preds = val_preds.argmax(dim=1)
                val_acc = (val_preds == val_targets).float().mean().item()
                history['val_accuracy'].append(val_acc)
                
                print(f'Val Acc: {val_acc:.4f}')
                
                # Update scheduler
                if scheduler is not None and scheduler_type == 'plateau':
                    scheduler.step(1 - val_acc)  # Use negative accuracy as loss
            
        elif scheduler is not None and scheduler_type == 'cosine':
            scheduler.step()
        
        # Early stopping check
        if val_loader is not None and early_stopping:
            val_acc = history['val_accuracy'][-1]
            val_loss = 1 - val_acc  # Use negative accuracy as loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Restore best model
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation if test_loader provided
    if test_loader is not None:
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                test_preds.append(output)
                test_targets.append(target)
        
        if test_preds and test_targets:
            test_preds = torch.cat(test_preds)
            test_targets = torch.cat(test_targets)
            if test_preds.dim() > 1:
                test_preds = test_preds.argmax(dim=1)
            test_acc = (test_preds == test_targets).float().mean().item()
            
            print(f'Final Test Accuracy: {test_acc:.4f}')
            history['test_accuracy'] = test_acc
    
    return history, model


def train(config, debug=False):
    dataset = config.get('dataset', '3d-circles')
    train_loader, test_loader = get_dataset(dataset)
    model = EnergyBasedModel(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        beta=config['beta'],
        dt=config['dt'],
        optimizer=None,  # Optimizer will be set separately
        n_steps=config['n_steps']
    )
    model.to(config.get('device', 'cpu'))
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    model.optimizer = optimizer
    model.debug = debug
    
    history, model = train_ebm(
        model, 
        train_loader, 
        test_loader=test_loader,  # Use test loader as validation for simplicity
        epochs=config.get('epochs', 10),
        scheduler_type=config.get('scheduler_type', None),
        patience=config.get('patience', 5),
        early_stopping=config.get('early_stopping', False),
        debug=debug
    )
    return model, history

if __name__ == "__main__":
    config = {
    'input_size': 3,  # 3D input
    'hidden_sizes': [128, 2],  # Hidden layer sizes
    'beta': 0.2,  # Temperature parameter
    'dt': 0.05,   # Step size
    'n_steps': 20 # Number of steps
}
    dataset = "3d-circles"
    model, history = train(config)
