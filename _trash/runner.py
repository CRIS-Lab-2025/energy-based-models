import os
import json
import logging
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from detection_model import select_det_model
from main_model import select_main_model
from dataloader_refusal import dataset

def create_dir(path):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def results(config):
    """
    Sets up directories and files for storing training results.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary with model paths.
    """
    # Construct the result path
    result_path = os.path.join(
        config['result_path'],
        f"{config['detection_model']}_{config['main_model']}_{config['dataset']}_{config['tag']}"
    )

    # Create result directory if it doesn't exist
    create_dir(result_path)

    # Set up logging to a file
    logging.basicConfig(
        filename=os.path.join(result_path, 'results.log'),
        level=logging.INFO
    )

    # Create a directory to save the detection model weights
    model_path = os.path.join(result_path, 'det_model_weights')
    create_dir(model_path)

    # Save the configuration to a JSON file
    with open(os.path.join(result_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Set the paths for the best model and model directory in the config
    config['best_model_path'] = os.path.join(result_path, 'best_model.pth')
    config['model_path'] = model_path

    return config

def set_config(args):
    """
    Loads the configuration file and updates it with command-line arguments.

    Args:
        args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        dict: Updated configuration dictionary.
    """
    # Load configuration from the JSON file
    with open(args.config) as f:
        config = json.load(f)

    # Override config with command-line arguments if provided
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            config[arg] = value

    # Disable wandb and logging if in debug mode
    if config.get('debug', False):
        config.update({'wandb': False, 'log': False})

    # Initialize wandb if enabled
    if config['wandb']:
        wandb.init(project='bad_content_detection', config=config)
        # Assign tags if specified
        if config['tag']:
            wandb.run.tags = [config['tag']]

    # Set up results directory and files
    config = results(config)

    return config


def train(config):
    """
    Trains the detection model using the specified configuration.

    Args:
        config (dict): Configuration dictionary containing hyperparameters, paths, and other settings.
    """
    # Determine device to use (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset from CSV file
    data = pd.read_csv(config['dataset_path'])

    # Split data into training and testing sets (80% train, 20% test)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Initialize the data loader for training data
    train_dataloader = DataLoader(
        dataset(train_data, config),
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Initialize the detection model and set it to training mode
    detection_model = select_det_model(config['detection_model'], config).to(device)
    detection_model.train()

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        detection_model.parameters(),
        lr=config['lr'],
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0  # Initialize best accuracy

    try:
        # Training loop
        for epoch in tqdm(range(config['epochs']), desc='Epochs'):
            total_correct = 0
            total_samples = 0
            print(f"Epoch: {epoch}")

            # Iterate over batches
            for i, data_batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training Batches'):
                # Unpack batch data
                response, safety_class, token_hidden_states, prompt_hidden_states = data_batch

                optimizer.zero_grad()

                # Extract the relevant hidden state
                state = token_hidden_states[:, config['layer'] - 1, config['token'] - 1, :]

                # Move data to device
                state = state.to(device)
                labels = safety_class.to(device).long()

                # Forward pass
                output = detection_model(state)

                # Compute loss
                loss = criterion(output, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Calculate batch accuracy
                _, predicted = torch.max(output.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            # Calculate epoch accuracy
            epoch_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0.0

            # Log metrics to wandb
            if config['wandb']:
                wandb.log({
                    "Epoch": epoch,
                    "Loss": loss.item(),
                    "Accuracy": epoch_accuracy
                })

            # Log epoch results
            if config['log']:
                logging.info(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {epoch_accuracy:.2f}%")

            # Save model checkpoint at specified intervals
            if epoch % config['checkpoint_epoch'] == 0:
                checkpoint_path = os.path.join(config['model_path'], f'epoch_{epoch}.pth')
                # torch.save(detection_model.state_dict(), checkpoint_path)

            # Save best model based on accuracy
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                # torch.save(detection_model.state_dict(), config['best_model_path'])

        # Finish wandb run
        if config['wandb']:
            wandb.finish()

        # Log best accuracy
        if config['log']:
            logging.info(f"Best accuracy: {best_accuracy:.2f}%")

    except Exception as e:
        # Handle exceptions
        error_message = str(e)
        if config['wandb']:
            wandb.alert(title="Training Error", text=error_message)
        if config['log']:
            logging.error(error_message)
        raise e

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train the detection model.')
    parser.add_argument('--config', '-c', type=str, default='config/config.json', help='Path to the config file.')
    parser.add_argument('--detection_model', '-dm', type=str, default='MLP', help='Type of detection model to use.')
    parser.add_argument('--main_model', '-mm', type=str, default='Llama2', help='Type of main model to use.')
    parser.add_argument('--dataset', '-ds', type=str, default='CIFAR10', help='Dataset to use for training.')
    parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode.')
    parser.add_argument('--wandb', type=bool, default=True, help='Enable Weights & Biases logging.')
    parser.add_argument('--log', type=bool, default=True, help='Enable logging to file.')
    parser.add_argument('--tag', type=str, default='', help='Tag for the wandb run.')
    parser.add_argument('-l', '--layer', type=int, default=0, help='Layer index for extracting hidden states.')
    parser.add_argument('-t', '--token', type=int, default=0, help='Token index for extracting hidden states.')

    # Parse arguments
    args = parser.parse_args()

    # Set up configuration
    config = set_config(args)

    # Start training
    train(config)

# =============================================================================
# Example usage (for testing purposes)
# =============================================================================
if __name__ == '__main__':
    # Example dummy implementations for network, updater, and differentiator.
    class DummyNetwork:
        def __init__(self):
            # For demonstration, create dummy weight and bias tensors.
            self.W = torch.nn.Parameter(torch.randn(5, 5))
            self.B = torch.nn.Parameter(torch.randn(5))
        def set_input(self, x, reset=False):
            # In a real network, you'd set the input and maybe update state.
            # Here, we just return our weight and a dummy state S (based on x).
            S = x.mean(dim=1)  # dummy state computation
            return self.W, S
        def state_dict(self):
            # Return a dictionary of parameters.
            return {"W": self.W, "B": self.B}
    
    class DummyUpdater:
        def compute_equilibrium(self, S, W, B):
            # Dummy equilibrium computation (for example, an identity operation).
            return W, S

    class DummyDifferentiator:
        def compute_gradient(self, S, W, B, target):
            # Dummy gradient computation.
            # Here we simply create gradients proportional to S.
            weight_grads = torch.ones_like(W) * S.mean()
            bias_grads = torch.ones_like(B) * S.mean()
            return weight_grads, bias_grads

    # Create a dummy dataloader.
    from torch.utils.data import DataLoader, TensorDataset
    x_dummy = torch.randn(100, 10)   # 100 samples, 10 features
    target_dummy = torch.randint(0, 2, (100,))  # Dummy target labels
    dataset_dummy = TensorDataset(x_dummy, target_dummy)
    train_loader = DataLoader(dataset_dummy, batch_size=16, shuffle=True)

    # Create dummy config.
    config = {
        "result_path": "./results",
        "model_path": "./results/model_weights",
        "best_model_path": "./results/best_model.pth",
        "checkpoint_epoch": 2,
        "epochs": 5,
        "wandb": False,    # Set to True to enable wandb tracking.
        "log": True,
        "tag": "test_run",
    }

    # Initialize dummy components.
    network = DummyNetwork()
    updater = DummyUpdater()
    differentiator = DummyDifferentiator()
    # Wrap network parameters with an optimizer.
    optimizer = torch.optim.SGD([network.W, network.B], lr=0.01)

    # Instantiate Runner.
    runner = Runner(config, network, train_loader, differentiator, updater, optimizer)

    # Run training.
    best_metric = runner.run_training()
    print(f"Best metric achieved: {best_metric:.4f}")

    # Run inference.
    outputs = runner.run_inference()
    print(f"Inference produced {len(outputs)} batches of outputs.")
