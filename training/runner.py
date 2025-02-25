import os
import json
import logging
import torch
from tqdm import tqdm
import wandb

def create_dir(path):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

class Runner:
    def __init__(self, config, network, dataloader, differentiator, updater, optimizer, inference_dataloader=None):
        """
        Initializes the Runner.
        
        Args:
            config (dict): Configuration dictionary with keys such as:
                - result_path, checkpoint_epoch, best_model_path, model_path, epochs, wandb, log, tag, etc.
            network: Your network object, which must implement:
                - set_input(x, reset=False) -> (W, S)
                - an attribute 'B' representing the bias parameters.
                - state_dict() for saving model checkpoints.
            dataloader: The training dataloader yielding tuples (x, target).
            differentiator: Object with a method compute_gradient(S, W, B, target) 
                that returns (weight_grads, bias_grads).
            updater: Object with a method compute_equilibrium(S, W, B) that updates the state.
            optimizer: An already constructed optimizer (e.g. from torch.optim) that updates the network's parameters.
            inference_dataloader (optional): A dataloader to use during inference.
                If not provided, training dataloader will be used.
        """
        self._config = config
        self._network = network
        self._dataloader = dataloader
        self._differentiator = differentiator
        self._updater = updater
        self._optimizer = optimizer
        self._inference_dataloader = inference_dataloader if inference_dataloader is not None else dataloader

        # Set up directories for saving checkpoints/results.
        self._result_path = config.get('result_path', './results')
        create_dir(self._result_path)
        self._model_path = config.get('model_path', os.path.join(self._result_path, 'model_weights'))
        create_dir(self._model_path)
        self._best_model_path = config.get('best_model_path', os.path.join(self._result_path, 'best_model.pth'))
        self._checkpoint_epoch = config.get('checkpoint_epoch', 1)
        self._epochs = config.get('epochs', 10)

        # Initialize wandb if enabled.
        self._use_wandb = config.get('wandb', False)
        if self._use_wandb:
            wandb.init(project=config.get('project_name', 'default_project'), config=config)
            if config.get('tag'):
                wandb.run.tags = [config['tag']]

        # Set up logging if enabled.
        self._log = config.get('log', True)
        if self._log:
            logging.basicConfig(filename=os.path.join(self._result_path, 'results.log'),
                                level=logging.INFO)
        
        # For tracking the best metric (assumed to be higher is better).
        self._best_metric = -float('inf')

    def training_epoch(self):
        """
        Performs one epoch of training.
        For each batch:
            - Sets the network input.
            - Lets the network settle to equilibrium via the updater.
            - Computes gradients with the differentiator.
            - Performs an optimizer step.
        
        Returns:
            The final updated weight and state tensors (W, S) from the last batch.
        """
        for x, target in tqdm(self._dataloader, desc="Training Batches"):
            # Zero gradients for current batch.
            self._optimizer.zero_grad()
            # Set the input and retrieve current parameters.
            S = self._network.set_input(x, reset=False)
            # Assume the network has an attribute B representing the bias matrix.
            W, B = self._network.weight, self._network.bias
            # Let the network settle to equilibrium.
            W, S = self._updater.compute_equilibrium(S, W, B)
            # Compute parameter gradients (assumed to return (weight_grads, bias_grads)).
            weight_grads, bias_grads = self._differentiator.compute_gradient(S, W, B, target)
            # Assign gradients to the parameters.
            W.grad, B.grad = weight_grads, bias_grads
            # Update parameters.
            self._optimizer.step()
        return S, W, B

    def inference_epoch(self):
        """
        Performs one epoch of inference.
        For each batch:
            - Sets the network input.
            - Lets the network settle to equilibrium via the updater.
        
        Returns:
            A list of tuples (W, S) for each batch.
        """
        outputs = []
        for x, _ in tqdm(self._inference_dataloader, desc="Inference Batches"):
            # (Zeroing gradients is optional during inference.)
            self._optimizer.zero_grad()
            S = self._network.set_input(x, reset=False)
            W, B = self._network.weight, self._network.bias
            W, S = self._updater.compute_equilibrium(S, W, B)
            outputs.append((W, S))
        return outputs

    def run_training(self):
        """
        Runs the full training loop over epochs.
        Logs metrics (dummy metric used here—you can replace with your own evaluation),
        saves checkpoints at intervals, and saves the best model.
        """
        for epoch in tqdm(range(self._epochs), desc="Epochs"):
            print(f"Epoch: {epoch}")
            W, S = self.training_epoch()
            self._network.weights = W
            self._network.state = S
            # Compute a dummy metric (for example, the mean of S) to decide on best model.
            # Replace this with your actual evaluation/metric computation.
            metric = S.mean().item()

            # Log metrics to wandb if enabled.
            if self._use_wandb:
                wandb.log({
                    "Epoch": epoch,
                    "Metric": metric,
                })

            # Log to file if enabled.
            if self._log:
                logging.info(f"Epoch: {epoch}, Metric: {metric:.4f}")

            # Save a checkpoint every checkpoint_epoch.
            if epoch % self._checkpoint_epoch == 0:
                checkpoint_path = os.path.join(self._model_path, f'epoch_{epoch}.pth')
                torch.save(self._network.state_dict(), checkpoint_path)

            # Save the best model if the metric improves.
            if metric > self._best_metric:
                self._best_metric = metric
                torch.save(self._network.state_dict(), self._best_model_path)

        if self._use_wandb:
            wandb.finish()
        return self._best_metric

    def run_inference(self):
        """
        Runs inference over the inference dataloader.
        
        Returns:
            A list of outputs (W, S) for each batch.
        """
        outputs = self.inference_epoch()
        return outputs

