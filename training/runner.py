import os
import json
import logging
import torch
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt



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
        self._result_path = config.path
        create_dir(self._result_path)
        self._model_path = config.path + "/model"
        create_dir(self._model_path)
        self._best_model_path = os.path.join(self._model_path, 'best_model.pth')
        self._checkpoint_epoch = config.training['checkpoint_interval']
        self._epochs = config.training['num_epochs']

        # Initialize wandb if enabled.
        self._use_wandb = config.training['wandb']
        if self._use_wandb:
            wandb.init(project=config.get('project_name', 'default_project'), config=config)
            if config.get('tag'):
                wandb.run.tags = [config['tag']]

        # Set up logging if enabled.
        self._log = config.training['log']
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
        # Get original parameters
        W, B = self._network.weights, self._network.biases
        # self._network._reset_state()
        # Expand without detaching/cloning so gradients can flow back:

        outputs = []
        targets = []
        inputs = []
        for x, target in tqdm(self._dataloader, desc="Training Batches"):

            W_expanded = W.unsqueeze(0).expand(1, *W.shape)
            B_expanded = B.unsqueeze(0).expand(1, *B.shape)

            # print("shape")
            # print(x.shape, target.shape)
            self._optimizer.zero_grad()
            S = self._network.set_input(x)
            
            # Let the network settle to equilibrium.
            S = self._updater.compute_equilibrium(S, W_expanded, B_expanded, target)
            # Compute parameter gradients.
            weight_grads, bias_grads = self._differentiator.compute_gradient(S, W_expanded, B_expanded, target)
            # Assign gradients to the original parameters.
            # print(W)
            W.grad, B.grad = weight_grads, bias_grads
            self._optimizer.step()
            output = S[:,self._network.layers[-1]].clone()
            outputs.append(output)
            targets.append(target)
            # print(output, target)
            inputs.append(x)
        self._network.clamp_weights()
        return inputs, outputs, targets, W, B

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
        targets = []
        for x, target in tqdm(self._inference_dataloader, desc="Inference Batches"):
            # (Zeroing gradients is optional during inference.)
            self._optimizer.zero_grad()
            S = self._network.set_input(x)
            W, B = self._network.weight, self._network.bias
            W, S = self._updater.compute_equilibrium(S, W, B)
            output = S[self._network.layers[-1]].clone()
            outputs.append(output)
            targets.append(target)
        return outputs


    def run_training(self):
        """Runs the full training loop over epochs with logging and side-by-side visualization."""
        
        for epoch in range(self._epochs):
            print(f"\nEpoch {epoch}")
            
            inputs, outputs, targets, W, B = self.training_epoch()
            inputs = torch.cat(inputs).cpu().numpy()
            targets = torch.cat(targets).cpu().numpy()
            
            outputs = torch.cat(outputs)
            # if self._config.model['layers'][self._config.model['output_layer']] > 1:
            #     outputs = outputs.argmax(dim=1)
            # else:
            outputs = (outputs > 0.5).float().flatten()
            
            outputs = outputs.cpu().numpy()

            # Define consistent colors for 0 and 1
            colors = {0.0: "red", 1.0: "green"}
            pred_colors = [colors[val] for val in outputs]
            target_colors = [colors[val] for val in targets]

            # Side-by-side scatter plots
            fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

            axes[0].scatter(inputs[:, 0], inputs[:, 1], c=pred_colors, alpha=0.6, edgecolors='k')
            axes[0].set_title("Predicted Labels")
            axes[0].set_xlabel("Feature 1")
            axes[0].set_ylabel("Feature 2")

            axes[1].scatter(inputs[:, 0], inputs[:, 1], c=target_colors, alpha=0.6, edgecolors='k')
            axes[1].set_title("Actual Labels")
            axes[1].set_xlabel("Feature 1")

            plt.tight_layout()
            plt.show()

            print("W last", W[-1])

            print("B", B)

            # Compute accuracy
            accuracy = (outputs == targets).mean()
            print(f"Accuracy: {accuracy:.4f}")

            if self._log:
                logging.info(f"Epoch {epoch}: Accuracy {accuracy:.4f}")

        return self._best_metric


    def run_inference(self):
        """
        Runs inference over the inference dataloader.
        
        Returns:
            A list of outputs (W, S) for each batch.
        """
        outputs = self.inference_epoch()
        return outputs

