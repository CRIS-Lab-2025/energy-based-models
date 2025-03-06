import json

class Config:
    def __init__(self, json_file="util/default_config.json", args=None):
        # Default configuration settings
        self.path = "papers/fast-drn/model/EP"
        self.comment = "Make dict of hyperparameters for ease of use."
        self.device = "cuda"
        self.model = {
            "model_name": "model",
            "weight_init_dist": "xavier_uniform",
            "layers": [1, 64, 1],
            "output_layer": -1
        }
        self.gradient_estimator = {
            "nudging": 0.2,
            "variant": "centered"
        }
        self.minimizer = {
            "training_mode": "asynchronous",
            "training_iterations": 20,
            "inference_mode": "asynchronous",
            "inference_iterations": 50
        }
        self.optimizer = {
            "optimizer_name": "adam",
            "learning_rates_weights": 0.001,
            "learning_rates_bias": 0.001,
            "weight_decay": 0.0,
            "momentum": 0.0
        }
        self.training = {
            "num_epochs": 25,
            "batch_size": 64
        }

        # Load configuration from a JSON file if provided
        if json_file:
            self.load_from_json(json_file)

        # Load configuration from command-line arguments if provided
        if args:
            self.load_from_args(args)

    def load_from_args(self, args):
        # Update configuration based on command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                if isinstance(getattr(self, key, None), dict):
                    # Update nested dictionary if the attribute is a dictionary
                    for sub_key, sub_value in value.items():
                        self.__dict__[key][sub_key] = sub_value
                else:
                    # Set attribute directly
                    setattr(self, key, value)

    def load_from_json(self, json_file):
        # Update configuration based on a JSON file
        with open(json_file, 'r') as file:
            json_config = json.load(file)  # Load JSON content
            for key, value in json_config.items():
                # Set attributes based on keys and values from the JSON file
                if isinstance(getattr(self, key, None), dict):
                    # Update nested dictionary if the attribute is a dictionary
                    for sub_key, sub_value in value.items():
                        self.__dict__[key][sub_key] = sub_value
                else:
                    setattr(self, key, value)

    def to_device(self, tensor_or_tensors):
        """
        Moves tensor(s) to the device specified in the config.
        
        Args:
            tensor_or_tensors: A single tensor or a list/tuple of tensors
            
        Returns:
            The tensor(s) moved to the specified device
        """
        import torch
        
        if isinstance(tensor_or_tensors, (list, tuple)):
            return [self.to_device(t) if isinstance(t, torch.Tensor) else t for t in tensor_or_tensors]
        elif isinstance(tensor_or_tensors, dict):
            return {k: self.to_device(v) if isinstance(v, torch.Tensor) else v for k, v in tensor_or_tensors.items()}
        elif isinstance(tensor_or_tensors, torch.Tensor):
            return tensor_or_tensors.to(self.device)
        else:
            return tensor_or_tensors