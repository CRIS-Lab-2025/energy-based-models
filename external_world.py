import os
import gzip
import pickle
import numpy as np
import torch
from urllib.request import urlretrieve

class ExternalWorld:
    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_path, "mnist.pkl.gz")

        # DOWNLOAD MNIST DATASET if it does not exist.
        if not os.path.isfile(path):
            origin = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
            print(f"Downloading data from {origin}")
            urlretrieve(origin, path)

        # LOAD MNIST DATASET
        with gzip.open(path, "rb") as f:
            # For Python 3, use encoding='latin1' to load the pickle.
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        
        train_x_values, train_y_values = train_set
        valid_x_values, valid_y_values = valid_set
        test_x_values, test_y_values = test_set

        # CONCATENATE all splits.
        x_values = np.concatenate([train_x_values, valid_x_values, test_x_values], axis=0)
        y_values = np.concatenate([train_y_values, valid_y_values, test_y_values], axis=0)

        self.x = torch.tensor(x_values, dtype=torch.float32)
        self.y = torch.tensor(y_values, dtype=torch.int64)
        self.size_dataset = len(x_values)
