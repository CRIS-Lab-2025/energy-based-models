import os
import gzip
import pickle
import numpy as np
import torch
from urllib.request import urlretrieve

class ExternalWorld:
    def __init__(self, x_values, y_values, size_dataset=None):
        self.x = torch.tensor(x_values, dtype=torch.float32)
        self.y = torch.tensor(y_values, dtype=torch.int64)
        self.size_dataset = size_dataset if size_dataset is not None else len(self.x)

class MNISTExternalWorld(ExternalWorld):
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

        super().__init__(x_values, y_values)

        #IMPLEMENTATION FROM train_model.py :
        # def __init__(self):
        #     path = os.path.join(os.getcwd(), "mnist.pkl.gz")
        #     if not os.path.isfile(path):
        #         urlretrieve("http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz", path)
        #     with gzip.open(path, "rb") as f:
        #         train, valid, test = pickle.load(f, encoding="latin1")
        #     self.x = torch.tensor(np.vstack((train[0], valid[0], test[0])), dtype=torch.float32)
        #     self.y = torch.tensor(np.hstack((train[1], valid[1], test[1])), dtype=torch.int64)
        #     self.size_dataset = len(self.x)