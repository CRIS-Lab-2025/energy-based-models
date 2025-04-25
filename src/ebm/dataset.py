import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class Dataset(ABC):
    def __init__(self, test_size, batch_size, random_state):
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.X, self.y = self._create_dataset()

    @abstractmethod
    def _create_dataset(self):
        pass 

    @abstractmethod
    def visualize(self):
        pass

    def get_data_loaders(self):
        """Return train and test dataloaders."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

class CirclesDataset3D(Dataset):
    """
    Dataset class for 3D Circles data.
    Creates two 3D circles in different planes for binary classification.
    """
    def __init__(self, test_size=0.2, batch_size=6, random_state=42, num_points=200):
        self.num_points = num_points
        super().__init__(test_size, batch_size, random_state)
        
    def _generate_3d_circle_data(self, center, radius, normal_vector, num_points=100):
        """Generate points on a 3D circle with given center, radius, and normal vector."""
        # Normalize the normal vector
        normal = np.array(normal_vector)
        normal = normal / np.linalg.norm(normal)
        
        # Find two orthogonal vectors in the circle plane
        if np.allclose(normal, [1, 0, 0]):
            v1 = np.array([0, 1, 0])
        else:
            v1 = np.cross(normal, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Generate circle points
        theta = np.linspace(0, 2 * np.pi, num_points)
        circle_points = center + radius * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
        
        return circle_points

    def _create_dataset(self):
        """Create the dataset with two 3D circles."""
        # Circle 1: centered at the origin, on the xy-plane
        center1 = np.array([0, 0, 0])
        radius1 = 1.0
        normal1 = np.array([0, 0, 1])  # Normal to xy-plane
        
        # Circle 2: centered at (0, 1, 0), on the yz-plane (perpendicular to Circle 1)
        center2 = np.array([0, 1, 0])
        radius2 = 1.0
        normal2 = np.array([1, 0, 0])  # Normal to yz-plane
        
        # Generate points
        circle1_points = self._generate_3d_circle_data(center1, radius1, normal1, self.num_points)
        circle2_points = self._generate_3d_circle_data(center2, radius2, normal2, self.num_points)
        
        # Combine into a dataset with labels
        X = np.vstack([circle1_points, circle2_points])
        y = np.array([0] * self.num_points + [1] * self.num_points)
        
        return X, y
        
    def visualize(self):
        """Visualize the dataset."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points for each circle
        colors = ['blue', 'red']
        for i, color in enumerate(colors):
            mask = (self.y == i)
            ax.scatter(self.X[mask, 0], self.X[mask, 1], self.X[mask, 2], c=color, s=10, label=f'Circle {i+1}')
        
        # Add circle centers
        centers = [np.array([0, 0, 0]), np.array([0, 1, 0])]
        ax.scatter([c[0] for c in centers], [c[1] for c in centers], [c[2] for c in centers], 
                  c='black', s=100, marker='x', label='Centers')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Two 3D Circles')
        ax.legend()
        plt.tight_layout()
        plt.show()

class MNISTDataset(Dataset):
    def __init__(self, test_size=0.2, batch_size=64, random_state=42):
        super().__init__(test_size, batch_size, random_state)
        
    def _create_dataset(self):
        """Load MNIST dataset."""
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        y = y.astype(np.int32)  # Convert labels to integers
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        return X, y
        
    def visualize(self, samples=10):
        """Visualize random samples from the dataset."""
        # Get random samples
        indices = np.random.choice(len(self.X), samples, replace=False)
        images = self.X[indices]
        labels = self.y[indices]
        
        # Plot
        _, axes = plt.subplots(1, samples, figsize=(samples*2, 3))
        for i, (image, label, ax) in enumerate(zip(images, labels, axes)):
            ax.imshow(image.reshape(28, 28), cmap='gray')
            ax.set_title(f"Label: {label}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

class XORDataset(Dataset):
    """
    Dataset class for XOR data.
    Creates a simple XOR dataset for binary classification.
    """
    def __init__(self, test_size=0.25, batch_size=4, random_state=42):
        super().__init__(test_size, batch_size, random_state)
        
    def _create_dataset(self):
        # XOR Dataset
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])  # XOR labels
        return X_xor, y_xor
    
    def visualize(self):
        """Visualize the XOR dataset."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], 
                    color='blue', label='Class 0', alpha=0.7)
        plt.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], 
                    color='red', label='Class 1', alpha=0.7)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('XOR Dataset')
        plt.legend()
        plt.grid(True)
        plt.show()
        
valid_datasets = {
    "3d-circles": CirclesDataset3D,
    "mnist": MNISTDataset,
    "xor": XORDataset,
}