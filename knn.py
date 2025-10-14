# Zijie Zhang, Sep 2025
# K-Nearest Neighbors (KNN) implementation in a modular class structure

import numpy as np
from collections import Counter

def _majority_label(y):
    """Finds the most common label in a list of labels."""
    if not list(y):
        return None
    return Counter(y).most_common(1)[0][0]

def _euclidean_distance(x1, x2):
    """Calculates the Euclidean distance between two numeric vectors."""
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    """A K-Nearest Neighbors classifier."""
    def __init__(self, k=5):
        """
        Initializes the KNN model.
        
        Args:
            k (int): The number of neighbors to use for classification.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        """
        Trains the model. For KNN, this simply means storing the dataset.
        
        Args:
            X_train (np.ndarray): The training feature data.
            y_train (np.ndarray): The training label data.
        """
        self.X_train = X_train
        self.y_train = y_train
        print(f"KNN model 'trained' with {len(self.X_train)} samples.")

    def predict(self, x_test):
        """
        Predicts the label for a single test data point.
        
        Args:
            x_test (np.ndarray): A single sample's features to predict.
            
        Returns:
            The predicted label.
        """
        # 1. Calculate distances from the test point to all training points
        distances = [_euclidean_distance(x_test, x_train) for x_train in self.X_train]

        # 2. Get the indices of the k-nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]

        # 3. Get the labels of these k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # 4. Return the most common label (majority vote)
        return _majority_label(k_nearest_labels)

# This block allows the file to be run standalone for testing
if __name__ == "__main__":
    filepath = "./data/example.csv"
    print(f"--- Running KNN Standalone Test ---")
    print(f"Loading data from: {filepath}")

    # Load data, skipping the header
    rawdata = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=object)
    
    # --- Data Preprocessing ---
    y = rawdata[:, -1].astype(float)
    X = rawdata[:, :-1].astype(float)

    # --- KNN Execution ---
    k_value = 5
    model = KNN(k=k_value)
    
    model.train(X, y)
    
    y_pred = [model.predict(x) for x in X]
    
    # --- Evaluation ---
    accuracy = np.mean(np.array(y_pred) == y)
    print(f"\nKNN Algorithm Self-Test Results:")
    print(f"Using k = {k_value}")
    print(f"Accuracy on training data: {accuracy:.4f}")