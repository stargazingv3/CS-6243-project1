# Zijie Zhang, Oct 2025
# Random Forest Classifier implementation in a modular class structure

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RandomForestModel:
    """A Random Forest classifier."""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42): # <-- ADDED min_samples_leaf
        """
        Initializes the Random Forest model.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf  # <-- ADDED this line
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the Random Forest model by fitting it to the data.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, # <-- ADDED this line
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        print(f"Random Forest model trained with {len(X_train)} samples.")
        
    def predict(self, x_test):
        """
        Predicts the label for a single test data point.
        
        Args:
            x_test (np.ndarray): A single sample's features to predict.
            
        Returns:
            The predicted label.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Please call .train() first.")
        
        # Predict the class for the single sample (must be reshaped to 2D)
        return self.model.predict(x_test.reshape(1, -1))[0]

# This block allows the file to be run standalone for testing
if __name__ == "__main__":
    # Test RandomForest on a dataset where SVMs struggle (the checkerboard one)
    filepath = "./data/nat_no_cheese.csv"
    print(f"--- Running RandomForest Standalone Test ---")
    print(f"Loading data from: {filepath}")

    # Load data
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]

    # Split data for a more realistic evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- RandomForest Execution ---
    model = RandomForestModel(n_estimators=150)
    model.train(X_train, y_train)
    
    # Predict on the test set one sample at a time
    y_pred = [model.predict(x) for x in X_test]
    
    # --- Evaluation ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Self-Test Results:")
    print(f"Accuracy on test data: {accuracy:.4f}")