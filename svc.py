# SVMClassifier implementation in a modular class structure

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVMClassifier:
    """A wrapper for the scikit-learn Support Vector Classifier."""

    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        """
        Initializes the SVM model.
        
        Args:
            kernel (str): Specifies the kernel type to be used in the algorithm.
            C (float): Regularization parameter.
            gamma (float or 'scale'/'auto'): Kernel coefficient.
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        print(f"SVMClassifier initialized with kernel={self.kernel}, C={self.C}, gamma={self.gamma}")

    def fit(self, X_train, y_train):
        """
        Trains (fits) the SVM model on the training data.
        
        Args:
            X_train (np.ndarray): The training feature data.
            y_train (np.ndarray): The training label data.
        """
        # Ensure y_train is a 1D array, as expected by sklearn
        if y_train.ndim > 1:
            y_train = y_train.ravel()
            
        self.model.fit(X_train, y_train)
        print(f"SVM model fitted on {len(X_train)} samples.")

    def predict(self, X_test):
        """
        Predicts class labels for samples in X_test.
        
        Args:
            X_test (np.ndarray): The feature data to predict.
            
        Returns:
            np.ndarray: The predicted labels.
        """
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        """
        Calculates the mean accuracy on the given test data and labels.
        
        Args:
            X_test (np.ndarray): The test feature data.
            y_test (np.ndarray): The true labels for X_test.
            
        Returns:
            float: The accuracy score.
        """
        return self.model.score(X_test, y_test)

# This block allows the file to be run standalone for testing
if __name__ == "__main__":
    filepath = "data/nat_data.csv"
    print(f"--- Running SVMClassifier Standalone Test ---")
    print(f"Loading data from: {filepath}")

    # --- Data Loading and Preprocessing ---
    df = pd.read_csv(filepath)
    X = df[["feature_1", "feature_2"]].values
    # .values on a single column Series returns a 1D array, which is preferred
    y = df["class"].values.astype(int) 

    # --- Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- SVM Execution ---
    # Instantiate the classifier with your specified parameters
    svm_clf = SVMClassifier(kernel="rbf", C=1.0, gamma=5.0)

    # Train the model
    svm_clf.fit(X_train, y_train)

    # --- Evaluation ---
    accuracy = svm_clf.score(X_test, y_test)
    print("\nSVM Classifier Test Results:")
    print(f"Accuracy on test data: {accuracy:.4f}")