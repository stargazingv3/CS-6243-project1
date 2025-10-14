# Zijie Zhang, Oct 2025
# ML Model Tournament Test Harness (Revised for dual evaluation)

import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Import Model Classes from Your Files ---
from random_forest import RandomForestModel
from svm import OptimizedSpiralSVM
from knn import KNN
from svc import SVMClassifier
from decision_tree import DecisionTreeClassifier

# --- Model Configuration ---
# This list makes it easy to add or remove models from the tournament.
# Each tuple contains: (display_name, model_class, initial_parameters)
MODELS_TO_TEST = [
    ("Tuned SVM (GridSearch)", OptimizedSpiralSVM, {}),
    ("Random Forest", RandomForestModel, { 'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 5 }),
    ("K-Nearest Neighbors", KNN, {'k': 5}),
    ("Standard SVC", SVMClassifier, {'kernel': 'rbf', 'C': 1.0, 'gamma': 5.0}),
    ("Decision Tree", DecisionTreeClassifier, {'max_depth': 5}),
]

def load_dataset(filepath):
    """Loads a dataset from a CSV file, skipping the header."""
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    except Exception as e:
        print(f"    Could not load or parse {filepath}. Error: {e}")
        return None, None

def run_evaluation():
    """
    Iterates through all datasets and models, printing the accuracy
    of each model on each dataset.
    """
    dataset_paths = glob.glob("data/*.csv")
    if not dataset_paths:
        print("No CSV files found in the 'data/' directory.")
        return

    print("--- Starting Model Tournament ---")

    dataset_paths = ["data/spirals_2class.csv"]

    for path in sorted(dataset_paths):
        dataset_name = os.path.basename(path)
        print(f"\n--- Testing on Dataset: {dataset_name} ---")

        X, y = load_dataset(path)
        if X is None:
            continue
            
        # Use a 70/30 split for train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        for name, ModelClass, params in MODELS_TO_TEST:
            try:
                print(f"  -> Evaluating Model: {name}")
                
                # 1. Instantiate the model
                model = ModelClass(**params)

                # 2. Train the model (handle both 'train' and 'fit' methods)
                if hasattr(model, 'train'):
                    model.train(X_train, y_train)
                elif hasattr(model, 'fit'):
                    # Handle DecisionTreeClassifier's specific 'fit' signature
                    if isinstance(model, DecisionTreeClassifier):
                         model.fit(X_train, y_train)
                    else:
                         model.fit(X_train, y_train)
                else:
                    print("     Model does not have a 'train' or 'fit' method. Skipping.")
                    continue

                # 3. Predict on BOTH training and test sets
                # Handle models that predict one sample vs. a full batch
                if name in ["Tuned SVM (GridSearch)", "Random Forest", "K-Nearest Neighbors"]:
                    # These models have a predict method for single samples
                    y_pred_train = np.array([model.predict(x) for x in X_train])
                    y_pred_test = np.array([model.predict(x) for x in X_test])
                else:
                    # These models predict on the entire set at once
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                
                # 4. Calculate and report accuracies
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)

                print(f"     Train Accuracy: {train_accuracy:.4f}")
                print(f"     Eval Accuracy:  {test_accuracy:.4f}")

            except Exception as e:
                print(f"     ERROR running {name} on {dataset_name}: {e}")

    print("\n--- Tournament Complete ---")


if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.isdir('data'):
        print("Error: 'data' directory not found. Please create it and add your CSV files.")
    else:
        run_evaluation()