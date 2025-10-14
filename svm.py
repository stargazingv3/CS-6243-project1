import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform

class OptimizedSpiralSVM:
    """
    RBF SVM optimized for intertwined spirals.
    
    Key insights:
    1. NO polar transformation - it destroys separability for intertwined spirals
    2. StandardScaler on original (x, y) coordinates is essential
    3. Careful gamma tuning prevents overfitting
    4. Lower gamma values (smoother decision boundaries) work better
    """
    
    def __init__(self, param_distributions=None):
        if param_distributions is None:
            # CRITICAL: Much lower gamma range to prevent overfitting
            # High gamma = memorizes training data
            # Low gamma = smooth, generalizable boundaries
            self.param_distributions = {
                'C': loguniform(1e-1, 1e3),      # Regularization strength
                'gamma': loguniform(1e-2, 1e1)   # RBF kernel bandwidth (LOWER range)
            }
        else:
            self.param_distributions = param_distributions
            
        self.scaler = None
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the RBF SVM on original (x, y) coordinates.
        """
        print("Scaling features...")
        # Scaling is CRITICAL for SVM - ensures features have similar ranges
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Tuning RBF SVM with RandomizedSearchCV...")
        print(f"Search space: C={self.param_distributions['C']}, gamma={self.param_distributions['gamma']}")
        
        # Use stratified k-fold to ensure each fold has balanced classes
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
        
        base_model = SVC(kernel='rbf', class_weight='balanced')
        
        random_search = RandomizedSearchCV(
            base_model,
            self.param_distributions,
            n_iter=50,  # More iterations for better search
            cv=stratified_kfold,
            n_jobs=-1,
            random_state=42,
            verbose=1,
            scoring='accuracy'
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        self.model = random_search.best_estimator_
        
        print(f"\nBest parameters found:")
        print(f"  C = {random_search.best_params_['C']:.6f}")
        print(f"  gamma = {random_search.best_params_['gamma']:.6f}")
        print(f"  CV score = {random_search.best_score_:.4f}")
        
        # Report train accuracy
        y_train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"\nTraining accuracy: {train_acc:.4f}")
        print(f"Model trained with {len(X_train)} samples.\n")
        
    def predict(self, x_test):
        """
        Predicts the label for a single test data point.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been trained yet. Please call .train() first.")
        
        x_test_reshaped = x_test.reshape(1, -1)
        x_test_scaled = self.scaler.transform(x_test_reshaped)
        return self.model.predict(x_test_scaled)[0]


if __name__ == "__main__":
    filepath = "./data/old/nat_no_cheese.csv"
    print("="*70)
    print("OPTIMIZED RBF SVM FOR INTERTWINED SPIRALS")
    print("="*70)
    print(f"\nLoading data from: {filepath}\n")

    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    except OSError:
        print(f"ERROR: Could not find file at: {filepath}")
        print("Please ensure the file exists.")
        exit(1)
        
    X = data[:, :-1]
    y = data[:, -1]

    print(f"Dataset: {len(X)} samples, {len(np.unique(y))} classes")
    print(f"Class distribution: {np.bincount(y.astype(int))}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples\n")

    # Train model
    model = OptimizedSpiralSVM()
    model.train(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = [model.predict(x) for x in X_test]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
