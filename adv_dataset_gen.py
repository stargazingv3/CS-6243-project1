# adversarial dataset generator

# Generates dataset with 2 numerical features, <= 5 classes, with size <= 1000

import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import sys

def adv_dataset_gen(num_rows: int = 500, num_classes: int = 3,
                adv_frac: float = 0.2, label_flip: float = 0.07, outlier_frac: float = 0.03,
                random_seed: int = 42, cluster_std: float = 1.2, out_path: str = "adv_dataset.csv"):
    
    np.random.seed(random_seed)
    
    # Base clustered data
    X, y = make_blobs(n_samples = num_rows, centers = num_classes, n_features = 2,
                      cluster_std = cluster_std, random_state = random_seed + 1)
    
    df = pd.DataFrame(X, columns = ['feature1', 'feature2'])
    df['class'] = y.astype(int)
    
    # Choose idx to modify
    n_adversarial = int(num_rows * adv_frac)
    n_label_flip = int(num_rows * label_flip)
    n_outliers = int(num_rows * outlier_frac)
    
    all_indices = np.arange(num_rows)
    np.random.shuffle(all_indices)
    
    adv_idx = all_indices[:n_adversarial]
    flip_idx = all_indices[n_adversarial:n_adversarial + n_label_flip]
    out_idx = all_indices[n_adversarial + n_label_flip:n_adversarial + n_label_flip + n_outliers]

    # Compute centroids from current data
    centroids = []
    
    for cls in range(num_classes):
        # if a class has zero points (rare with make_blobs), use zeros
        subset = df[df["class"] == cls][["feature1", "feature2"]]
        if len(subset) == 0:
            centroids.append(np.array([0.0, 0.0]))
        else:
            centroids.append(subset.mean().values)

    # Move point partway toward another class centroid (label unchanged)
    for i in adv_idx:
        current_label = int(df.at[i, "class"])
        other_classes = [c for c in range(num_classes) if c != current_label]
        
        if not other_classes:
            continue  # nothing to do if single-class
        target = np.random.choice(other_classes)
        point = df.loc[i, ["feature1", "feature2"]].values.astype(float)
        vec_to_target = centroids[target] - point
        # epsilon chosen to make the point significantly closer to the other centroid
        epsilon = np.random.uniform(0.35, 0.7)
        new_point = point + epsilon * vec_to_target
        df.at[i, "feature1"] = float(new_point[0])
        df.at[i, "feature2"] = float(new_point[1])
        # label unchanged

    # Replace label by a random different class
    for i in flip_idx:
        current_label = int(df.at[i, "class"])
        other_classes = [c for c in range(num_classes) if c != current_label]
        if not other_classes:
            continue
        df.at[i, "class"] = int(np.random.choice(other_classes))

    # Outliers
    # We'll choose them from a larger uniform range based on cluster centers' spread
    # Estimate scale from the dataset to choose a reasonable outlier magnitude
    spread = np.std(df[["feature1", "feature2"]].values)
    if spread <= 0:
        spread = 10.0
    outlier_scale = max(10.0, spread * 6.0)
    for i in out_idx:
        far_point = np.random.uniform(-outlier_scale, outlier_scale, size=2)
        df.at[i, "feature1"] = float(far_point[0])
        df.at[i, "feature2"] = float(far_point[1])
        # assign a random label within existing range
        df.at[i, "class"] = int(np.random.randint(0, num_classes))

    # Final check
    assert df.shape[1] == 3, "Expect exactly 3 columns (2 features + class)."
    if df.shape[0] != num_rows:
        print("Warning: row count changed unexpectedly.", file=sys.stderr)
    if df["class"].nunique() > 5:
        raise RuntimeError("Number of classes exceed 5 (unexpected).")

    # Save CSV
    df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    df = adv_dataset_gen()