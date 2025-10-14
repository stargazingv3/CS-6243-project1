# advanced_datasets.py
# Adversarial Dataset Generators for ML Tournament
# GOAL: Create datasets that are high-frequency and highly overlapping
#       to defeat tree-based and linear models.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Configuration ---
sns.set_theme(style="whitegrid")

def plot_dataset(df, title, filename):
    """Visualizes and saves a 2D classification dataset."""
    plt.figure(figsize=(10, 8))
    # Use a 'scramble' colormap and low alpha to show overlap
    sns.scatterplot(
        data=df,
        x='feature_1',
        y='feature_2',
        hue='class',
        palette='deep',
        s=20,          # Smaller points
        alpha=0.5,     # More transparency
        legend='full'
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.legend(title='Class')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()

# =========================================================================
# --- APPROACH 1: High-Frequency Checkerboard ---
# =========================================================================

def generate_high_freq_checkerboard(n_samples=1000, n_classes=2, freq=10.0, noise=0.1, filename="adv_checker.csv"):
    """
    Generates a high-frequency checkerboard pattern.
    
    TARGET: Defeat Tree-based models and Linear models.
    WHY: This is a high-frequency XOR pattern. A linear model cannot
         solve it (accuracy will be ~1/n_classes). A tree model will
         be forced to make thousands of splits to approximate the
         tiny squares and will fail to generalize.
         An RBF SVM *might* solve this with a tuned C and gamma.
    """
    print(f"Generating High-Frequency Checkerboard ({n_classes} classes)...")
    
    X = np.random.uniform(-1, 1, (n_samples, 2))
    
    # Create the high-frequency grid
    grid_x = np.floor(X[:, 0] * freq)
    grid_y = np.floor(X[:, 1] * freq)
    
    # Assign class based on the sum of grid indices, modulo n_classes
    y = (grid_x + grid_y) % n_classes
    
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    df = pd.DataFrame({'feature_1': X[:, 0], 'feature_2': X[:, 1], 'class': y.astype(int)})
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}\n")
    return df

# =========================================================================
# --- APPROACH 2: Intertwined Spirals ---
# =========================================================================

def generate_intertwined_spirals(n_samples=1000, n_classes=3, tightness=5.0, noise=0.2, filename="adv_spirals.csv"):
    """
    Generates multiple, tightly-wound, intertwined spirals.
    
    TARGET: Defeat Tree-based models and Linear models.
    WHY: This is a classic, difficult non-linear problem. The decision
         boundaries are long, curved, and non-monotonic. Trees and
         linear models will fail completely.
    """
    print(f"Generating Intertwined Spirals ({n_classes} classes)...")
    
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    
    n_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        start_idx = i * n_per_class
        end_idx = (i + 1) * n_per_class if i < n_classes - 1 else n_samples
        
        # Generate points with increasing radius
        radius = np.linspace(0.1, 1.0, end_idx - start_idx)
        
        # Base angle for this spiral
        base_angle = (i * 2 * np.pi / n_classes)
        
        # Angle for each point along the spiral
        angle = radius * tightness * 2 * np.pi + base_angle
        
        # Add noise to angle and radius
        noise_shape = (end_idx - start_idx,)
        angle_noise = np.random.normal(0, noise, noise_shape)
        radius_noise = np.random.normal(0, noise / 5.0, noise_shape)

        X[start_idx:end_idx, 0] = (radius + radius_noise) * np.cos(angle + angle_noise)
        X[start_idx:end_idx, 1] = (radius + radius_noise) * np.sin(angle + angle_noise)
        y[start_idx:end_idx] = i

    df = pd.DataFrame({'feature_1': X[:, 0], 'feature_2': X[:, 1], 'class': y})
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}\n")
    return df

# =========================================================================
# --- APPROACH 3: High-Frequency Rotated Checkerboard ---
# =========================================================================

def generate_high_freq_rotated(n_samples=1000, n_classes=4, freq=10.0, noise=0.1, filename="adv_rotated.csv"):
    """
    Generates a high-frequency checkerboard pattern rotated by 45 degrees.
    
    TARGET: A direct attack on Tree-based models.
    WHY: Trees *only* make axis-aligned splits. By rotating the
         problem, we make it maximally inefficient for them. They
         will fail, while an RBF SVM (which is rotation-invariant)
         can solve this just as easily as the regular checkerboard.
    """
    print(f"Generating Rotated High-Frequency Checkerboard ({n_classes} classes)...")

    X_orig = np.random.uniform(-1, 1, (n_samples, 2))
    
    # Create 45-degree rotation matrix
    theta = np.pi / 4.0
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Rotate the coordinates *before* assigning classes
    X_rot = X_orig @ rotation_matrix.T
    
    # Assign classes based on the *rotated* grid
    grid_x_rot = np.floor(X_rot[:, 0] * freq)
    grid_y_rot = np.floor(X_rot[:, 1] * freq)
    y = (grid_x_rot + grid_y_rot) % n_classes
    
    # Add noise to the *original* coordinates
    X = X_orig + np.random.normal(0, noise, X_orig.shape)

    df = pd.DataFrame({'feature_1': X[:, 0], 'feature_2': X[:, 1], 'class': y.astype(int)})
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}\n")
    return df

# =========================================================================
# --- APPROACH 4: Wave Interference (Your Sine Wave Idea) ---
# =========================================================================

def generate_wave_interference(n_samples=1000, n_classes=5, freq=8.0, noise=0.1, filename="adv_wave.csv"):
    """
    Generates classes based on the interference pattern of two
    high-frequency sine waves.
    
    TARGET: Defeat all simple models.
    WHY: This is your "high-D" concept. The class is a function
         of sin(k*x) + sin(k*y). This creates complex, non-linear,
         non-monotonic, and repeating class boundaries. It will
         look like a total mess of overlapping points, but a
         fine-tuned RBF SVM may be able to find the hidden pattern.
    """
    print(f"Generating Wave Interference ({n_classes} classes)...")

    X = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
    
    # Create the wave interference pattern
    wave_pattern = np.sin(X[:, 0] * freq) + np.sin(X[:, 1] * freq)
    
    # Cut the resulting 1D values into n_classes
    # We find the quantiles to create N classes of equal size
    bins = np.quantile(wave_pattern, np.linspace(0, 1, n_classes + 1))
    bins[-1] += 0.01 # Make sure the max value is included
    bins[0] -= 0.01  # Make sure the min value is included
    
    y = pd.cut(wave_pattern, bins=bins, labels=False, right=True)

    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    df = pd.DataFrame({'feature_1': X[:, 0], 'feature_2': X[:, 1], 'class': y.astype(int)})
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}\n")
    return df

# =========================================================================
# --- Main execution block: Generate variations ---
# =========================================================================

if __name__ == "__main__":
    data_dir = 'data_advanced'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define common parameters
    params = {
        'checker_freq': 10.0, 'checker_noise': 0.05,
        'spiral_tightness': 5.0, 'spiral_noise': 0.2,
        'rotated_freq': 8.0, 'rotated_noise': 0.1,
        'wave_freq': 7.0, 'wave_noise': 0.1
    }

    # *** MODIFIED: Loop from 2 to 5 classes (inclusive) ***
    for n_classes in range(2, 6):
        print(f"\n--- Generating {n_classes}-Class Variations ---")

        # 1. High-Frequency Checkerboard
        filename_csv = os.path.join(data_dir, f"checker_{n_classes}class.csv")
        filename_png = os.path.join(data_dir, f"checker_{n_classes}class.png")
        title = f"Adversarial Checkerboard ({n_classes} Classes)"
        
        df_checker = generate_high_freq_checkerboard(
            n_classes=n_classes, 
            freq=params['checker_freq'], 
            noise=params['checker_noise'], 
            filename=filename_csv
        )
        plot_dataset(df_checker, title, filename_png)

        # 2. Intertwined Spirals
        filename_csv = os.path.join(data_dir, f"spirals_{n_classes}class.csv")
        filename_png = os.path.join(data_dir, f"spirals_{n_classes}class.png")
        title = f"Adversarial Spirals ({n_classes} Classes)"
        
        df_spirals = generate_intertwined_spirals(
            n_classes=n_classes, 
            tightness=params['spiral_tightness'], 
            noise=params['spiral_noise'], 
            filename=filename_csv
        )
        plot_dataset(df_spirals, title, filename_png)
        
        # 3. High-Frequency Rotated Checkerboard
        filename_csv = os.path.join(data_dir, f"rotated_{n_classes}class.csv")
        filename_png = os.path.join(data_dir, f"rotated_{n_classes}class.png")
        title = f"Adversarial Rotated ({n_classes} Classes)"

        df_rotated = generate_high_freq_rotated(
            n_classes=n_classes, 
            freq=params['rotated_freq'], 
            noise=params['rotated_noise'], 
            filename=filename_csv
        )
        plot_dataset(df_rotated, title, filename_png)
        
        # 4. Wave Interference
        filename_csv = os.path.join(data_dir, f"wave_{n_classes}class.csv")
        filename_png = os.path.join(data_dir, f"wave_{n_classes}class.png")
        title = f"Adversarial Wave Interference ({n_classes} Classes)"

        df_wave = generate_wave_interference(
            n_classes=n_classes, 
            freq=params['wave_freq'], 
            noise=params['wave_noise'], 
            filename=filename_csv
        )
        plot_dataset(df_wave, title, filename_png)

    print("\n--- All dataset variations (2-5 classes) generated. ---")