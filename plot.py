import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
csv_file_path = os.path.join('data', 'example.csv')
output_image_path = 'example.png'
# ---------------------

def create_plot(csv_path, output_path):
    """
    Reads data from a CSV file, creates a scatter plot, and saves it.
    
    Args:
        csv_path (str): The path to the input CSV file.
        output_path (str): The path to save the output PNG image.
    """
    # Check if the file exists before trying to read it
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found.")
        print("Please make sure the CSV file is in the correct directory.")
        # Create dummy data and directory to allow the script to run for demonstration
        print("Creating a dummy 'data/example.csv' for demonstration purposes.")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        dummy_data = {
            'feature_1': [1.44, -0.97, -0.80, 0.5, -0.5],
            'feature_2': [0.33, -1.15, -1.17, 0.6, -0.4],
            'class': [1, 2, 2, 1, 2]
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(csv_path, index=False)
        print(f"Dummy file created at '{csv_path}'.")
    else:
        # Load the dataset from the CSV file
        df = pd.read_csv(csv_path)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create a scatter plot
    # The 'c' argument colors the points based on the 'class' column
    # The 'cmap' argument defines the color map to use
    scatter = ax.scatter(
        x=df['feature_1'], 
        y=df['feature_2'], 
        c=df['class'], 
        cmap='viridis',
        alpha=0.8,
        edgecolor='k',
        linewidth=0.5
    )

    # --- Aesthetics and Labels ---
    ax.set_title('Scatter Plot of Features by Class', fontsize=16)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)

    # Create a legend
    # It automatically finds the unique classes and assigns the correct colors
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    # --- Save and Show ---
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == '__main__':
    create_plot(csv_file_path, output_image_path)