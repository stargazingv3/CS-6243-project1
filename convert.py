import pandas as pd
import os

# --- Configuration ---
# Define the input and output file paths
input_file = 'data/nat_no_cheese.csv'
output_file = 'data/nat_strings.csv'

# Define the mapping for the 'class' column
# This dictionary tells the script what to replace: key -> value
class_mapping = {
    5: 'yes',
    6: 'no'
}

# --- Script Logic ---
def convert_csv_labels():
    """
    Reads a CSV, converts numerical class labels to strings,
    and saves the result to a new file.
    """
    # 1. Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir: # Check if the path contains a directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' is ready.")

    # 2. Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        print("Please create the file with your data and try again.")
        # Create a dummy file for demonstration purposes
        print("Creating a dummy 'data/nat_no_cheese.csv' for you to run the script.")
        dummy_data = {
            'feature_1': [0.0, 0.0033, 0.1, 0.2],
            'feature_2': [0.0, -0.544, 0.98, -0.12],
            'class': [5, 5, 6, 5]
        }
        pd.DataFrame(dummy_data).to_csv(input_file, index=False)
        print("Dummy file created.")

    # 3. Read the CSV file into a pandas DataFrame
    print(f"Reading data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Failed to read the CSV file. Error: {e}")
        return

    # 4. Apply the replacement to the 'class' column
    # The .replace() method is perfect for this task.
    if 'class' in df.columns:
        df['class'] = df['class'].replace(class_mapping)
        print("Applied mapping: 5 -> 'yes', 6 -> 'no'")
    else:
        print("Error: 'class' column not found in the input file.")
        return

    # 5. Save the modified DataFrame to the new CSV file
    # index=False prevents pandas from writing the DataFrame index as a new column
    df.to_csv(output_file, index=False)
    print(f"Successfully created '{output_file}' with updated labels.")


if __name__ == "__main__":
    convert_csv_labels()