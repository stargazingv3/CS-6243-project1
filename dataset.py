import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 1000

# Create a dictionary to hold the data
data = {
    # Generate 1000 random 0s and 1s for the first feature
    'feature1': np.random.randint(0, 2, num_samples),
    
    # Generate 1000 random 0s and 1s for the second feature
    'feature2': np.random.randint(0, 2, num_samples)
}

# Create a pandas DataFrame from the dictionary
df = pd.DataFrame(data)

# Calculate the 'class' column using the XOR rule
# The ^ symbol is the bitwise XOR operator
df['class'] = df['feature1'] ^ df['feature2']

# Save the DataFrame to a CSV file
# index=False prevents pandas from writing the row numbers to the file
df.to_csv('xor_dataset.csv', index=False)

print("Successfully created 'xor_dataset.csv' with 1000 samples.")