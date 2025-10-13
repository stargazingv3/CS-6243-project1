import csv

def modify_class_label(input_filename, output_filename):
    """
    Reads a CSV file, multiplies the 'class' column by 10, and saves it to a new file.

    Args:
        input_filename (str): The name of the CSV file to read.
        output_filename (str): The name of the CSV file to write the results to.
    """
    try:
        with open(input_filename, mode='r', newline='') as infile, \
             open(output_filename, mode='w', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 1. Read and write the header row
            header = next(reader)
            writer.writerow(header)

            # 2. Find the index of the 'class' column
            # This makes the script more robust if column order changes
            try:
                class_column_index = header.index('class')
            except ValueError:
                print("Error: 'class' column not found in the header.")
                return

            # 3. Process the remaining rows
            for row in reader:
                # Get the value from the 'class' column
                class_value_str = row[class_column_index]

                # Convert to float, multiply by 10, and convert to an integer
                try:
                    class_value_float = float(class_value_str)
                    new_class_value = int(class_value_float * 10)
                    
                    # Update the value in the row (must be a string)
                    row[class_column_index] = str(new_class_value)
                    
                    # Write the modified row to the output file
                    writer.writerow(row)
                except ValueError:
                    print(f"Warning: Could not process value '{class_value_str}' in row. Skipping.")
                    # Optionally, write the row as-is
                    # writer.writerow(row)
        
        print(f"Successfully processed the file and saved the output to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")

# --- Main execution ---
if __name__ == "__main__":
    input_csv = "data/nat_data.csv"
    output_csv = "data_updated.csv"
    modify_class_label(input_csv, output_csv)