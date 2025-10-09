from math import *
import random

output_file="./data/nats_data.csv"
MAX_ROWS = 1000
MAX_FEATURES = 2
DATA = []

CLASSES = [0.5, 0.6, 5374.87]
num_classes = 3

# even split in features
for class_i in range(num_classes):
    for row_i in range(MAX_ROWS // num_classes):
        row = ["UNSET" for _ in range(MAX_FEATURES)]
        if class_i == 0:
            row[0] = row_i / 300
            row[1] = sin(row_i * 10)
        elif class_i == 1:
            row[0] = row_i / 300
            row[1] = sin((row_i + 10)* 10) * 1.05
        elif class_i == 2:
            if row_i % 2 == 0:
                row[0] = 0.37
                row[1] = 1.07
            else:
                class_i = 0
                row[0] = row_i / 300
                row[1] = sin(row_i * 10)
        row.append(CLASSES[class_i])
        DATA.append(row)

with open(output_file, 'w') as f:
    f.write(",".join([f"feature_{i}" for i in range(MAX_FEATURES)]) + ",class\n")
    for row in DATA:
        f.write(",".join([str(datapoint) for datapoint in row]) + "\n")
