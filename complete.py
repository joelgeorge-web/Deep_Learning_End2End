import pickle
import numpy as np
model = pickle.load(open('EEGNet.pkl', 'rb'))

import numpy as np 
import csv
import time  # Assuming you still use time.sleep


def main():
    # Load data from CSV
    with open('X.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        original_shape = None

        for row in reader:
            if 'Original Shape:' in row:
                original_shape = tuple(map(int, row[1].strip('()').split(',')))
            else:
                data.append([float(val) for val in row]) 

    # Reshape into the original 3D array
    if original_shape:
        X_test_loaded = np.array(data).reshape(original_shape)

        # Predict for first 4 rows only
        X_test_loaded = X_test_loaded[:4]
        data = model.predict(X_test_loaded).argmax(axis=-1)  
    time.sleep(2)
    for direction in data:
        print(direction)
        if direction == 0:
            print("Forward")
        elif direction == 1:
            print("Left")
        elif direction == 2:
            print("Right")
        elif direction == 3:
            print("Reverse")
        time.sleep(5)
# Call the main function
if __name__ == "__main__":
    main()


