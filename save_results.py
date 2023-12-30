import pandas as pd
import os

directory = "/Users/furka/PycharmProjects/CMP3005Project/experiment_results"


def save_results_to_csv(data, algorithm_name, directory =directory):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Path for the CSV file
    csv_file_path = os.path.join(directory, f"{algorithm_name}_results.csv")

    # Check if the file exists
    if os.path.exists(csv_file_path):
        # If file exists, append data
        with open(csv_file_path, 'a') as f:
            data.to_csv(f, header=False, index=False)
    else:
        # If file does not exist, create it and add data
        data.to_csv(csv_file_path, index=False)
