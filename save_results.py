import os
import pandas as pd

def get_experiment_results_directory():
    # Get the current working directory (cwd)
    cwd = os.getcwd()
    # Define the path for the directory to store experiment results
    results_dir = os.path.join(cwd, 'experiment_results')
    # Create the directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def save_results_to_csv(data, algorithm_name, directory=None):
    # If no directory is provided, use the default experiment results directory
    if directory is None:
        directory = get_experiment_results_directory()

    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Path for the CSV file
    csv_file_path = os.path.join(directory, f"{algorithm_name}_results.csv")

    # If file does not exist or is empty, write with header, otherwise append without header
    if not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        data.to_csv(csv_file_path, index=False)
    else:
        # Append data without header
        data.to_csv(csv_file_path, mode='a', header=False, index=False)