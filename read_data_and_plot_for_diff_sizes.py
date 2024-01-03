import pandas as pd
import matplotlib.pyplot as plt
import os


def get_experiment_results_directory():
    # Get the current working directory (cwd)
    cwd = os.getcwd()
    # Define the path for the directory to store experiment results
    results_dir = os.path.join(cwd, 'experiment_results')
    # Create the directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


experiment_results_dir = get_experiment_results_directory()


# Read the CSV file into DataFrames
df_iterative = pd.read_csv(os.path.join(experiment_results_dir, "iterative_refinement_results.csv"))
df_HEM = pd.read_csv(os.path.join(experiment_results_dir, "GPP_with_HEM_results.csv"))
df_SP = pd.read_csv(os.path.join(experiment_results_dir, "spectral_partitioning_results.csv"))

# Calculate the average edge cuts for each graph size for all algorithms
average_edge_cuts_iterative = df_iterative.groupby('Graph Size')['Edge Cuts'].mean().reset_index()
average_edge_cuts_HEM = df_HEM.groupby('Graph Size')['Edge Cuts'].mean().reset_index()
average_edge_cuts_SP = df_SP.groupby('Graph Size')['Edge Cuts'].mean().reset_index()  # New algorithm

# Plot the average edge cuts with error bars for all algorithms
plt.errorbar(average_edge_cuts_iterative['Graph Size'], average_edge_cuts_iterative['Edge Cuts'],
             yerr=df_iterative.groupby('Graph Size')['Edge Cuts'].std(), fmt='-o', capsize=5,
             label='Iterative Refinement')
plt.errorbar(average_edge_cuts_HEM['Graph Size'], average_edge_cuts_HEM['Edge Cuts'],
             yerr=df_HEM.groupby('Graph Size')['Edge Cuts'].std(), fmt='-x', capsize=5, label='Using HEM')
plt.errorbar(average_edge_cuts_SP['Graph Size'], average_edge_cuts_SP['Edge Cuts'],
             yerr=df_SP.groupby('Graph Size')['Edge Cuts'].std(), fmt='-^', capsize=5, label='Spectral Partitioning')

plt.title('Average Edge Cuts vs Graph Size')
plt.xlabel('Graph Size')
plt.ylabel('Average Edge Cuts')
plt.legend()
plt.grid(True)
plt.show()
