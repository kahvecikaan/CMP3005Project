import pandas as pd
import matplotlib.pyplot as plt
import os


# Function to get or create the experiment results directory
def get_experiment_results_directory():
    # Get the current working directory (cwd)
    cwd = os.getcwd()
    # Define the path for the directory to store experiment results
    results_dir = os.path.join(cwd, 'experiment_results')
    # Create the directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


# Use the function to get or create the directory
experiment_results_dir = get_experiment_results_directory()


# Modify the CSV file paths to use the experiment_results_dir
df_iterative = pd.read_csv(os.path.join(experiment_results_dir, "iterative_refinement_different_partition_results.csv"))
df_HEM = pd.read_csv(os.path.join(experiment_results_dir, "GPP_with_HEM_different_partition_results.csv"))
df_SP = pd.read_csv(os.path.join(experiment_results_dir, "spectral_partitioning_different_partition_results.csv"))


# Calculate the mean edge cuts for each number of partitions
mean_edge_cuts_iterative = df_iterative.groupby('Number of Partitions')['Edge Cuts'].mean()
mean_edge_cuts_HEM = df_HEM.groupby('Number of Partitions')['Edge Cuts'].mean()
mean_edge_cuts_SP = df_SP.groupby('Number of Partitions')['Edge Cuts'].mean()

# Plot the average results for all algorithms
plt.plot(mean_edge_cuts_iterative.index, mean_edge_cuts_iterative.values, '-o', label='Iterative Refinement')
plt.plot(mean_edge_cuts_HEM.index, mean_edge_cuts_HEM.values, '-x', label='HEM')
plt.plot(mean_edge_cuts_SP.index, mean_edge_cuts_SP.values, '-^', label='Spectral Partitioning')

plt.title('Average Edge Cuts vs Number of Partitions (Graph Size = 300)')
plt.xlabel('Number of Partitions')
plt.ylabel('Average Edge Cuts')

# Set x-ticks to the specific partition counts tested
partitions_list = sorted(df_iterative['Number of Partitions'].unique())
plt.xticks(partitions_list)

plt.legend()
plt.grid(True)
plt.show()
