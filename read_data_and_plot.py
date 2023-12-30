import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df_iterative = pd.read_csv("experiment_results/iterative_refinement_results.csv")
df_HEM = pd.read_csv("experiment_results/GPP_with_HEM_results.csv")

# Calculate the average edge cuts for each graph size
average_edge_cuts_iterative = df_iterative.groupby('Graph Size')['Edge Cuts'].mean().reset_index()
average_edge_cuts_HEM = df_HEM.groupby('Graph Size')['Edge Cuts'].mean().reset_index()

# Plot the average edge cuts with error bars for both algorithms
plt.errorbar(average_edge_cuts_iterative['Graph Size'], average_edge_cuts_iterative['Edge Cuts'],
             yerr=df_iterative.groupby('Graph Size')['Edge Cuts'].std(), fmt='-o', capsize=5,
             label='Iterative Refinement')
plt.errorbar(average_edge_cuts_HEM['Graph Size'], average_edge_cuts_HEM['Edge Cuts'],
             yerr=df_HEM.groupby('Graph Size')['Edge Cuts'].std(), fmt='-x', capsize=5, label='Using HEM')
plt.title('Average Edge Cuts vs Graph Size')
plt.xlabel('Graph Size')
plt.ylabel('Average Edge Cuts')
plt.legend()
plt.grid(True)
plt.show()
