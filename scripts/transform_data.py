import numpy as np
import sys
from scipy.stats import lognorm, expon, gamma

# Load fitted distribution parameters
try:
    fitted_distributions = np.loadtxt(snakemake.input[0])  # Adjust based on your output format
except OSError as e:
    print(f"Error loading fitted distributions: {e}")
    sys.exit(1)

# Check if fitting was successful
# Assuming fitted_distributions contains info on success or failure
if np.all(fitted_distributions[:, 1] > 0):  # Example condition for successful fit
    # Example: Transform data based on the fitted distribution
    if fitted_distributions[0, 0] == 'lognormal':
        # Perform log transformation
        transformed_data = np.log(kmer_counts + 1)  # Adding 1 to avoid log(0)
    elif fitted_distributions[0, 0] == 'exponential':
        # If exponential, consider scaling
        transformed_data = kmer_counts / np.mean(kmer_counts)
    elif fitted_distributions[0, 0] == 'gamma':
        # Perform a gamma transformation (placeholder)
        transformed_data = (kmer_counts / np.mean(kmer_counts)) ** 2  # Example transformation
    else:
        print("No valid transformation identified.")
        sys.exit(1)

    # Save the transformed data
    np.savetxt(snakemake.output[0], transformed_data)

else:
    print("Fitting alternate distributions failed. No transformation applied.")
