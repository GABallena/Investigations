import numpy as np
from scipy import stats
import sys

# Load wavelet-normalized values for contaminants and SCGs
try:
    contaminants = np.loadtxt(snakemake.input.contam)
    scgs = np.loadtxt(snakemake.input.scgs)
except OSError as e:
    print(f"Error loading input files: {e}")
    sys.exit(1)

# Load fitted distribution parameters
try:
    fitted_distributions = np.loadtxt(snakemake.input.fitted_distributions)  # Adjust based on your output format
except OSError as e:
    print(f"Error loading fitted distributions: {e}")
    sys.exit(1)

# Check if fitting was successful
if np.all(fitted_distributions[:, 1] > 0):  # Example condition for successful fit
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(contaminants, scgs)

    # Set a threshold for statistical significance (e.g., p < 0.05)
    significant = np.where(p_value < 0.05, 'significant', 'ambiguous')

    # Save ambiguous sequences to file
    with open(snakemake.output.ambiguous_sequences, 'w') as f:
        for i, label in enumerate(significant):
            if label == 'ambiguous':
                f.write(f"Sequence {i} is ambiguous\n")

    # Save t-test results
    with open(snakemake.output[0], 'w') as f:
        f.write(f"T-statistic: {t_stat}\n")
        f.write(f"P-value: {p_value}\n")
else:
    # Null the results if fitting failed
    with open(snakemake.output[0], 'w') as f:
        f.write("Fitting failed; results are null.\n")
    with open(snakemake.output.ambiguous_sequences, 'w') as f:
        f.write("Fitting failed; no ambiguous sequences to report.\n")
