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
