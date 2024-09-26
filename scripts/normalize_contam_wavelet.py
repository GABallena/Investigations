import numpy as np
import pywt
from scipy.stats import zscore
import sys

# Load k-mer counts for contaminants from the input file
try:
    kmer_counts = np.loadtxt(snakemake.input[0])
except OSError as e:
    print(f"Error loading input file: {e}")
    sys.exit(1)

# Apply continuous wavelet transform (CWT)
wavelet_name = 'morl'
widths = np.arange(1, 20)
coefficients, frequencies = pywt.cwt(kmer_counts, widths, wavelet_name)

# Calculate Z-scores for wavelet coefficients
z_scores = zscore(coefficients, axis=0)

# Set a p-value threshold for significant wavelets (corresponding to p=0.05)
threshold = 1.96  # Z-score for p = 0.05 in a two-tailed test

# Filter out wavelet coefficients with Z-scores below the threshold
significant_coefficients = np.where(np.abs(z_scores) >= threshold, coefficients, 0)

# Calculate normalized contaminants from the significant wavelet coefficients
normalized_contam = np.mean(significant_coefficients, axis=0)

# Save normalized contaminant counts
np.savetxt(snakemake.output[0], normalized_contam)

print("Wavelet-based contaminant normalization with p-value filtering complete.")
