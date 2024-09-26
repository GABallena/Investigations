import numpy as np
import pywt
from scipy.stats import zscore

# Load k-mer counts from the input file
kmer_counts = np.loadtxt('results/kmer_stats.txt')

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

# Calculate normalized SCGs from the significant wavelet coefficients
normalized_scgs = np.mean(significant_coefficients, axis=0)

# Save normalized SCG counts
np.savetxt('results/wavelet_normalized_scgs.txt', normalized_scgs)

print("Wavelet-based SCG normalization with p-value filtering complete.")
