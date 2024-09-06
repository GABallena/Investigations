import numpy as np
import pywt

# Load k-mer counts from the input file (results/kmer_stats.txt)
# For simplicity, let's assume this file has one k-mer count per line
kmer_counts = np.loadtxt('results/kmer_stats.txt')

# Apply continuous wavelet transform (CWT) to k-mer counts
wavelet_name = 'morl'
widths = np.arange(1, 20)  # Adjust based on the scales of interest

coefficients, frequencies = pywt.cwt(kmer_counts, widths, wavelet_name)

# Filter out small-scale coefficients (e.g., repetitive or noisy regions)
threshold_scale = 5  # Example threshold for filtering small signals
filtered_coefficients = np.where(widths < threshold_scale, 0, coefficients)

# Use filtered coefficients for normalization (taking mean over all relevant scales)
normalized_scgs = np.mean(filtered_coefficients, axis=0)

# Save normalized SCG counts to output file
np.savetxt('results/wavelet_normalized_scgs.txt', normalized_scgs)

print("Wavelet-based SCG normalization complete.")
