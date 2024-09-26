from scipy.stats import kstest, norm, lognorm, expon, gamma

# Load k-mer counts
kmer_counts = np.loadtxt(snakemake.input[0])

# Fit parameters from the fit_distributions step
with open(snakemake.input[1]) as f:
    params = [eval(line.split(": ")[1]) for line in f.readlines()]

norm_params, lognorm_params, expon_params, gamma_params = params

# Perform KS test for each distribution
ks_stat_norm, p_value_norm = kstest(kmer_counts, 'norm', args=norm_params)
ks_stat_lognorm, p_value_lognorm = kstest(kmer_counts, 'lognorm', args=lognorm_params)
ks_stat_expon, p_value_expon = kstest(kmer_counts, 'expon', args=expon_params)
ks_stat_gamma, p_value_gamma = kstest(kmer_counts, 'gamma', args=gamma_params)

# Write results to a file
with open("goodness_of_fit.txt", 'w') as f:
    f.write(f"Normal: KS stat={ks_stat_norm}, P-value={p_value_norm}\n")
    f.write(f"Log-normal: KS stat={ks_stat_lognorm}, P-value={p_value_lognorm}\n")
    f.write(f"Exponential: KS stat={ks_stat_expon}, P-value={p_value_expon}\n")
    f.write(f"Gamma: KS stat={ks_stat_gamma}, P-value={p_value_gamma}\n")

