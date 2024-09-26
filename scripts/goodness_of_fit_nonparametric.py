import numpy as np
from scipy import stats
import sys

# Load k-mer counts
try:
    kmer_counts = np.loadtxt(snakemake.input[0])
except OSError as e:
    print(f"Error loading input file: {e}")
    sys.exit(1)

# Function to perform goodness-of-fit tests
def goodness_of_fit_tests(data):
    results = {}

    # Kolmogorov-Smirnov Test for Uniform Distribution
    ks_stat, ks_p_value = stats.kstest(data, 'uniform')
    results['KS Test (Uniform)'] = {'Statistic': ks_stat, 'p-value': ks_p_value}

    # Anderson-Darling Test for Uniform Distribution
    ad_stat, critical_values, sig_level = stats.anderson(data, dist='uniform')
    results['Anderson-Darling Test (Uniform)'] = {'Statistic': ad_stat, 'Critical Values': critical_values}

    # Kolmogorov-Smirnov Test for Exponential Distribution
    ks_stat_exp, ks_p_value_exp = stats.kstest(data, 'expon')
    results['KS Test (Exponential)'] = {'Statistic': ks_stat_exp, 'p-value': ks_p_value_exp}

    # Anderson-Darling Test for Exponential Distribution
    ad_stat_exp, critical_values_exp, sig_level_exp = stats.anderson(data, dist='expon')
    results['Anderson-Darling Test (Exponential)'] = {'Statistic': ad_stat_exp, 'Critical Values': critical_values_exp}

    # Bootstrapping for Uniform Distribution
    bootstrapped_means = []
    for _ in range(1000):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(boot_sample))
    results['Bootstrapped Means (Uniform)'] = np.mean(bootstrapped_means)

    # Add similar bootstrapping for other distributions as needed...

    return results

# Run tests
results = goodness_of_fit_tests(kmer_counts)

# Output results to a file
with open("goodness_of_fit.txt", 'w') as f:
    for test, result in results.items():
        f.write(f"{test}: {result}\n")


print("Goodness-of-fit tests completed.")
