import pandas as pd
import sys

# Load the fit results
try:
    fit_results = pd.read_csv(snakemake.input[0], sep='\t')  # Input: fit_results.tsv
except OSError as e:
    print(f"Error loading fit results: {e}")
    sys.exit(1)

# Identify the best-fitting model based on the highest P-value or another criterion
best_fit = fit_results.loc[fit_results['P-value'].idxmax()]

best_model = best_fit['Model']
best_p_value = best_fit['P-value']

# Load the optimal alpha values (assuming you have a file for this)
try:
    alpha_values = pd.read_csv(snakemake.input[1], sep='\t')  # Input: goodness_of_fit.txt or other alpha file
except OSError as e:
    print(f"Error loading alpha values: {e}")
    sys.exit(1)

# Get the optimal alpha for the best-fitting model
optimal_alpha = alpha_values.loc[alpha_values['Model'] == best_model, 'Optimal Alpha'].values[0]

# Save the best model and optimal alpha for the next step
with open(snakemake.output[0], 'w') as f:
    f.write(f"Best fitting model: {best_model}\n")
    f.write(f"Optimal alpha: {optimal_alpha}\n")
