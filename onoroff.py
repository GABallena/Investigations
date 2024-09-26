import pandas as pd
import numpy as np

# Read the DeSeq2 results
deseq_results = pd.read_csv('results/deseq_analysis.txt', sep='\t')

# Apply your half-life model logic
# Example: If the log2 fold change is negative and below a certain threshold, mark the gene as "off"
threshold = -1.0  # Set your threshold for repression (you can adjust this)
deseq_results['status'] = np.where(deseq_results['log2FoldChange'] <= threshold, 'off', 'on')

# Calculate how many genes are "off"
total_genes = len(deseq_results)
off_genes = deseq_results[deseq_results['status'] == 'off']
percentage_off = (len(off_genes) / total_genes) * 100

# Output the threshold information
with open('results/gene_expression_thresholds.txt', 'w') as f:
    f.write(f"{percentage_off}% of genes are off based on the threshold {threshold}.\n")
    f.write(f"Half-life threshold reached when {percentage_off}% of genes are repressed.\n")
