rule check_distribution:
    input:
        "kmer_counts.txt"
    output:
        "distribution_check.txt"
    script:
        "scripts/normality_test.py"

rule fit_alternate_distributions:
    input:
        "distribution_check.txt"
    output:
        "fitted_distributions.txt"
    script:
        "scripts/fit_distributions.py"

rule transform_data_if_needed:
    input:
        "fitted_distributions.txt"
    output:
        "transformed_data.txt"
    script:
        "scripts/goodness_of_fit.py"

# Parametric Path
rule goodness_of_fit_parametric:
    input:
        "fitted_distributions.txt"
    output:
        "goodness_of_fit.txt"
    script:
        "scripts/goodness_of_fit_parametric.py"

rule best_alpha_parametric:
    input:
        "goodness_of_fit.txt"
    output:
        "best_alpha_parametric.txt"
    script:
        "scripts/best_alpha_parametric.py"

# Non-Parametric Path
rule goodness_of_fit_nonparametric:
    input:
        "kmer_counts.txt"
    output:
        "goodness_of_fit.txt"
    script:
        "scripts/goodness_of_fit_nonparametric.py"

rule best_alpha_nonparametric:
    input:
        "goodness_of_fit.txt"
    output:
        "best_alpha_nonparametric.txt"
    script:
        "scripts/best_alpha_nonparametric.py"

# Best Fit Determination
rule best_fit:
    input:
        "goodness_of_fit.txt"
    output:
        "fit_results.tsv"
    script:
        "scripts/best_fit.py"
