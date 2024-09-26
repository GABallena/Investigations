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

rule goodness_of_fit_parametric:
    input:
        "fitted_distributions.txt"
    output:
        "goodness_of_fit.txt" 
    script:
        "scripts/goodness_of_fit.py"  

rule non_parametric_workflow:
    input:
        "fitted_distributions.txt"
    output:
        "non_parametric_results.txt"
    script:
        "scripts/non_parametric_analysis.py"

rule goodness_of_fit_nonparametric:
    input:
        "kmer_counts.txt"
    output:
        "goodness_of_fit.txt"  
    script:
        "scripts/goodness_of_fit_nonparametric.py"

rule best_fit:
    input:
        "kmer_counts.txt"
    output:
        "fit_results.tsv" 
    script:
        "scripts/best_fit.py"