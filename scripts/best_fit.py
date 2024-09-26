def analyze_fit_results(file_path):
    results = {}

    # Read the combined output file
    with open(file_path, 'r') as f:
        for line in f:
            if "KS stat" in line or "p-value" in line:
                parts = line.split(',')
                dist_name = parts[0].split(':')[0].strip()
                p_value = float(parts[1].split('=')[1].strip())
                results[dist_name] = p_value

    # Determine the best fit based on the highest p-value
    best_fit = max(results, key=results.get)
    best_p_value = results[best_fit]

    print(f"Best fitting distribution: {best_fit} with p-value: {best_p_value}")

# Example usage
analyze_fit_results("goodness_of_fit.txt")
