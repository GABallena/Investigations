import pandas as pd
import random

# Assuming the dataset is already loaded in a DataFrame (df)
# Example data loading (replace this with your actual data loading process if needed)
# df = pd.read_csv('your_data.csv')

# List of site names to exclude from random sampling
exclusions = ["Manila Bay - Baseline", "Manila Bay Site 2"]  # Add any sites you want to exclude

# Filter out excluded sites
df_filtered = df[~df["Name"].isin(exclusions)]

# Function to randomize sites with optional stratification and exclusions
def randomize_sites(df, num_groups=2, stratify_by=None, seed=None):
    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Calculate the approximate number of sites per group
    num_sites_per_group = len(df) // num_groups
    remaining_sites = len(df) % num_groups
    
    # If a stratification factor is provided, group sites based on that factor
    if stratify_by and stratify_by in df.columns:
        stratified_groups = df.groupby(stratify_by)
        randomized_groups = {i: [] for i in range(1, num_groups + 1)}

        for _, group in stratified_groups:
            sites = group.sample(frac=1, random_state=seed).to_dict(orient='records')
            for i, site in enumerate(sites):
                # Distribute sites across groups ensuring each group gets approximately equal count
                group_number = (i % num_groups) + 1
                randomized_groups[group_number].append(site)
    else:
        # Random sampling without stratification
        sites = df.sample(frac=1, random_state=seed).to_dict(orient='records')
        randomized_groups = {i: [] for i in range(1, num_groups + 1)}
        
        # Fill groups while ensuring equal distribution
        site_index = 0
        for group_number in range(1, num_groups + 1):
            group_size = num_sites_per_group + (1 if remaining_sites > 0 else 0)
            randomized_groups[group_number].extend(sites[site_index:site_index + group_size])
            site_index += group_size
            remaining_sites -= 1

    return randomized_groups

# Set parameters
num_groups = 4  # Number of groups for random assignment
seed = 42  # Seed for reproducibility
stratify_by = "location"  # Replace with any column like "source", "season", etc., or None for random sampling

# Randomize sites with exclusions and stratification applied
randomized_assignments = randomize_sites(df_filtered, num_groups=num_groups, stratify_by=stratify_by, seed=seed)

# Display the results with stratification values
for group, sites in randomized_assignments.items():
    print(f"\nGroup {group}:")
    for site in sites:
        stratification_value = site.get(stratify_by, "N/A")  # Fetch the stratification value if available
        print(f" - {site['Name']} (X: {site['X']}, Y: {site['Y']}, {stratify_by}: {stratification_value})")
