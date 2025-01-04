#!/bin/bash

# Set BioProject ID
BIOPROJECT="PRJNA1089772"

# Step 1: Fetch run accessions
echo "Fetching run accessions for BioProject $BIOPROJECT..."
esearch -db sra -query "$BIOPROJECT" | efetch -format runinfo | cut -d ',' -f 1 | grep SRR > sra_accessions.txt

# Step 2: Download all runs
echo "Downloading all runs..."
cat sra_accessions.txt | xargs -n 1 prefetch

# Step 3: Convert to FASTQ
echo "Converting SRA files to FASTQ..."
for file in *.sra; do
    fastq-dump --split-files --gzip "$file"
done

# Step 4: Verify files
echo "Validating downloaded files..."
vdb-validate *.sra

echo "Download and conversion complete!"
