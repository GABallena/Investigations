#!/bin/bash

# Ensure the SRA and raw_reads directories exist
mkdir -p SRA
mkdir -p raw_reads

# Loop through all .sra files in the SRA directory
for sra_file in SRA/*.sra; do
    echo "Processing $sra_file..."
    
    # Generate FASTQ files directly in the SRA folder
    fasterq-dump --split-files "$sra_file" --outdir SRA
    
    # Compress the resulting FASTQ files as .fq.gz
    gzip -c SRA/"$(basename "$sra_file" .sra)"_1.fastq > SRA/"$(basename "$sra_file" .sra)"_1.fq.gz
    gzip -c SRA/"$(basename "$sra_file" .sra)"_2.fastq > SRA/"$(basename "$sra_file" .sra)"_2.fq.gz
    
    # Remove the uncompressed FASTQ files
    rm SRA/"$(basename "$sra_file" .sra)"_1.fastq
    rm SRA/"$(basename "$sra_file" .sra)"_2.fastq
    
    # Move the compressed .fq.gz files to the raw_reads folder
    mv SRA/"$(basename "$sra_file" .sra)"_1.fq.gz raw_reads/
    mv SRA/"$(basename "$sra_file" .sra)"_2.fq.gz raw_reads/
    
    echo "Files for $sra_file processed, compressed as .fq.gz, and moved to raw_reads."
done
