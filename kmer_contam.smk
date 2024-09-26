GENESTOFIND="link"


rule all:
    input:
        "results/cleaned_spikes_assembled.fasta",
        "results/biological_classification.txt"

# Step 1: Download marker genes
rule download_marker_genes:
    output:
        "marker_genes.fasta"
    shell:
        """
        # Example: Replace this with the actual source of your marker genes
        mkdir -p databases
        aria2 -x 16 ~/.Database/$GENESTOFIND 
        """

# Step 2: Generate k-mers from the marker genes
rule generate_kmers:
    input:
        "marker_genes.fasta"
    output:
        "marker_kmers.fa"
    shell:
        """
        jellyfish count -m 31 -s 100M -C {input} -o marker_kmers.jf
        jellyfish dump marker_kmers.jf > {output}
        """

# Step 3: Map k-mers to raw reads
rule map_kmers:
    input:
        reads="raw_reads.fastq",
        kmers="marker_kmers.fa"
    output:
        "results/kmer_stats.txt"
    shell:
        """
        bbduk.sh in={input.reads} ref={input.kmers} k=31 stats={output}
        """

# Step 4: Normalize SCGs by k-mer using wavelets
rule normalize_scgs_by_kmer:
    input:
        "results/kmer_stats.txt"
    output:
        "results/wavelet_normalized_scgs.txt"
    script:
        "scripts/normalize_scgs_wavelet.py"

# Step 5: Retrieve spikes of interest and save in FASTA format
rule retrieve_spikes_fasta:
    input:
        reads="raw_reads.fastq",
        normalized="results/wavelet_normalized_scgs.txt"
    output:
        "results/spikes_of_interest.fasta"
    script:
        "scripts/extract_spikes_fasta.py"

# Step 6: Assemble k-mers from spikes of interest
rule assemble_kmers:
    input:
        "results/spikes_of_interest.fasta"
    output:
        "results/spikes_assembled.fasta"
    shell:
        """
        megahit -r {input} -o results/spikes_assembly --min-contig-len 200
        mv results/spikes_assembly/final.contigs.fa {output}
        """

# Step 7: Classify assembled sequences (plasmids, repeats, viruses)
rule classify_assembled_sequences:
    input:
        "results/spikes_assembled.fasta"
    output:
        "results/biological_classification.txt"
    shell:
        """
        kraken2 --db /path/to/kraken_db --output {output} --report {output}.report --use-names {input}
        """

# Step 8: Filter contaminants using BBDuk
rule filter_contaminants:
    input:
        assembled="results/spikes_assembled.fasta",
        classification="results/biological_classification.txt"
    output:
        "results/cleaned_spikes_assembled.fasta"
    shell:
        """
        # Extract non-biological sequences from the classification report
        grep 'contaminant' {input.classification} | awk '{{print $1}}' > contaminants_list.txt
        
        # Create a contaminant database for BBDuk
        bbduk.sh in={input.assembled} out={output} ref=contaminants_list.txt k=31 stats=results/bbduk_cleaning_stats.txt
        """
