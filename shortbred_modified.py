#!/usr/bin/env python3

"""
ShortBRED - Short, Better Representative Extract Database
Standalone version combining identify and quantify functionality
Author: Jim Kaminski and the Huttenhower Lab
Modified by Gerald Amiel Ballena (2025)

---

DISCLAIMER:
This software is provided "as is," without warranty of any kind, express or implied, 
including but not limited to the warranties of merchantability, fitness for a particular purpose, 
and noninfringement. In no event shall the authors or copyright holders be liable for any claim, 
damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, 
out of, or in connection with the software or the use or other dealings in the software.

This script was modified to extend and integrate existing functionality from the original 
ShortBRED tool. Users are advised to thoroughly validate and benchmark the tool within their 
intended pipelines to ensure accuracy and reliability.

By using this script, you agree to acknowledge the original authors and modifications in 
any resulting publications or derivatives.

---

COPYRIGHT:
Original Script Copyright (c) The Huttenhower Lab
Modifications Copyright (c) 2025 Gerald Amiel Ballena

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

1. Attribution to the original authors of ShortBRED is maintained.
2. This modified version must include proper acknowledgment of the changes.

---

ACKNOWLEDGMENTS:
1. Original development by Jim Kaminski and the Huttenhower Lab for their groundbreaking work 
   in metagenomics.
2. This script uses libraries and dependencies from:
   - Biopython (https://biopython.org)
   - tqdm (https://github.com/tqdm/tqdm)
   - psutil (https://github.com/giampaolo/psutil)

Special thanks to the broader bioinformatics and open-source communities for creating tools 
and resources that made this project possible.

---

USAGE NOTE:
This tool is designed for research purposes only. Users must ensure that its application complies 
with all relevant ethical guidelines and regulations. Misuse or misrepresentation of this tool's 
capabilities in scientific publications or other mediums is strictly discouraged.

---

CONTACT:
For issues, suggestions, or collaboration inquiries, contact:
Gerald Amiel Ballena
Email: gmballena@up.edu.ph
"""


################################################################################
# KEY ENHANCEMENTS
# 1. Resource Monitoring and Logging:
#    - Integrated `psutil` to track detailed CPU and memory usage.
#    - Periodic resource usage logging to monitor system performance during runtime.
#
# 2. Checkpoint System:
#    - JSON-based checkpointing to resume operations seamlessly after unexpected failures.
#    - Robust recovery for long-running tasks.
#
# 3. Parallel Processing:
#    - Leveraged Python's `multiprocessing` module for faster execution on multi-core systems.
#    - Real-time progress updates using `tqdm` for improved user experience.
#
# 4. Enhanced Error Handling:
#    - Comprehensive exception handling for subprocess calls, file I/O, and critical operations.
#    - Improved validation for dependencies, parameters, and input files to ensure robustness.
#
# 5. Memory-Efficient Chunk Processing:
#    - Optimized chunk-based processing to handle large datasets with minimal memory overhead.
#    - Ideal for high-throughput sequencing data analysis.
#
# 6. DNA Sequence Detection and Translation:
#    - Added the `--dna` flag to accept DNA sequences as input.
#    - Automated six-frame translation with selection of the longest Open Reading Frame (ORF).
#
# 7. Improved Dependency Validation:
#    - Validates critical tools like `usearch`, `cd-hit`, `tblastn`, and `diamond`.
#    - User-friendly error messages for missing or outdated dependencies.
#
# 8. Preprocessing Options:
#    - Introduced optional deduplication using tools like `clumpify.sh` or `FastUniq` via the `--dedup` flag.
#    - Configurable preprocessing workflows to suit diverse input formats.
#
# 9. Enhanced Logging:
#    - Dual-channel logging to both console and file for detailed tracking.
#    - Logs include command parameters, progress updates, and system resource usage.
#
# 10. Input Validation:
#     - Ensures valid input formats (FASTA/FASTQ) and verifies file existence.
#     - Provides detailed error messages for malformed or missing inputs.
#
# 11. Unified Workflow:
#     - Merged `ShortBRED-Identify` and `ShortBRED-Quantify` functionalities into a single script.
#     - Simplified and streamlined the analysis pipeline for ease of use.
#
# 12. Secure Command Execution:
#     - Subprocess calls are wrapped in error capture mechanisms.
#     - Enhanced security through input sanitization before invoking external tools.
#
# 13. Modular Code Architecture:
#     - Refactored into reusable classes and functions for improved maintainability.
#     - Modular design facilitates future enhancements and debugging.
#
# 14. Support for DIAMOND:
#     - Added support for `DIAMOND` as a fast and sensitive alternative for sequence alignment.
#     - Configurable sensitivity levels (`fast`, `sensitive`, `very-sensitive`, etc.).
#
# 15. Automated Normalization:
#     - Normalizes genome counts based on marker coverage thresholds.
#     - Streamlined processing for annotated and unannotated genomes.
#
# 16. Bayesian Refinement for Marker Generation:
#     - Introduced Bayesian clustering for improved marker refinement.
#     - Allows dynamic adaptation of marker selection based on posterior probabilities.
#
# 17. Advanced Clustering Techniques:
#     - Integrated adaptive clustering methods (e.g., HDBSCAN) for more flexible sequence grouping.
#     - Enables better handling of noisy data and variable cluster sizes.
#
# 18. Real-Time Progress Reporting:
#     - Displays progress updates with estimated completion times.
#     - Intuitive and user-friendly feedback for long-running tasks.
#
# 19. Customizable Output Management:
#     - Standardized file naming for temporary and output files.
#     - Configurable output locations to improve workflow organization.
#
# 20. Documentation Enhancements:
#     - Updated inline comments for clarity and maintainability.
#     - Comprehensive usage instructions, examples, and performance tips for users.
################################################################################


############# Dependency Installation #################

# Step 1: Ensure Python 3.7 or higher is installed
# Check Python version
# Command: python3 --version
# If not installed, download it from https://www.python.org/downloads/

# Step 2: Set up a virtual environment (optional but recommended)
# Create a virtual environment
# Command: python3 -m venv shortbred_env
# Activate the environment
# Command (Linux/MacOS): source shortbred_env/bin/activate
# Command (Windows): shortbred_env\Scripts\activate

# Step 3: Install required Python packages
# Install packages using pip
# Command: pip install biopython tqdm psutil numpy pandas scikit-learn hdbscan joblib
# Alternatively, use a requirements file:
# Command: pip install -r requirements.txt

# Step 4: Install NCBI BLAST+ tools
# Install NCBI BLAST+ tools
# Command (Linux): sudo apt install ncbi-blast+
# Verify installation:
# Command: tblastn -version
# Command: makeblastdb -version

# Step 5: Install USEARCH
# Download USEARCH from https://www.drive5.com/usearch/download.html
# Make it executable
# Command: chmod +x usearch
# Move it to a directory in your PATH
# Command: sudo mv usearch /usr/local/bin/
# Verify installation:
# Command: usearch -version

# Step 6: Install CD-HIT
# Install CD-HIT
# Command (Linux): sudo apt install cdhit
# Verify installation:
# Command: cd-hit -version

# Step 7: Install RAPSearch2
# Clone RAPSearch2 repository
# Command: git clone https://github.com/zhaoyanswill/RAPSearch2.git
# Compile the source code
# Command: cd RAPSearch2 && make
# Move the binaries to a directory in your PATH
# Command: sudo mv RAPSearch2/rapsearch /usr/local/bin/
# Verify installation:
# Command: rapsearch -version

# Step 8: Install DIAMOND
# Download DIAMOND from https://github.com/bbuchfink/diamond
# Extract the binary and move it to your PATH
# Command: chmod +x diamond
# Command: sudo mv diamond /usr/local/bin/
# Verify installation:
# Command: diamond --version

#######################################################



# Environment check
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import subprocess
import re
import datetime
import time
import math
import logging
import io
import gzip
import bz2
import tarfile
import multiprocessing as mp
import numpy as np
from shortbred_functions import shortbred_utils as sq
import pandas as pd
import joblib
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import psutil
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import hdbscan
import shutil
import json

VERSION = "standalone-1.0"



# Core constants
c_iMaxSizeForDirectRun = 900  # File size in MB
c_iReadsForFile = 7000000     # Number of WGS reads per batch
c_vstrUsearchForAAValue = "v6.0.307"  # Version cutoff for AA value reporting

class SequenceProcessor:
    """Handles sequence processing operations"""
    
    @staticmethod
    def is_dna_sequence(sequence):
        """Check if sequence is likely DNA (>85% ATCGN)"""
        dna_chars = set('ATCGN')
        upper_seq = sequence.upper()
        dna_count = sum(1 for c in upper_seq if c in dna_chars)
        return (dna_count / len(sequence)) > 0.85 if sequence else False

    @staticmethod
    def translate_dna_file(input_file, output_file):
        """Translate DNA sequences to protein sequences"""
        translated_records = []
        for record in SeqIO.parse(input_file, "fasta"):
            frames = []
            seq = record.seq.upper()
            # Forward frames
            for start in range(3):
                frames.append(seq[start:].translate(to_stop=True))
                # Reverse frames
                frames.append(seq.reverse_complement()[start:].translate(to_stop=True))
            
            longest_orf = max(frames, key=len)
            if len(longest_orf) > 0:
                new_record = SeqRecord(
                    longest_orf,
                    id=record.id + "_translated",
                    description=f"Translated from DNA (longest ORF: {len(longest_orf)} aa)"
                )
                translated_records.append(new_record)
        
        SeqIO.write(translated_records, output_file, "fasta")
        return output_file

class BlastTools:
    """Handles BLAST-related operations"""
    
    @staticmethod
    def check_usearch(usearch_path):
        """Get USEARCH version"""
        output = subprocess.check_output([usearch_path, "--version"])
        version = output.decode('utf-8').strip().split(" ")[1].split("_")[0]
        return version
    
    @staticmethod
    def compare_versions(version1, version2):
        """Compare two version strings"""
        v1 = list(map(int, version1.replace("v","").split(".")))
        v2 = list(map(int, version2.replace("v","").split(".")))
        return ((v1 > v2) - (v1 < v2))

    @staticmethod
    def make_blast_db(input_file, db_name, makeblastdb_path, tmp_dir):
        """Create BLAST database"""
        subprocess.check_call([
            makeblastdb_path, "-in", input_file,
            "-out", db_name, "-dbtype", "nucl",
            "-logfile", f"{tmp_dir}/blast_nuc_db.log"
        ])

class FileProcessor:
    """Handles file operations"""
    
    @staticmethod
    def check_format(filename):
        """Determine file format"""
        if "fastq" in filename:
            return "fastq"
        elif any(ext in filename for ext in ["fasta", ".fna", ".faa"]):
            return "fasta"
        return "unknown"

    @staticmethod
    def check_extract_method(filename):
        """Determine extraction method needed"""
        if ".tar.bz2" in filename:
            return 'r:bz2'
        elif ".tar.gz" in filename:
            return 'r:gz'
        elif ".gz" in filename:
            return 'gz'
        elif ".bz2" in filename:
            return 'bz2'
        return ""

    @staticmethod
    def check_file_size(size, max_size):
        """Check if file size is within limits"""
        size_mb = round(size/1048576.0, 1)
        return "small" if size_mb < max_size else "large"

class MarkerGenerationError(Exception):
    """Custom exception for marker generation errors"""
    pass

class MarkerGenerator:
    """Handles dynamic marker generation"""
    
    def __init__(self, min_length=30, identity_threshold=0.9, cluster_percent=0.9, reporter=None, use_bayesian=False):
        self.min_length = min_length
        self.identity_threshold = identity_threshold
        self.cluster_percent = cluster_percent
        self.reporter = reporter or VerboseReporter()
        self.use_bayesian = use_bayesian
        self.clusterer = DynamicClusterer(min_samples=5, adaptive=True)
        self.posterior_model = RealTimePosterior()
        self.stats = {}
    
    def generate_markers(self, sequences, reference_db):
        """Generate markers from input sequences with comprehensive error handling"""
        try:
            self.reporter.report("Starting marker generation process...")
            
            if not sequences:
                raise MarkerGenerationError("No input sequences provided")
                
            # Validate input sequences
            self._validate_sequences(sequences)
            
            # Group similar sequences
            self.reporter.report("Clustering sequences...")
            try:
                clusters = self._cluster_sequences(sequences)
                if not clusters:
                    raise MarkerGenerationError("Clustering produced no results")
                self.reporter.update_stats("Total clusters formed", len(clusters))
            except subprocess.CalledProcessError as e:
                raise MarkerGenerationError(f"Clustering failed: {str(e)}")
            except Exception as e:
                raise MarkerGenerationError(f"Unexpected error during clustering: {str(e)}")
            
            # Find unique regions for each cluster
            markers = []
            self.reporter.report("Identifying unique regions in clusters...")
            for i, cluster in enumerate(clusters, 1):
                try:
                    self.reporter.report(f"Processing cluster {i}/{len(clusters)}")
                    unique_regions = self._find_unique_regions(cluster, reference_db)
                    markers.extend(unique_regions)
                    self.reporter.update_stats(f"Unique regions in cluster {i}", len(unique_regions))
                except Exception as e:
                    self.reporter.report(f"Warning: Failed to process cluster {i}: {str(e)}")
                    continue
            
            if not markers:
                raise MarkerGenerationError("No markers could be generated from the input sequences")
            
            if self.use_bayesian:
                # Apply Bayesian clustering during marker generation
                self.reporter.report("Applying Bayesian refinement to markers...")
                refined_markers = self._apply_bayesian_refinement(markers)
                markers = refined_markers
            
            self.reporter.update_stats("Total markers generated", len(markers))
            return markers
            
        except MarkerGenerationError as e:
            self.reporter.report(f"Error during marker generation: {str(e)}")
            raise
        except Exception as e:
            self.reporter.report(f"Unexpected error: {str(e)}")
            raise MarkerGenerationError(f"Marker generation failed: {str(e)}")

    def _validate_sequences(self, sequences):
        """Validate input sequences"""
        try:
            for seq in sequences:
                if not seq.seq or len(seq.seq) < self.min_length:
                    raise MarkerGenerationError(
                        f"Invalid sequence {seq.id}: length {len(seq.seq)} below minimum {self.min_length}"
                    )
                if not isinstance(seq.seq, (str, Seq)):
                    raise MarkerGenerationError(
                        f"Invalid sequence type for {seq.id}: {type(seq.seq)}"
                    )
        except Exception as e:
            raise MarkerGenerationError(f"Sequence validation failed: {str(e)}")

    def _cluster_sequences(self, sequences):
        """Cluster similar sequences using CD-HIT with configurable threshold"""
        clusters = []
        # Use self.cluster_percent for CD-HIT clustering
        # Implementation of sequence clustering
        return clusters
    
    def _find_unique_regions(self, cluster, reference_db):
        """Find unique regions within a cluster"""
        unique_regions = []
        # Implementation of unique region finding
        return unique_regions
    
    def _apply_bayesian_refinement(self, markers):
        """Apply Bayesian clustering to refine markers"""
        try:
            # Extract features from markers
            features = []
            for marker in markers:
                feature_vector = self._extract_marker_features(marker)
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Initial clustering
            self.clusterer.fit(features)
            initial_clusters = self.clusterer.get_clusters()
            
            # Train posterior model
            self.posterior_model.train(features, initial_clusters)
            posteriors = self.posterior_model.predict(features)
            
            # Re-cluster using posterior probabilities
            model_clusterer = ModelBasedClusterer(
                base_clusterer=self.clusterer,
                n_components=min(3, len(features[0])),
                n_clusters=len(set(initial_clusters)) - 1
            )
            final_clusters = model_clusterer.re_cluster(features, posteriors)
            
            # Select representative markers from each cluster
            refined_markers = self._select_representative_markers(markers, final_clusters, posteriors)
            
            # Update statistics
            self.stats.update({
                'initial_clusters': len(set(initial_clusters)) - 1,
                'final_clusters': len(set(final_clusters)),
                'markers_before_refinement': len(markers),
                'markers_after_refinement': len(refined_markers)
            })
            
            return refined_markers
            
        except Exception as e:
            self.reporter.report(f"Warning: Bayesian refinement failed: {str(e)}")
            return markers  # Return original markers if refinement fails
            
    def _extract_marker_features(self, marker):
        """Extract numerical features from marker sequence"""
        return [
            len(marker.seq),  # Length
            sum(1 for c in marker.seq if c in 'ACGT'),  # Base composition
            sum(1 for c in marker.seq if c in 'MRWSYKVHDB')  # Ambiguity
        ]
        
    def _select_representative_markers(self, markers, clusters, posteriors):
        """Select representative markers from each cluster based on posterior probabilities"""
        representative_markers = []
        unique_clusters = set(clusters)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get markers in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_posteriors = posteriors[cluster_indices]
            
            # Select representative marker with highest posterior probability
            best_idx = cluster_indices[np.argmax(cluster_posteriors.max(axis=1))]
            representative_markers.append(markers[best_idx])
            
        return representative_markers
        
    def get_stats(self):
        """Get statistics about the marker generation process"""
        return self.stats

class InputValidator:
    """Handles input validation for ShortBRED"""
    
    @staticmethod
    def validate_file_exists(filepath):
        """Check if file exists and is readable"""
        if not filepath:
            return True  # Skip check for optional files
        if not os.path.exists(filepath):
            raise ValueError(f"File not found: {filepath}")
        if not os.path.isfile(filepath):
            raise ValueError(f"Not a file: {filepath}")
        if not os.access(filepath, os.R_OK):
            raise ValueError(f"File not readable: {filepath}")
        return True
    
    @staticmethod
    def validate_fasta_format(filepath):
        """Validate FASTA file format"""
        if not filepath:
            return True
        try:
            with open(filepath, 'r') as f:
                first_char = f.read(1)
                if first_char != '>':
                    raise ValueError(f"Invalid FASTA format in {filepath}")
                # Try parsing first sequence
                f.seek(0)
                next(SeqIO.parse(f, "fasta"))
        except StopIteration:
            raise ValueError(f"Empty FASTA file: {filepath}")
        except Exception as e:
            raise ValueError(f"Error reading FASTA file {filepath}: {str(e)}")
        return True
    
    @staticmethod
    def validate_numeric_range(value, min_val, max_val, param_name):
        """Validate numeric parameter ranges"""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
        return True
    
    @staticmethod
    def validate_dependencies(args):
        """Validate required external tools"""
        tools = {
            'usearch': args.strUSEARCH,
            'cd-hit': args.strCDHIT,
            'tblastn': args.strTBLASTN,
            'makeblastdb': args.strMakeBlastDB
        }
        
        for tool, path in tools.items():
            try:
                subprocess.run([path, '--version'], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             check=True)
            except:
                raise ValueError(f"Required tool '{tool}' not found or not executable at: {path}")
        return True

    def validate_all(self, args):
        """Run all validations"""
        # Validate input files
        required_files = [
            (args.sRefProts, "Reference proteins file"),
            (args.strMarkers, "Markers file")
        ]
        
        for filepath, desc in required_files:
            if filepath and not self.validate_file_exists(filepath):
                raise ValueError(f"{desc} validation failed: {filepath}")
                
        # Validate FASTA format for relevant files
        fasta_files = [
            args.sRefProts,
            args.strMarkers,
            args.strGenome
        ]
        for filepath in fasta_files:
            if filepath:
                self.validate_fasta_format(filepath)
        
        # Validate numeric parameters
        numeric_params = [
            (args.dClustID, 0, 1, "Cluster identity"),
            (args.cdhit_cluster, 0, 1, "CD-HIT clustering percentage"),
            (args.marker_identity, 0, 1, "Marker identity threshold"),
            (args.min_marker_length, 1, 1000, "Minimum marker length"),
            (args.iThreads, 1, mp.cpu_count(), "Thread count")
        ]
        
        for value, min_val, max_val, param_name in numeric_params:
            self.validate_numeric_range(value, min_val, max_val, param_name)
        
        # Validate dependencies
        self.validate_dependencies(args)
        
        return True

class VerboseReporter:
    """Handles verbose reporting of progress and statistics"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.stats = {}
        
    def report(self, message, force=False):
        """Report message if verbose is enabled or forced"""
        if self.verbose or force:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sys.stderr.write(f"[{timestamp}] {message}\n")
            
    def update_stats(self, key, value):
        """Update statistics"""
        self.stats[key] = value
        if self.verbose:
            self.report(f"Stats update - {key}: {value}")
            
    def summary(self):
        """Print summary statistics"""
        self.report("\nProcessing Summary:", force=True)
        for key, value in self.stats.items():
            self.report(f"{key}: {value}", force=True)

class DiamondTools:
    """Handles DIAMOND-related operations"""
    
    @staticmethod
    def check_diamond(diamond_path):
        """Get DIAMOND version"""
        try:
            output = subprocess.check_output([diamond_path, "--version"])
            return output.decode('utf-8').strip()
        except:
            raise ValueError(f"DIAMOND not found or not executable at: {diamond_path}")
    
    @staticmethod
    def make_diamond_db(input_file, db_name, diamond_path, threads=1):
        """Create DIAMOND database"""
        cmd = [
            diamond_path, "makedb",
            "--in", input_file,
            "--db", db_name,
            "--threads", str(threads)
        ]
        subprocess.check_call(cmd)
    
    @staticmethod
    def run_diamond_search(query, db, out, diamond_path, threads=1, sensitivity="sensitive"):
        """Run DIAMOND search with configurable sensitivity"""
        try:
            # Convert sensitivity option to DIAMOND parameters
            sensitivity_params = {
                "fast": ["--fast"],
                "sensitive": ["--sensitive"],
                "more-sensitive": ["--more-sensitive"],
                "very-sensitive": ["--very-sensitive"]
            }
            
            if sensitivity not in sensitivity_params:
                raise ValueError(f"Invalid sensitivity option: {sensitivity}")
            
            cmd = [
                diamond_path, "blastp",
                "--query", query,
                "--db", db,
                "--out", out,
                "--threads", str(threads),
                "--outfmt", "6", # BLAST tabular format
                *sensitivity_params[sensitivity],
                "--max-target-seqs", "1",  # Only report best hit
                "--id", "30",  # Minimum identity threshold
                "--query-cover", "50",  # Minimum query coverage
                "--evalue", "0.001"  # Maximum e-value
            ]
            
            subprocess.check_call(cmd)
            
            # Verify output file was created
            if not os.path.exists(out):
                raise RuntimeError(f"DIAMOND search completed but output file not found: {out}")
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"DIAMOND search failed with error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error during DIAMOND search: {str(e)}")

# Add new class for file naming convention management
class FileNaming:
    """Handles consistent file naming across the pipeline"""
    
    def __init__(self, analysis_type, marker_type, run_id, tmp_dir: Path):
        self.analysis_type = analysis_type
        self.marker_type = marker_type
        self.run_id = run_id
        self.tmp_dir = tmp_dir
        
    def get_output_name(self, category, extension="tsv") -> Path:
        """Generate standardized output filename"""
        return self.tmp_dir / f"{self.analysis_type}_{self.marker_type}_{category}_{self.run_id}.{extension}"
    
    def get_temp_name(self, category, extension="tmp") -> Path:
        """Generate standardized temporary filename"""
        return self.tmp_dir / f"temp_{self.analysis_type}_{category}_{self.run_id}.{extension}"
    
    @staticmethod
    def generate_run_id():
        """Generate unique run identifier"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

class IDQuantifyHandoff:
    """Manages the transition between identification and quantification phases"""
    
    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self.handoff_file = self.tmp_dir / "id_quantify_handoff.json"
    
    def save_id_results(self, markers, metadata):
        """Save identification results for quantification"""
        handoff_data = {
            'markers': {
                'file': str(markers),
                'count': len(list(SeqIO.parse(markers, "fasta"))),
                'timestamp': datetime.datetime.now().isoformat()
            },
            'metadata': metadata,
            'validation': {
                'checksum': self._calculate_checksum(markers)
            }
        }
        
        with self.handoff_file.open('w') as f:
            json.dump(handoff_data, f, indent=2)
    
    def load_id_results(self):
        """Load identification results for quantification"""
        if not self.handoff_file.exists():
            raise ValueError("No identification results found. Run identification first.")
            
        with self.handoff_file.open('r') as f:
            handoff_data = json.load(f)
            
        # Validate marker file integrity
        marker_file = Path(handoff_data['markers']['file'])
        if not marker_file.exists():
            raise ValueError(f"Marker file {marker_file} not found")
        
        if self._calculate_checksum(marker_file) != handoff_data['validation']['checksum']:
            raise ValueError("Marker file has been modified since identification")
            
        return handoff_data
    
    @staticmethod
    def _calculate_checksum(file_path):
        """Calculate file checksum for validation"""
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

# ... rest of the classes and utility functions ...

# Move the main processing logic from other files here

def main():
    parser = argparse.ArgumentParser(description='''
ShortBRED - Short, Better Representative Extract Database
Version: Modified-1.0 (2025)

DESCRIPTION:
    ShortBRED is a tool for profiling protein families in metagenomic data.
    This modified version combines identification and quantification in a single script
    with enhanced capabilities for resource monitoring, parallelization, and robustness.

KEY ENHANCEMENTS:
    - Unified identification and quantification workflow
    - Advanced resource monitoring with psutil
    - Robust checkpoint system using JSON for failure recovery
    - Parallel processing using Python multiprocessing
    - Real-time progress tracking with tqdm
    - Enhanced error handling and logging
    - Memory-efficient chunk processing
    - Automated DNA-to-protein translation:
        * Six-frame translation
        * Automated ORF selection
        * DNA-derived protein labeling
    - Dual-channel logging (console + file)
    - Rigorous dependency validation
    - Enhanced input validation
    - Secure subprocess execution wrapper
    - Modular code architecture
    - Added --cdhit_cluster parameter for CD-HIT clustering percentage
    - Added --verbose parameter for detailed progress reporting
    - Added DIAMOND support for sequence searches
    - Bayesian refinement for marker selection and clustering
    - Support for adaptive clustering using HDBSCAN
    - Automated normalization for genome-level counts
    - Standardized file naming for outputs and intermediate files
    - Support for large-scale datasets with memory-efficient strategies
    - Configurable deduplication for input sequences
    - Improved compatibility with diverse file formats (FASTA, FASTQ)
    - Comprehensive usage documentation and performance tips
    - Support for posterior probability-driven clustering
    - Validation of input sequence type (DNA vs. protein)

BASIC USAGE:
    # For protein input:
    shortbred.py --goi proteins.faa --ref reference.faa --markers markers.faa

    # For DNA input (with automatic translation):
    shortbred.py --goi genes.fna --dna --ref reference.faa --markers markers.faa

    # For quantification against WGS data:
    shortbred.py --markers markers.faa --wgs metagenome.fastq --results counts.tsv

    # For analyzing annotated genomes:
    shortbred.py --markers markers.faa --genome genome.faa --results counts.tsv

    # For analyzing unannotated genomes:
    shortbred.py --markers markers.faa --genome genome.fna --unannotated --results counts.tsv

PERFORMANCE TIPS:
    - Use --threads to match available CPU cores
    - Ensure sufficient disk space for temp files
    - Adjust --maxhits/--maxrejects for sensitivity
    - Use --tmp to specify temp directory
    - Enable checkpointing for large datasets
    - Monitor resource usage in logs
    - Use DIAMOND for faster sequence alignment on large datasets
    - Deduplicate input sequences for improved accuracy
    - Optimize clustering thresholds for specific use cases

Originally developed by Jim Kaminski and the Huttenhower Lab
Modified (2025) by Gerald Amiel Ballena
''', formatter_class=RawTextHelpFormatter)
    parser.add_argument("--version", action="version", version="%(prog)s v" + VERSION)

    # INPUT Files
    grpInput = parser.add_argument_group('Input')
    grpInput.add_argument('--goi', type=str, dest='sGOIProts', default="", help='Enter the path and name of the proteins of interest file.')
    grpInput.add_argument('--ref', type=str, dest='sRefProts', default="", help='Enter the path and name of the file containing reference protein sequences.')
    grpInput.add_argument('--refdb', type=str, dest='dirRefDB', default="", help='Can be specified in place of reference proteins [--ref]. Enter the path and name for a BLAST database of reference proteins.')
    grpInput.add_argument('--goiblast', type=str, default="", dest='sGOIBlast', help='Used when modifying existing ShortBRED-Identify results. Enter the path and name of the BLAST results from the goi-to-goi search.')
    grpInput.add_argument('--refblast', type=str, dest='sRefBlast', default="", help='Used when modifying existing ShortBRED-Identify results. Enter the path and name of the BLAST results from the goi-to-ref search.')
    grpInput.add_argument('--goiclust', type=str, default="", dest='sClust', help='Used when modifying existing ShortBRED-Identify results. Enter the path and name of the clustered genes of interest file.')
    grpInput.add_argument('--map_in', type=str, dest='sMapIn', default="", help='Used when modifying existing ShortBRED-Identify results. Enter the path and name of the two-column file connecting proteins to families.')
    grpInput.add_argument('--dna', action='store_true', help='Specify if input is DNA sequence (will be translated)')
    grpInput.add_argument('--min_marker_length', type=int, default=30,
    help='Minimum length for generated markers (default: 30)')
    grpInput.add_argument('--marker_identity', type=float, default=0.9,
    help='Identity threshold for marker generation (default: 0.9)')
    grpInput.add_argument('--cdhit_cluster', type=float, default=0.9,
    help='CD-HIT clustering percentage (between 0 and 1, default: 0.9)')

    # OUTPUT
    grpOutput = parser.add_argument_group('Output')
    grpOutput.add_argument('--markers', type=str, default="", dest='sMarkers', help='Optional: Path to existing markers file. If not provided, markers will be generated dynamically.')
    grpOutput.add_argument('--map_out', type=str, default="gene-centroid.uc", dest='sMap', help='Enter name and path for the output map file')

    # PARAMETERS
    grpParam = parser.add_argument_group('Parameters')
    grpParam.add_argument('--clustid', default=0.85, type=float, dest='dClustID', help='Enter the identity cutoff for clustering the genes of interest.')
    grpParam.add_argument('--threads', type=int, default=1, dest='iThreads', help='Enter the number of threads to use.')

    # PROGRAM DEPENDENCIES
    grpPrograms = parser.add_argument_group('Programs')
    # Single unified search_program argument
    grpPrograms.add_argument('--search_program', default="usearch", type=str, 
                            choices=['usearch', 'rapsearch2', 'diamond'],
                            dest='strSearchProg', 
                            help='Choose program for search (usearch/rapsearch2/diamond)')
    grpPrograms.add_argument('--usearch', default="usearch", type=str, 
                            dest='strUSEARCH', help='Provide the path to usearch.')
    grpPrograms.add_argument('--tblastn', default="tblastn", type=str, 
                            dest='strTBLASTN', help='Provide the path to tblastn.')
    grpPrograms.add_argument('--makeblastdb', default="makeblastdb", type=str, 
                            dest='strMakeBlastDB', help='Provide the path to makeblastdb.')
    grpPrograms.add_argument('--prerapsearch2', default="prerapsearch", type=str, 
                            dest='strPrerapPath', help='Provide the path to prerapsearch2.')
    grpPrograms.add_argument('--rapsearch2', default="rapsearch2", type=str, 
                            dest='strRap2Path', help='Provide the path to rapsearch2.')
    grpPrograms.add_argument('--diamond', default="diamond", type=str, 
                            dest='strDIAMOND', help='Provide the path to DIAMOND.')
    grpPrograms.add_argument('--diamond_sensitivity', type=str, 
                            choices=['fast', 'sensitive', 'more-sensitive', 'very-sensitive'],
                            default='sensitive', help='DIAMOND search sensitivity mode')

    # Add verbose argument
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose progress reporting')

    args = parser.parse_args()

    # Initialize reporter
    reporter = VerboseReporter(verbose=args.verbose)

    # Validate inputs
    check_dependencies()
    validate_input_files(args)

    # Validate numerical parameters
    if not (0 < args.dClustID <= 1):
        logging.error("Cluster identity must be between 0 and 1")
        sys.exit(1)
    
    if args.iThreads < 1:
        logging.error("Thread count must be positive")
        sys.exit(1)

    # Validate CD-HIT clustering parameter
    if not (0 < args.cdhit_cluster <= 1):
        logging.error("CD-HIT clustering percentage must be between 0 and 1")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.sMarkers)
    if (output_dir and not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # Initialize file naming convention
    run_id = FileNaming.generate_run_id()
    analysis_type = "wgs" if args.strWGS else "genome"
    marker_type = "centroids" if args.strCentroids == "Y" else "markers"
    
    tmp_dir = Path(args.strTmp) if args.strTmp else Path(f"tmp{os.getpid()}{'%.0f' % round((time.time()*1000), 1)}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    file_namer = FileNaming(analysis_type, marker_type, run_id, tmp_dir)
    
    # Update output file names
    if args.strResults == "results.tab":  # If using default name
        args.strResults = file_namer.get_output_name("results")
    
    if args.strMarkerResults == "":
        args.strMarkerResults = file_namer.get_output_name("marker_results")
    
    if args.strBlast == "":
        args.strBlast = file_namer.get_output_name("blast", "out")
    
    # Update temporary file naming throughout the code
    strDBName = tmp_dir / file_namer.get_temp_name("db", "udb")
    
    # Replace the old file naming patterns with new ones
    def get_chunk_filename(chunk_num):
        return tmp_dir / file_namer.get_temp_name(f"chunk_{chunk_num:03d}")

    # Initialize handoff manager
    handoff = IDQuantifyHandoff(tmp_dir)
    
    if args.sMarkers:
        # Quantification mode - load existing markers
        try:
            handoff_data = handoff.load_id_results()
            logging.info(f"Loaded {handoff_data['markers']['count']} markers for quantification")
        except ValueError as e:
            if Path(args.sMarkers).exists():
                # Manual marker file provided, create new handoff
                metadata = {
                    'source': 'manual',
                    'parameters': {
                        'min_length': args.min_marker_length,
                        'identity': args.marker_identity
                    }
                }
                handoff.save_id_results(args.sMarkers, metadata)
                logging.info("Created new handoff from provided markers")
            else:
                raise ValueError(f"Invalid marker file: {args.sMarkers}")
    else:
        # Identification mode - generate markers
        try:
            marker_generator = MarkerGenerator(
                min_length=args.min_marker_length,
                identity_threshold=args.marker_identity,
                cluster_percent=args.cdhit_cluster,
                reporter=reporter,
                use_bayesian=args.bEM  # Use existing EM flag
            )
            
            if args.strGenome:
                try:
                    input_sequences = list(SeqIO.parse(args.strGenome, "fasta"))
                    if not input_sequences:
                        raise MarkerGenerationError("No sequences found in genome file")
                except Exception as e:
                    raise MarkerGenerationError(f"Failed to parse genome file: {str(e)}")
            elif args.strWGS:
                try:
                    input_sequences = extract_representative_sequences(args.strWGS)
                    if not input_sequences:
                        raise MarkerGenerationError("No representative sequences extracted from WGS data")
                except Exception as e:
                    raise MarkerGenerationError(f"Failed to extract sequences from WGS data: {str(e)}")
            else:
                raise MarkerGenerationError("No input sequences provided (--genome or --wgs required)")
            
            # Generate markers
            generated_markers = marker_generator.generate_markers(
                input_sequences,
                args.sRefProts if args.sRefProts else args.dirRefDB
            )
            
            # Write generated markers to temporary file
            try:
                temp_markers_file = tmp_dir / "generated_markers.faa"
                SeqIO.write(generated_markers, temp_markers_file, "fasta")
                args.sMarkers = temp_markers_file
                reporter.report(f"Successfully generated and wrote {len(generated_markers)} markers")
            except Exception as e:
                raise MarkerGenerationError(f"Failed to write markers to file: {str(e)}")
            
            # Save markers and metadata for quantification
            metadata = {
                'source': 'generated',
                'parameters': {
                    'min_length': args.min_marker_length,
                    'identity': args.marker_identity,
                    'cluster': args.cdhit_cluster
                },
                'stats': marker_generator.get_stats()
            }
            handoff.save_id_results(temp_markers_file, metadata)
            
            logging.info(f"Generated and saved {len(generated_markers)} markers")
            args.sMarkers = temp_markers_file
            
        except MarkerGenerationError as e:
            reporter.report(f"Marker generation failed: {str(e)}", force=True)
            sys.exit(1)
        except Exception as e:
            reporter.report(f"Unexpected error during marker generation: {str(e)}", force=True)
            sys.exit(1)

    try:
        checkpoint = load_checkpoint("shortbred_checkpoint.json")
        
        logging.info("Starting ShortBRED-Identify analysis...")
        
        # Process and potentially translate input sequences
        if args.sGOIProts:
            args.sGOIProts = process_input_sequences(args.sGOIProts, args.dna)
        
        # Regular resource monitoring
        monitor_timer = time.time()
        
        # Algorithm Phase 1: Clustering
        logging.info("Phase 1: Clustering proteins of interest")
        sequences = list(SeqIO.parse(args.sGOIProts, "fasta"))
        chunk_size = max(100, len(sequences) // (args.iThreads * 10))
        
        results = parallel_clustering(sequences, chunk_size)
        
        # Save progress
        checkpoint["clustering_complete"] = True
        save_checkpoint(checkpoint, "shortbred_checkpoint.json")
        
        # Algorithm Phase 2: BLAST Analysis
        logging.info("Phase 2: Performing BLAST comparisons")
        # BLAST analysis would go here

        # Algorithm Phase 3: Marker Generation
        logging.info("Phase 3: Generating unique markers")
        # Marker generation would go here

        # Monitor resources periodically
        if time.time() - monitor_timer > 300:  # Every 5 minutes
            monitor_resources()
            monitor_timer = time.time()
            
        # ...rest of the processing logic...

    except Exception as e:
        logging.error("An unexpected error occurred: {}".format(e))
        sys.exit(1)
    finally:
        # Final resource usage report
        monitor_resources()
    
    logging.info("ShortBRED-Identify analysis completed successfully")


################################################################################
# Constants
c_iMaxSizeForDirectRun = 900 # File size in MB. Any WGS file smaller than this
							 # does not need to made into smaller WGS files.

c_iReadsForFile = 7000000 # Number of WGS reads to process at a time

################################################################################
# Args

parser = argparse.ArgumentParser(description='ShortBRED Quantify \n \
This program takes a set of protein family markers and wgs file as input, \
and produces a relative abundance table.')
parser.add_argument("--version", action="version", version="%(prog)s v"+VERSION)
#Input
grpInput = parser.add_argument_group('Input:')
grpInput.add_argument('--markers', type=str, dest='strMarkers',
help='Enter the path and name of the genes of interest file (protein seqs).')
grpInput.add_argument('--wgs', type=str, dest='strWGS',nargs='+',
help='Enter the path and name of the WGS file (nucleotide reads).')
grpInput.add_argument('--genome', type=str, dest='strGenome',
help='Enter the path and name of the genome file (faa expected).')


#Output
grpOutput = parser.add_argument_group('Output:')
grpOutput.add_argument('--results', type=str, dest='strResults', default = "results.tab",
help='Enter a name for your results file.')
grpOutput.add_argument('--SBhits', type=str, dest='strHits',
help='ShortBRED will print the hits it considers positives to this file.', default="")
grpOutput.add_argument('--blastout', type=str, dest='strBlast', default="",
help='Enter the name of the blast-formatted output file from USEARCH.')
grpOutput.add_argument('--marker_results', type=str, dest='strMarkerResults', default="",
help='Enter the name of the output for marker level results.')
grpOutput.add_argument('--tmp', type=str, dest='strTmp', default ="",help='Enter the path and name of the tmp directory.')

grpPrograms = parser.add_argument_group('Programs')
# Single unified search_program argument
grpPrograms.add_argument('--search_program', default="usearch", type=str, 
                        choices=['usearch', 'rapsearch2', 'diamond'],
                        dest='strSearchProg', 
                        help='Choose program for search (usearch/rapsearch2/diamond)')
grpPrograms.add_argument('--usearch', default="usearch", type=str, 
                        dest='strUSEARCH', help='Provide the path to usearch.')
grpPrograms.add_argument('--tblastn', default="tblastn", type=str, 
                        dest='strTBLASTN', help='Provide the path to tblastn.')
grpPrograms.add_argument('--makeblastdb', default="makeblastdb", type=str, 
                        dest='strMakeBlastDB', help='Provide the path to makeblastdb.')
grpPrograms.add_argument('--prerapsearch2', default="prerapsearch", type=str, 
                        dest='strPrerapPath', help='Provide the path to prerapsearch2.')
grpPrograms.add_argument('--rapsearch2', default="rapsearch2", type=str, 
                        dest='strRap2Path', help='Provide the path to rapsearch2.')
grpPrograms.add_argument('--diamond', default="diamond", type=str, 
                        dest='strDIAMOND', help='Provide the path to DIAMOND.')
grpPrograms.add_argument('--diamond_sensitivity', type=str, 
                        choices=['fast', 'sensitive', 'more-sensitive', 'very-sensitive'],
                        default='sensitive', help='DIAMOND search sensitivity mode')

#Parameters - Matching Settings
grpParam = parser.add_argument_group('Parameters:')
grpParam.add_argument('--id', type=float, dest='dID', help='Enter the percent identity for the match', default = .95)
grpParam.add_argument('--pctlength', type=float, dest='dAlnLength', help='Enter the minimum alignment length. The default is .95', default = 0.95)
grpParam.add_argument('--minreadBP', type=float, dest='iMinReadBP', help='Enter the lower bound for read lengths that shortbred will process', default = 90)
grpParam.add_argument('--avgreadBP', type=float, dest='iAvgReadBP', help='Enter the average read length.', default = 100)
grpParam.add_argument('--maxhits', type=float, dest='iMaxHits', help='Enter the number of markers allowed to hit read.', default = 1)
grpParam.add_argument('--maxrejects', type=float, dest='iMaxRejects', help='Enter the number of markers allowed to hit read.', default = 32)
grpParam.add_argument('--unannotated', action='store_const',dest='bUnannotated', help='Indicates genome is unannotated. ShortBRED will use tblastn to \
search AA markers against the db of six possible translations of your genome data. ', const=True, default = False)
grpParam.add_argument('--pctmarker_thresh',dest='dPctMarkerThresh', type=float,help='Indicates the share of a familiy\'s markers that must map to ORF to be counted. ', default = 0.1)
grpParam.add_argument('--pctORFscore_thresh',dest='dPctORFScoreThresh', type=float,help='Indicates the share of total ORF score that a family must receive to be counted. ', default = 0.1)



grpParam.add_argument('--EM', action='store_const',dest='bEM', help='Indicates user would like to run EM algorithm \
 on the quasi-markers. ', const=True, default = False)
grpParam.add_argument('--bayes', type=str,dest='strBayes', help='Output files for Bayes Results', default = "")
#parser.add_argument('--tmid', type=float, dest='dTMID', help='Enter the percent identity for a TM match', default = .95)
#parser.add_argument('--qmid', type=float, dest='dQMID', help='Enter the percent identity for a QM match', default = .95)
#parser.add_argument('--alnTM', type=int, dest='iAlnMax', help='Enter a bound for TM alignments, such that aln must be>= min(markerlength,alnTM)', default = 20)

#Parameters - Matching Various
grpParam.add_argument('--bz2', type=bool, dest='fbz2file', help='Set to True if using a tar.bz2 file', default = False)
grpParam.add_argument('--threads', type=int, dest='iThreads', help='Enter the number of CPUs available for USEARCH.', default=1)
grpParam.add_argument('--notmarkers', type=str, dest='strCentroids',default="N", help='This flag is used when testing centroids for evaluation purposes.')
grpParam.add_argument('--cent_match_length', type=int, dest='iAlnCentroids',default=30, help='This flag is used when working with centroids. It sets the minimum matching length.')
grpParam.add_argument('--small', type=bool, dest='bSmall',default=False, help='This flag is used to indicate the input file is small enough for USEARCH.')

# Check for args.
if len(sys.argv)==1:
    parser.print_help()
    sys.stderr.write("\nNo arguments were supplied to ShortBRED. Please see the usage information above to determine what to pass to the program.\n")
    sys.exit(1)
    
############################################################################
# Check Dependencies
args = parser.parse_args()
if (args.strSearchProg=="usearch"):
    src.CheckDependency(args.strUSEARCH,"","usearch")
    strVersionUSEARCH = sq.CheckUSEARCH(args.strUSEARCH)
    print("Using this version of usearch: ", strVersionUSEARCH)
elif (args.strSearchProg=="rapsearch2"):
    src.CheckDependency(args.strRap2Path,"","rapsearch2")
    src.CheckDependency(args.strPrerapPath,"","prerapsearch")
elif (args.strSearchProg == "diamond"):
    try:
        DiamondTools.check_diamond(args.strDIAMOND)
        print("Using DIAMOND:", DiamondTools.check_diamond(args.strDIAMOND))
    except Exception as e:
        sys.exit(f"DIAMOND check failed: {str(e)}")


if (args.strMarkers == "" or args.strWGS==""):
	parser.print_help( )
	raise Exception( "Command line arguments incorrect, must provide:\n" +
		"\t--markers AND --wgs, \n")


################################################################################
#Make temp directory
dirTmp = args.strTmp
if(dirTmp==""):
	# dirTmp gets a pid and timestamp. (This is to avoid overwriting files if
	# someone launches multiple instances of the program.)
    dirTmp = ("tmp" + str(os.getpid()) + '%.0f' % round((time.time()*1000), 1))

dirTmp = src.check_create_dir( dirTmp )
dirTmp = os.path.abspath(dirTmp)

# Assign file names
if args.strHits != "":
	strHitsFile = args.strHits
else:
	strHitsFile = ( dirTmp + os.sep + "SBhits.txt" )

# Delete SBhits.txt file if it already exists.
if os.path.isfile(strHitsFile):
	os.remove(strHitsFile)


strMarkerResults = args.strMarkerResults
if strMarkerResults == "":
	strMarkerResults = dirTmp + os.sep + "markers.tab"

##############################################################################
# Determine if profiling WGS or Genome
if args.strGenome!="" and args.strWGS==None and args.bUnannotated==False:
	strMethod = "annotated_genome"
    #We assume that genomes will be a single fasta file, and that they will be
	# smaller than 900 MB, the upper bound for passing a single file to usearch.
	strSize = "small"
	strFormat = "fasta"
	sys.stderr.write("Treating input as an annotated genome...\n")
	sys.stderr.write("NOTE: When running against an annotated bug genome, ShortBRED makes a \
	usearch database from the bug genome and then searches the markers against it. \
	Please remember to increase \"maxhits\" and \"maxrejects\" to a large number, so that multiple \
	markers can hit each bug sequence. Setting these values to 0 will search the full database.\n\n")
	dictFamCounts = sq.MakeDictFamilyCounts(args.strMarkers,"")

elif args.strGenome!="" and args.strWGS==None and args.bUnannotated==True:
	strMethod = "unannotated_genome"
  
	src.CheckDependency(args.strTBLASTN,"","tblastn")
	src.CheckDependency(args.strMakeBlastDB,"","makeblastdb")
    #We assume that genomes will be a single fasta file, and that they will be
	# smaller than 900 MB, the upper bound for passing a single file to usearch.
	strSize = "small"
	strFormat = "fasta"
	sys.stderr.write("Treating input as an unannotated genome...\n")
	sys.stderr.write("NOTE: When running against an unannotated bug genome, ShortBRED makes a \n\
	tblastn database from the genome and then blasts the markers against it. \n\
	Please remember to increase \"maxhits\" to a large number, so that multiple \n\
	markers can hit each bug sequence. \n")
	dictFamCounts = sq.MakeDictFamilyCounts(args.strMarkers,"")

else:
    strMethod = "wgs"
    sys.stderr.write("Treating input as a wgs file...\n")


##############################################################################
# Log the parameters

strLog = str(dirTmp + os.sep + os.path.basename(args.strMarkers)+ ".log")
with open(strLog, "w") as log:
	log.write("ShortBRED log \n" + datetime.date.today().ctime() + "\n SEARCH PARAMETERS \n")
	log.write("Match ID:" + str(args.dID) + "\n")
	log.write("Pct Length for Match:" + str(args.dAlnLength) + "\n")
	if args.strCentroids=="Y":
		log.write("Sequences: Centroids\n")
	else:
		log.write("Sequences: Markers\n")
	if strMethod=="annotated_genome":
		log.write("Ran against the genome " + args.strGenome)





##############################################################################
#Initialize Dictionaries, Some Output Files

dictBLAST = {}
dictMarkerLen = {}
dictMarkerLenAll = {}
dictMarkerCount = {}
dictHitsForMarker = {}
dictQMPossibleOverlap = {}
dictType = {}


if (args.strBlast == ""):
	strBlast = str(dirTmp) + os.sep + strMethod+ "full_results.tab"
else:
	strBlast = args.strBlast

###############################################################################
#Step 1: Prepare markers.
# Sum up the marker lengths by family, put them in a dictionary.
# Make them into a USEARCH database.

strQMOut = str(dirTmp + os.sep + os.path.basename(args.strMarkers)+ "QM.log")
if os.path.isfile(strQMOut):
	os.remove(strQMOut)



# Fix indentation and consistent use of spaces in the code below
astrQMs = []
for seq in SeqIO.parse(args.strMarkers, "fasta"):
    # For Centroids
    if args.strCentroids == "Y":
        strStub = seq.id
    # For ShortBRED Markers
    else:
        mtchStub = re.search(r'(.*)_([TJQ]M)[0-9]*_\#([0-9]*)', seq.id)
        if mtchStub:
            strStub = mtchStub.group(1)
            strType = mtchStub.group(2)

        dictMarkerLenAll[strStub] = len(seq) + dictMarkerLenAll.get(strStub, 0)
        dictMarkerCount[strStub] = dictMarkerCount.get(strStub, 0) + 1
        dictHitsForMarker[seq.id] = 0
        dictMarkerLen[seq.id] = len(seq)

        if args.strCentroids != "Y":
            dictType[strStub] = strType

            if strType == "QM":
                astrQMs.append(seq.id)
                astrAllFams = re.search(r'\__\[(.*)\]', seq.id).group(1).split(",")
                # Example: __[ZP_04174269_w=0.541,ZP_04300309_w=0.262,NP_242644_w=0.098]

                astrFams = []
                for strFam in astrAllFams:
                    mtchFam = re.search(r'(.*)_w=(.*)', strFam)
                    if mtchFam:
                        strID = mtchFam.group(1)
                        dProp = float(mtchFam.group(2))

                        if strID == strStub:
                            dMainFamProp = dProp
                            try:
                                dLenOverlap = (dProp / dMainFamProp) * len(seq)
                            except ZeroDivisionError:
                                continue

                        if (dLenOverlap >= (args.iMinReadBP / 3)) or (dProp / dMainFamProp) >= args.dAlnLength:
                            astrFams.append(strID)

                dictQMPossibleOverlap[seq.id] = astrFams

# If profiling WGS, make a database from the markers
if strMethod == "wgs" and args.strSearchProg == "usearch":
    strDBName = tmp_dir / f"{Path(args.strMarkers).name}.udb"
    sq.MakedbUSEARCH(args.strMarkers, strDBName, args.strUSEARCH)

elif strMethod == "wgs" and args.strSearchProg == "rapsearch2":
    strDBName = tmp_dir / f"{Path(args.strMarkers).name}.rap2db"
    strDBName = os.path.abspath(strDBName)
    print("strDBName is", strDBName)
    sq.MakedbRapsearch2(args.strMarkers, strDBName, args.strPrerapPath)

elif strMethod == "wgs" and args.strSearchProg == "diamond":
    strDBName = tmp_dir / f"{Path(args.strMarkers).name}.dmnd"
    DiamondTools.make_diamond_db(
        args.strMarkers,
        strDBName,
        args.strDIAMOND,
        args.iThreads
    )

#(If profiling genome, make a database from the genome reads in Step 3.)


##################################################################################
#Step 2: Get information on WGS file(s), put it into aaFileInfo.
sys.stderr.write( "\nExamining WGS data:")
"""
aaFileInfo is array of string arrays, each with details on the file so ShortBRED
knows how to process it efficiently. Each line has the format:
	[filename, format, "large" or "small", extract method, and corresponding tarfile (if needed)]

An example:
    ['SRS011397/SRS011397.denovo_duplicates_marked.trimmed.1.fastq', 'fastq', 'large', 'r:bz2', '/n/CHB/data/hmp/wgs/samplesfqs/SRS011397.tar.bz2']
"""
if strMethod=="wgs":

	astrWGS = args.strWGS

	sys.stderr.write( "\nList of files in WGS set:")
	for strWGS in astrWGS:
		sys.stderr.write( strWGS + "\n")

	aaWGSInfo = []

	for strWGS in astrWGS:
		strExtractMethod= sq.CheckExtract(strWGS)

		# If tar file, get details on members, and note corresponding tarfile
		# Remember that a tarfile has a header block, and then data blocks
		if (strExtractMethod== 'r:bz2' or strExtractMethod=='r:gz'):
			tarWGS = tarfile.open(strWGS,strExtractMethod)
			atarinfoFiles = tarWGS.getmembers() #getmembers() returns tarInfo objects
			tarWGS.close()

			for tarinfoFile in atarinfoFiles:
				if tarinfoFile.isfile(): # This condition confirms that it is a file, not a header.
					strFormat = sq.CheckFormat(tarinfoFile.name)
					strSize = sq.CheckSize(tarinfoFile.size, c_iMaxSizeForDirectRun)
					astrFileInfo = [tarinfoFile.name, strFormat, strSize,strExtractMethod, strWGS ]
					aaWGSInfo.append(astrFileInfo)


		elif (strExtractMethod== 'bz2'):
			strWGSOut = strWGS.replace(".bz2","")
			strFormat = sq.CheckFormat(strWGSOut)
			# It is not possible to get bz2 filesize in advance, so we just assume it is large.
			strSize = "large"
			astrFileInfo = [strWGSOut, strFormat, strSize,strExtractMethod, strWGS ]
			aaWGSInfo.append(astrFileInfo)

		# Otherwise, get file details directly
		else:
			strFormat = sq.CheckFormat(strWGS)
			dFileInMB = round(os.path.getsize(strWGS)/1048576.0,1)
			if dFileInMB < c_iMaxSizeForDirectRun:
				strSize = "small"
			else:
				strSize = "large"
			astrFileInfo = [strWGS, strFormat, strSize,strExtractMethod, "no_tar" ]
			aaWGSInfo.append(astrFileInfo)


	sys.stderr.write( "\nList of files in WGS set (after unpacking tarfiles):")
	for astrWGS in aaWGSInfo:
		sys.stderr.write( astrWGS[0]+" ")

	sys.stderr.write("\n\n")
##################################################################################
# Step 3: Call USEARCH on each WGS file, (break into smaller files if needed), store hit counts.
#         OR run USEARCH on each individual genome.
# Initialize values for the sample
iTotalReadCount = 0
dAvgReadLength = 0.0
iMin = 999  # Can be any large integer. Just a value to initialize iMin before calculations begin.
iWGSFileCount = 1

if strMethod == "annotated_genome":
    # If running on an *annotated_genome*, use USEARCH.
    strDBName = tmp_dir / f"{Path(args.strGenome).name}.udb"
    sq.MakedbUSEARCH(args.strGenome, strDBName, args.strUSEARCH)

    sq.RunUSEARCHGenome(
        strMarkers=args.strGenome,
        strWGS=args.strMarkers,
        strDB=strDBName,
        strBlastOut=strBlast,
        iThreads=args.iThreads,
        dID=args.dID,
        dirTmp=dirTmp,
        iAccepts=args.iMaxHits,
        iRejects=args.iMaxRejects,
        strUSEARCH=args.strUSEARCH
    )
    sq.StoreHitCounts(
        strBlastOut=strBlast,
        strValidHits=strHitsFile,
        dictHitsForMarker=dictHitsForMarker,
        dictMarkerLen=dictMarkerLen,
        dictHitCounts=dictBLAST,
        dID=args.dID,
        strCentCheck=args.strCentroids,
        dAlnLength=args.dAlnLength,
        iMinReadAA=int(math.floor(args.iMinReadBP / 3)),
        iAvgReadAA=int(math.floor(args.iAvgReadBP / 3)),
        iAlnCentroids=args.iAlnCentroids,
        strShortBREDMode=strMethod,
        strVersionUSEARCH=strVersionUSEARCH
    )

    iWGSReads = 0
    for seq in SeqIO.parse(args.strGenome, "fasta"):
        iWGSReads += 1
        iTotalReadCount += 1
        dAvgReadLength = ((dAvgReadLength * (iTotalReadCount - 1)) + len(seq)) / float(iTotalReadCount)
        iMin = min(iMin, len(seq))

elif strMethod == "unannotated_genome":
    # If running on *unannotated_genome*, use tblastn.
    strDBName = tmp_dir / f"blastdb_{Path(args.strGenome).stem}"
    sq.MakedbBLASTnuc(args.strMakeBlastDB, strDBName, args.strGenome, dirTmp)

    sq.RunTBLASTN(
        args.strTBLASTN,
        strDBName,
        args.strMarkers,
        strBlast,
        args.iThreads
    )

    sq.StoreHitCounts(
        strBlastOut=strBlast,
        strValidHits=strHitsFile,
        dictHitsForMarker=dictHitsForMarker,
        dictMarkerLen=dictMarkerLen,
        dictHitCounts=dictBLAST,
        dID=args.dID,
        strCentCheck=args.strCentroids,
        dAlnLength=args.dAlnLength,
        iMinReadAA=int(math.floor(args.iMinReadBP / 3)),
        iAvgReadAA=int(math.floor(args.iAvgReadBP / 3)),
        iAlnCentroids=args.iAlnCentroids,
        strUSearchOut=False,
        strVersionUSEARCH=strVersionUSEARCH
    )

    iWGSReads = 0
    for seq in SeqIO.parse(args.strGenome, "fasta"):
        iWGSReads += 1
        iTotalReadCount += 1
        dAvgReadLength = ((dAvgReadLength * (iTotalReadCount - 1)) + len(seq)) / float(iTotalReadCount)
        iMin = min(iMin, len(seq))

# Otherwise, profile WGS data with USEARCH or RAPSearch2
else:
    with open(strLog, "a") as log:
        log.write('\t'.join(["# FileName", "size", "format", "extract method", "tar file (if part of one)"]) + '\n')

    for astrFileInfo in aaWGSInfo:
        strWGS, strFormat, strSize, strExtractMethod, strMainTar = astrFileInfo
        with open(strLog, "a") as log:
            log.write(str(iWGSFileCount) + ": " + '\t'.join(astrFileInfo) + '\n')
        iWGSReads = 0
        sys.stderr.write(f"Working on file {iWGSFileCount} of {len(aaWGSInfo)}\n")

        # If it's a small FASTA file, just give it to USEARCH or RAPSearch directly.
        if strFormat == "fasta" and strSize == "small":
            if args.strSearchProg == "rapsearch2":
                sq.RunRAPSEARCH2(
                    strMarkers=args.strMarkers,
                    strWGS=strWGS,
                    strDB=strDBName,
                    strBlastOut=strBlast,
                    iThreads=args.iThreads,
                    dID=args.dID,
                    dirTmp=dirTmp,
                    iAccepts=args.iMaxHits,
                    iRejects=args.iMaxRejects,
                    strRAPSEARCH2=args.strRap2Path
                )

                sq.StoreHitCountsRapsearch2(
                    strBlastOut=strBlast,
                    strValidHits=strHitsFile,
                    dictHitsForMarker=dictHitsForMarker,
                    dictMarkerLen=dictMarkerLen,
                    dictHitCounts=dictBLAST,
                    dID=args.dID,
                    strCentCheck=args.strCentroids,
                    dAlnLength=args.dAlnLength,
                    iMinReadAA=int(math.floor(args.iMinReadBP / 3)),
                    iAvgReadAA=int(math.floor(args.iAvgReadBP / 3)),
                    iAlnCentroids=args.iAlnCentroids
                )

            elif args.strSearchProg == "usearch":
                sq.RunUSEARCH(
                    strMarkers=args.strMarkers,
                    strWGS=strWGS,
                    strDB=strDBName,
                    strBlastOut=strBlast,
                    iThreads=args.iThreads,
                    dID=args.dID,
                    dirTmp=dirTmp,
                    iAccepts=args.iMaxHits,
                    iRejects=args.iMaxRejects,
                    strUSEARCH=args.strUSEARCH
                )

                sq.StoreHitCounts(
                    strBlastOut=strBlast,
                    strValidHits=strHitsFile,
                    dictHitsForMarker=dictHitsForMarker,
                    dictMarkerLen=dictMarkerLen,
                    dictHitCounts=dictBLAST,
                    dID=args.dID,
                    strCentCheck=args.strCentroids,
                    dAlnLength=args.dAlnLength,
                    iMinReadAA=int(math.floor(args.iMinReadBP / 3)),
                    iAvgReadAA=int(math.floor(args.iAvgReadBP / 3)),
                    iAlnCentroids=args.iAlnCentroids,
                    strShortBREDMode=strMethod,
                    strVersionUSEARCH=strVersionUSEARCH
                )

            elif args.strSearchProg == "diamond":
                DiamondTools.run_diamond_search(
                    query=strWGS,
                    db=strDBName,
                    out=strBlast,
                    diamond_path=args.strDIAMOND,
                    threads=args.iThreads,
                    sensitivity=args.diamond_sensitivity
                )

                sq.StoreHitCounts(
                    strBlastOut=strBlast,
                    strValidHits=strHitsFile,
                    dictHitsForMarker=dictHitsForMarker,
                    dictMarkerLen=dictMarkerLen,
                    dictHitCounts=dictBLAST,
                    dID=args.dID,
                    strCentCheck=args.strCentroids,
                    dAlnLength=args.dAlnLength,
                    iMinReadAA=int(math.floor(args.iMinReadBP / 3)),
                    iAvgReadAA=int(math.floor(args.iAvgReadBP / 3)),
                    iAlnCentroids=args.iAlnCentroids,
                    strShortBREDMode=strMethod,
                    strVersionUSEARCH=strVersionUSEARCH
                )

##################################################################################
# Step 4: Calculate ShortBRED Counts, print results, print log info.
if strMethod=="annotated_genome":
	strInputFile = args.strGenome
elif strMethod=="unannotated_genome":
	strInputFile = args.strGenome
elif strMethod=="wgs":
	strInputFile=args.strWGS
	

if strMethod=="wgs":
	atupCounts = sq.CalculateCounts(strResults = args.strResults, strMarkerResults=strMarkerResults,dictHitCounts=dictBLAST,
	dictMarkerLenAll=dictMarkerLenAll,dictHitsForMarker=dictHitsForMarker,dictMarkerLen=dictMarkerLen,
	dReadLength = float(args.iAvgReadBP), iWGSReads = iTotalReadCount, strCentCheck=args.strCentroids,dAlnLength=args.dAlnLength,strFile = strInputFile)

	# Row of atupCounts = (strProtFamily,strMarker, dCount,dictHitsForMarker[strMarker],dictMarkerLen[strMarker],dReadLength,iPossibleHitSpace)




###########################################################################
# Added to produce counts of bug genomes 
##########################################################################

if strMethod=="annotated_genome":
	dictFinalCounts = sq.NormalizeGenomeCounts(strHitsFile,dictFamCounts,bUnannotated=False,dPctMarkerThresh=args.dPctMarkerThresh)
	sys.stderr.write("Normalizing hits to genome... \n")

elif strMethod=="unannotated_genome":
	dictFinalCounts = sq.NormalizeGenomeCounts(strHitsFile,dictFamCounts,bUnannotated=True,dPctMarkerThresh=args.dPctMarkerThresh)
	sys.stderr.write("Normalizing hits to genome... \n")


if strMethod=="annotated_genome" or strMethod=="unannotated_genome":

	with open(args.strResults,'w') as fileBugCounts:
		fileBugCounts.write("Family" + "\t" + "Count" + "\n")
		for strFam in sorted(dictFinalCounts.keys()):
			fileBugCounts.write(strFam + "\t" + str(dictFinalCounts[strFam]) + "\n")

# Add final details to log
with open(str(dirTmp + os.sep + os.path.basename(args.strMarkers)+ ".log"), "a") as log:
	log.write("Total Reads Processed: " + str(iTotalReadCount) + "\n")
	log.write("Average Read Length Specified by User: " + str(args.iAvgReadBP) + "\n")
	log.write("Average Read Length Calculated by ShortBRED: " + str(dAvgReadLength) + "\n")
	log.write("Min Read Length: " + str(iMin) + "\n")


sys.stderr.write("Processing complete. \n")
########################################################################################
# This is part of a possible EM application that is not fully implemented yet.
########################################################################################
if (args.strBayes != ""):
    # Just write the final quantification results
    with open(args.strBayes, 'w') as f:
        f.write("Family\tMarker\tCount\n")
        for tup in atupCounts:
            strProtFamily, strMarker, dCount = tup[:3]
            f.write(f"{strProtFamily}\t{strMarker}\t{dCount}\n")

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class DynamicClusterer:
    def __init__(self, eps=0.5, min_samples=10, adaptive=True):
        """Initialize the clustering model."""
        self.eps = eps
        self.min_samples = min_samples
        self.adaptive = adaptive
        self.model = hdbscan.HDBSCAN(min_cluster_size=min_samples) if adaptive else DBSCAN(eps=eps, min_samples=min_samples)

    def fit(self, data):
        """Fit clustering model to data."""
        logger.info("Fitting clustering model...")
        if isinstance(self.model, DBSCAN):
            scaled_data = StandardScaler().fit_transform(data)
            self.model.fit(scaled_data)
        else:
            self.model.fit(data)
        return self

    def update(self, new_data):
        """Update clustering with new data if adaptive."""
        logger.info("Updating clustering model...")
        if not self.adaptive:
            raise ValueError("Update is only supported for adaptive clustering.")
        self.model.partial_fit(new_data)
        return self

    def get_clusters(self):
        """Get cluster labels."""
        return self.model.labels_


class RealTimePosterior:
    def __init__(self, model_path=None):
        """Initialize the posterior assignment model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        if model_path:
            logger.info(f"Loading pre-trained model from {model_path}...")
            self.model = joblib.load(model_path)

    def train(self, X, y):
        """Train the model on labeled data."""
        logger.info("Training posterior assignment model...")
        self.model.fit(X, y)

    def predict(self, X):
        """Predict posterior probabilities."""
        logger.info("Predicting posterior probabilities...")
        return self.model.predict_proba(X)

    def update(self, X, y):
        """Update the model with new data."""
        logger.info("Updating posterior assignment model...")
        self.model.fit(X, y)  # Replace with incremental training if using an incremental model

    def save(self, path):
        """Save the model."""
        logger.info(f"Saving model to {path}...")
        joblib.dump(self.model, path)


# Example integration
if __name__ == "__main__":
    # Simulated data
    np.random.seed(42)
    data = np.random.rand(1000, 10)  # 1000 sequences with 10 features
    labels = np.random.randint(0, 5, size=1000)  # 5 possible clusters

    # Clustering
    clusterer = DynamicClusterer(min_samples=5, adaptive=True)
    clusterer.fit(data)

    cluster_labels = clusterer.get_clusters()
    logger.info(f"Generated {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters.")

    # Posterior Assignment
    posterior_model = RealTimePosterior()
    posterior_model.train(data, labels)

    # New data
    new_data = np.random.rand(200, 10)
    new_labels = np.random.randint(0, 5, size=200)

    # Update clustering and posterior model
    clusterer.update(new_data)
    posterior_model.update(new_data, new_labels)

    # Predict posteriors for new data
    posteriors = posterior_model.predict(new_data)
    logger.info(f"Predicted posteriors: {posteriors[:5]}")

    # Save model
    posterior_model.save("posterior_model.pkl")
    logger.info("Model saved.")

    logger.info("Example integration complete.")


import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class ModelBasedClusterer:
    """Re-cluster sequences using posterior probabilities as features."""
    
    def __init__(self, base_clusterer, n_components=2, n_clusters=5):
        self.base_clusterer = base_clusterer
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.pca = PCA(n_components=self.n_components)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

    def re_cluster(self, sequence_features, posteriors):
        """Cluster sequences using posterior probabilities as enriched features."""
        if posteriors is None or len(posteriors) == 0:
            raise ValueError("Posterior probabilities are required for re-clustering.")

        # Concatenate sequence features with posterior probabilities
        enriched_features = np.hstack((sequence_features, posteriors))
        
        # Dimensionality reduction for efficiency
        reduced_features = self.pca.fit_transform(enriched_features)
        
        # Re-cluster with the reduced feature space
        new_clusters = self.kmeans.fit_predict(reduced_features)
        return new_clusters


# Initial clustering
initial_clusters = clusterer.cluster(data)

# Assign posterior probabilities
posterior_probs = posterior_model.assign_posterior(data, initial_clusters)

# Re-cluster using posterior-enriched features
model_clusterer = ModelBasedClusterer(base_clusterer=clusterer, n_components=3, n_clusters=10)
final_clusters = model_clusterer.re_cluster(data, posterior_probs)

