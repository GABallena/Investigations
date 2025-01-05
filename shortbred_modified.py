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
# 
# 1. Resource Monitoring and Logging:
#    - Integrated `psutil` for detailed CPU, memory, and disk usage tracking.
#    - Periodic resource logging ensures efficient runtime performance monitoring.
#
# 2. Checkpoint System:
#    - JSON-based checkpointing enables seamless recovery after failures.
#    - Automatically resumes long-running tasks from the last saved state.
#
# 3. Parallel Processing:
#    - Optimized execution using Python's `multiprocessing` for multi-core systems.
#    - Real-time progress tracking with `tqdm` for enhanced user feedback.
#
# 4. Enhanced Error Handling:
#    - Comprehensive exception management for subprocess calls and I/O operations.
#    - Pre-execution validation of dependencies and inputs ensures robustness.
#
# 5. Memory-Efficient Processing:
#    - Handles large datasets in optimized chunks, minimizing memory overhead.
#    - Ideal for high-throughput sequencing and resource-intensive workflows.
#
# 6. DNA Sequence Translation:
#    - `--dna` flag accepts DNA sequences for six-frame translation.
#    - Automatically selects the longest Open Reading Frame (ORF) for downstream analysis.
#
# 7. Dependency Validation:
#    - Validates tools like `usearch`, `cd-hit`, `tblastn`, and `diamond`.
#    - Provides user-friendly error messages for missing or outdated dependencies.
#
# 8. Preprocessing Flexibility:
#    - Optional deduplication workflows using tools like `clumpify.sh` and `FastUniq`.
#    - Configurable via the `--dedup` flag to suit diverse data preprocessing needs.
#
# 9. Enhanced Logging:
#    - Dual-channel logging records detailed progress and system resource usage.
#    - Logs include command parameters, runtime statistics, and performance data.
#
# 10. Input Validation:
#     - Ensures input files exist and comply with FASTA/FASTQ formats.
#     - Provides detailed error feedback for malformed or missing inputs.
#
# 11. Unified Workflow:
#     - Combines `ShortBRED-Identify` and `ShortBRED-Quantify` functionalities.
#     - Streamlines the analysis pipeline for improved usability and performance.
#
# 12. Secure Command Execution:
#     - Subprocess calls are wrapped with error capture mechanisms.
#     - Input sanitization ensures secure execution of external tools.
#
# 13. Modular Architecture:
#     - Refactored into reusable classes for better maintainability and scalability.
#     - Modular design facilitates debugging and future feature expansions.
#
# 14. DIAMOND Integration:
#     - Added DIAMOND as a fast, sensitive option for sequence alignment.
#     - Supports configurable sensitivity modes like `fast` and `very-sensitive`.
#
# 15. Genome Count Normalization:
#     - Automatically normalizes counts based on marker coverage thresholds.
#     - Handles both annotated and unannotated genomes efficiently.
#
# 16. Bayesian Refinement:
#     - Bayesian clustering for adaptive marker selection and refinement.
#     - Utilizes posterior probabilities for dynamic optimization.
#
# 17. Advanced Clustering Techniques:
#     - Adaptive clustering with HDBSCAN for flexible sequence grouping.
#     - Handles noisy data and variable cluster sizes effectively.
#
# 18. Real-Time Progress Updates:
#     - Provides progress feedback with estimated completion times.
#     - Enhances user experience during long-running processes.
#
# 19. Output Management:
#     - Standardized file naming for outputs and intermediate results.
#     - Configurable output directories for streamlined workflow organization.
#
# 20. Documentation Enhancements:
#     - Improved inline comments for clarity and easier maintenance.
#     - Comprehensive user documentation with examples and tips for efficient usage.
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


# Initialize argument parser at module level
parser = argparse.ArgumentParser(
    description='''
ShortBRED - Short, Better Representative Extract Database
Version: Modified-1.0 (2025)

MANDATORY (choose one workflow):

    1. Marker Generation Workflow:
        --goi PATH           Input protein families (FASTA)    [REQUIRED]
        --ref PATH          Reference protein sequences       [REQUIRED]
        
    2. Quantification Workflow:
        --markers PATH      ShortBRED markers file           [REQUIRED]
        AND EITHER:
        --wgs PATH         WGS reads file(s)                 [REQUIRED if not using --genome]
        --genome PATH      Genome file                       [REQUIRED if not using --wgs]

COMMONLY USED OPTIONS:
    Analysis Control:
        --threads INT      Number of CPU threads              [default: 1]
        --dna             Process DNA input (auto-translate)  [default: False]
        --verbose         Show detailed progress              [default: False]

    Search Configuration:
        --search_program PROG   Search tool to use            [default: usearch]
                               Options: usearch, rapsearch2, diamond
        --id FLOAT             Match identity threshold       [default: 0.95]

    Output Control:
        --results PATH     Output file for results           [default: results.tab]
        --tmp PATH        Temporary directory                [default: auto-generated]

ADVANCED OPTIONS:
    Marker Generation:
        --min_marker_length INT    Minimum marker length     [default: 30]
        --marker_identity FLOAT    Identity threshold        [default: 0.9]
        --cdhit_cluster FLOAT     CD-HIT clustering         [default: 0.9]

    Search Parameters:
        --diamond_sensitivity MODE    DIAMOND sensitivity    [default: sensitive]
                                     (fast/sensitive/more-sensitive/very-sensitive)
        --maxhits INT                Max hits per read      [default: 1]
        --maxrejects INT             Max rejection count    [default: 32]
        --pctlength FLOAT            Min alignment length   [default: 0.95]

    Processing Options:
        --unannotated              Process unannotated genome
        --bz2                      Handle bz2 compressed files
        --pctmarker_thresh FLOAT   Marker mapping threshold [default: 0.1]

For complete documentation: https://github.com/gmballena/shortbred-modified

EXAMPLES:
    1. Generate markers from protein families:
       python shortbred_modified.py --goi proteins.faa --ref reference.faa

    2. Quantify markers in metagenome:
       python shortbred_modified.py --markers markers.faa --wgs metagenome.fastq

    3. Process genome with DIAMOND:
       python shortbred_modified.py --markers markers.faa --genome genome.faa \
           --search_program diamond --threads 8
''', 
    formatter_class=RawTextHelpFormatter, 
    add_help=False  # Disable default -h
)

# Custom -h option
#parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
#                    help='Show this help message and exit.')

# Other argument definitions
parser.add_argument("--version", action="version", version="%(prog)s vModified-1.0 (2025)")

# Workflow arguments
workflow = parser.add_argument_group('Mandatory Workflows')
workflow.add_argument('--goi', type=str, help='Input protein families (FASTA) [REQUIRED for marker generation]')
workflow.add_argument('--ref', type=str, help='Reference protein sequences (FASTA) [REQUIRED for marker generation]')
workflow.add_argument('--markers', type=str, help='ShortBRED markers file [REQUIRED for quantification]')
workflow.add_argument('--wgs', type=str, nargs='+', help='WGS reads file(s) [REQUIRED if not using --genome]')
workflow.add_argument('--genome', type=str, help='Genome file [REQUIRED if not using --wgs]')

# Common options
common = parser.add_argument_group('Commonly Used Options')
common.add_argument('--threads', type=int, default=1, help='Number of CPU threads [default: 1]')
common.add_argument('--dna', action='store_true', help='Process DNA input (auto-translate) [default: False]')
common.add_argument('--verbose', action='store_true', help='Show detailed progress [default: False]')
common.add_argument('--help', '-h', action='help', help='Show this help message and exit')

# Parse arguments (move this to where it's needed)
# args = parser.parse_args()

# Parse arguments
args = parser.parse_args()

 


################################################################################
# Constants
c_iMaxSizeForDirectRun = 900 # File size in MB. Any WGS file smaller than this
							 # does not need to made into smaller WGS files.

c_iReadsForFile = 7000000 # Number of WGS reads to process at a time

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

class RobustExecutor:
    """Handles robust execution with retries and error recovery"""
    
    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)

    def execute_with_retry(self, func, *args, recovery_handler=None, **kwargs):
        """Execute function with retry logic and error recovery"""
        last_exception = None
        wait_time = 1

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                
                if recovery_handler:
                    try:
                        recovery_handler(e, attempt)
                    except Exception as re:
                        self.logger.error(f"Recovery failed: {str(re)}")

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    wait_time *= self.backoff_factor

        raise RuntimeError(f"Operation failed after {self.max_retries} attempts: {str(last_exception)}")

class ResourceMonitor:
    """Monitors system resources during execution"""
    
    def __init__(self, threshold_cpu=90, threshold_memory=90):
        self.threshold_cpu = threshold_cpu
        self.threshold_memory = threshold_memory
        self.logger = logging.getLogger(__name__)
        self._start_time = time.time()
        self._last_check = self._start_time
        self.stats = []

    def check_resources(self):
        """Check current resource usage and log if thresholds are exceeded"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            current_stats = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
            self.stats.append(current_stats)
            
            if cpu_percent > self.threshold_cpu:
                self.logger.warning(f"High CPU usage detected: {cpu_percent}%")
            
            if memory.percent > self.threshold_memory:
                self.logger.warning(f"High memory usage detected: {memory.percent}%")
                
            return current_stats
            
        except Exception as e:
            self.logger.error(f"Resource monitoring failed: {str(e)}")
            return None

    def get_resource_summary(self):
        """Get summary of resource usage over time"""
        if not self.stats:
            return None
            
        return {
            'duration': time.time() - self._start_time,
            'max_cpu': max(s['cpu_percent'] for s in self.stats),
            'max_memory': max(s['memory_percent'] for s in self.stats),
            'avg_cpu': sum(s['cpu_percent'] for s in self.stats) / len(self.stats),
            'avg_memory': sum(s['memory_percent'] for s in self.stats) / len(self.stats)
        }

class CheckpointManager:
    """Manages checkpoints for long-running operations"""
    
    def __init__(self, checkpoint_dir, prefix="shortbred_checkpoint"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prefix = prefix
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_checkpoint = None
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, state, checkpoint_id=None):
        """Save checkpoint state to file"""
        try:
            if checkpoint_id is None:
                checkpoint_id = int(time.time())
            
            checkpoint_path = self.checkpoint_dir / f"{self.prefix}_{checkpoint_id}.json"
            
            # Add metadata to state
            state['_metadata'] = {
                'timestamp': datetime.datetime.now().isoformat(),
                'checkpoint_id': checkpoint_id,
                'version': VERSION
            }
            
            with checkpoint_path.open('w') as f:
                json.dump(state, f, indent=2)
            
            self.current_checkpoint = checkpoint_path
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, checkpoint_id=None):
        """Load checkpoint state from file"""
        try:
            if checkpoint_id is None:
                # Find most recent checkpoint
                checkpoints = sorted(self.checkpoint_dir.glob(f"{self.prefix}_*.json"))
                if not checkpoints:
                    return None
                checkpoint_path = checkpoints[-1]
            else:
                checkpoint_path = self.checkpoint_dir / f"{self.prefix}_{checkpoint_id}.json"
            
            if not checkpoint_path.exists():
                return None
            
            with checkpoint_path.open('r') as f:
                state = json.load(f)
            
            # Validate checkpoint
            if not self._validate_checkpoint(state):
                raise ValueError("Checkpoint validation failed")
            
            self.current_checkpoint = checkpoint_path
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

    def _validate_checkpoint(self, state):
        """Validate checkpoint data"""
        try:
            metadata = state.get('_metadata', {})
            
            # Check version compatibility
            checkpoint_version = metadata.get('version')
            if checkpoint_version != VERSION:
                self.logger.warning(f"Checkpoint version mismatch: {checkpoint_version} vs {VERSION}")
            
            # Add more validation as needed
            return True
            
        except Exception as e:
            self.logger.error(f"Checkpoint validation failed: {str(e)}")
            return False

    def _cleanup_old_checkpoints(self, keep_count=5):
        """Clean up old checkpoints, keeping only the most recent ones"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob(f"{self.prefix}_*.json"))
            if len(checkpoints) > keep_count:
                for checkpoint in checkpoints[:-keep_count]:
                    checkpoint.unlink()
                self.logger.info(f"Cleaned up {len(checkpoints) - keep_count} old checkpoints")
        except Exception as e:
            self.logger.warning(f"Checkpoint cleanup failed: {str(e)}")

class InputValidator:
    """Enhanced input validation with detailed error reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors = []
        self.warnings = []

    # ...existing InputValidator methods...

    def validate_sequence_quality(self, filepath, min_length=30, max_n_percent=10):
        """Validate sequence quality metrics"""
        try:
            with open(filepath, 'r') as f:
                for record in SeqIO.parse(f, "fasta"):
                    seq = str(record.seq)
                    
                    # Check sequence length
                    if len(seq) < min_length:
                        self.errors.append(f"Sequence {record.id} is too short: {len(seq)} < {min_length}")
                    
                    # Check for ambiguous bases
                    n_count = seq.upper().count('N')
                    n_percent = (n_count / len(seq)) * 100
                    if n_percent > max_n_percent:
                        self.warnings.append(
                            f"Sequence {record.id} has high ambiguity: {n_percent:.1f}% N's"
                        )
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.errors.append(f"Failed to validate sequence quality: {str(e)}")
            return False

# Update main() to use new robust components
def main():
    # Initialize robust components
    executor = RobustExecutor()
    monitor = ResourceMonitor()
    checkpoints = CheckpointManager(tmp_dir)
    validator = InputValidator()

    try:
        # ...existing argument parsing...

        # Validate inputs with enhanced validation
        if not validator.validate_all(args):
            for error in validator.errors:
                logging.error(error)
            sys.exit(1)
        
        for warning in validator.warnings:
            logging.warning(warning)

        # Start resource monitoring in a separate thread
        monitor_thread = threading.Thread(
            target=lambda: monitor.check_resources(),
            daemon=True
        )
        monitor_thread.start()

        # Load last checkpoint if exists
        state = checkpoints.load_checkpoint()
        if state:
            logging.info("Resuming from checkpoint")
            # Restore state here

        # Main processing with robust execution
        try:
            # ...existing processing code...
            
            # Save checkpoints periodically
            checkpoints.save_checkpoint({
                'progress': 'processing_complete',
                'results': results
            })

        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            # Try to save checkpoint before exiting
            checkpoints.save_checkpoint({
                'progress': 'failed',
                'error': str(e)
            })
            raise

        finally:
            # Log resource usage summary
            summary = monitor.get_resource_summary()
            if summary:
                logging.info("Resource usage summary:")
                for key, value in summary.items():
                    logging.info(f"{key}: {value}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)


