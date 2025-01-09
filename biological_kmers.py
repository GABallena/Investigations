import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import SGDClassifier  # Add this import
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import log
import sgd_utils  # SGD hash for Bloom filter
import bitarray

try:
    from cmdstanpy import CmdStanModel
    HAS_STAN = True
except ImportError:
    HAS_STAN = False

# Add new imports
from ete3 import PhyloTree, TreeStyle
from Bio import AlignIO
import warnings
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
from Bio import motifs
from Bio.Data import CodonTable
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                           average_precision_score, confusion_matrix)
from Bio import Entrez
from Bio.Blast import NCBIWWW
import subprocess
import tempfile
from ete3 import Tree
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import dendropy
from scipy.stats import chi2_contingency
from Bio.Phylo.PAML import yn00
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import networkx as nx
from joblib import dump, load
import os.path
import datetime
import logging
import logging.handlers
from subprocess import TimeoutError, CalledProcessError
from Bio import SearchIO  # Add this import

def setup_logging(log_file='biological_kmers.log'):
    """Configure logging with both file and console handlers"""
    logger = logging.getLogger('biological_kmers')
    logger.setLevel(logging.DEBUG)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class SubprocessError(Exception):
    """Custom exception for subprocess errors with detailed information"""
    def __init__(self, message, cmd=None, output=None, error=None):
        self.message = message
        self.cmd = cmd
        self.output = output
        self.error = error
        super().__init__(self.message)

def run_subprocess(cmd, timeout=300, check=True):
    """Run subprocess with robust error handling and logging.
    
    Args:
        cmd (str): Command to execute
        timeout (int): Maximum execution time in seconds
        check (bool): Whether to check return code
        
    Returns:
        subprocess.CompletedProcess: Completed process object
        
    Raises:
        SubprocessError: If process fails or times out
    """
    try:
        logger.debug(f"Running command: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=check
        )
        logger.debug(f"Command completed successfully: {cmd}")
        return result
        
    except TimeoutError:
        error_msg = f"Command timed out after {timeout}s: {cmd}"
        logger.error(error_msg)
        raise SubprocessError(error_msg, cmd=cmd)
        
    except CalledProcessError as e:
        error_msg = f"Command failed with exit code {e.returncode}: {cmd}"
        logger.error(error_msg)
        logger.error(f"Error output: {e.stderr}")
        raise SubprocessError(
            error_msg,
            cmd=cmd,
            output=e.output,
            error=e.stderr
        )
        
    except Exception as e:
        error_msg = f"Unexpected error running command: {cmd}"
        logger.error(error_msg, exc_info=True)
        raise SubprocessError(error_msg, cmd=cmd, error=str(e))

class BloomFilter:
    """Bloom filter for efficient sequence membership testing"""
    def __init__(self, size, num_hash_functions):
        self.size = size
        self.num_hash_functions = num_hash_functions
        self.bit_array = bitarray.bitarray(size)
        self.bit_array.setall(0)
    
    def add(self, item):
        for seed in range(self.num_hash_functions):
            index = mmh3.hash(str(item), seed) % self.size
            self.bit_array[index] = 1
    
    def __contains__(self, item):
        for seed in range(self.num_hash_functions):
            index = mmh3.hash(str(item), seed) % self.size
            if not self.bit_array[index]:
                return False
        return True

def create_bloom_filter(items, false_positive_rate=0.01):
    """Create a Bloom filter with optimal parameters"""
    n = len(items)
    m = int(-n * log(false_positive_rate) / (log(2) ** 2))
    k = int(m * log(2) / n)
    bloom = BloomFilter(m, k)
    for item in items:
        bloom.add(item)
    return bloom

def extract_features(sequence):
    """Extract features from k-mer sequence.
    
    Args:
        sequence (str): Input DNA sequence to analyze.
        
    Returns:
        list: List of numerical features including:
            - GC content (float): Ratio of G and C nucleotides
            - Sequence complexity (float): Ratio of unique to total nucleotides
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    complexity = len(set(sequence)) / len(sequence)
    return [gc_content, complexity]

def get_nucleotide_freq(seq, n):
    """Calculate n-nucleotide frequencies in a sequence.
    
    Args:
        seq (str): Input DNA sequence.
        n (int): Length of nucleotide patterns to count (e.g., 2 for dinucleotides).
        
    Returns:
        dict: Mapping of n-nucleotide patterns to their frequencies.
            Keys are nucleotide patterns (str)
            Values are normalized frequencies (float)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if len(seq) < n:
        return {}
    
    freqs = {}
    for i in range(len(seq) - n + 1):
        nuc = seq[i:i+n]
        freqs[nuc] = freqs.get(nuc, 0) + 1
    return {k: v/(len(seq)-n+1) for k,v in freqs.items()}

def load_contaminant_db(contaminants_folder):
    """Load and index contaminant sequences from a folder.
    
    Args:
        contaminants_folder (str): Path to folder containing FASTA files.
        
    Returns:
        Union[BloomFilter, set]: Either a Bloom filter (for large datasets) or 
            set containing contaminant sequences. The Bloom filter is used for
            memory-efficient storage when database size exceeds 10000 sequences.
    
    Note:
        Supports both .fasta and .fa file extensions.
        Uses Bloom filter for large datasets to reduce memory usage.
    """
    contaminants = []
    for filename in os.listdir(contaminants_folder):
        if filename.endswith(('.fasta', '.fa')):
            filepath = os.path.join(contaminants_folder, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                contaminants.append(str(record.seq))
    
    # Create Bloom filter index for large databases
    if len(contaminants) > 10000:
        return create_bloom_filter(contaminants)
    return set(contaminants)

def generate_random_sequences(n, length=100):
    """Generate random DNA sequences"""
    bases = ['A', 'T', 'G', 'C']
    return [''.join(np.random.choice(bases, length)) for _ in range(n)]

def is_contaminant(sequence, contaminants_folder="contaminants"):
    """Check if sequence matches known contaminants"""
    contaminant_db = load_contaminant_db(contaminants_folder)
    
    def create_feature_vectors_parallel(sequences):
        """Create feature vectors for multiple sequences in parallel"""
        with ProcessPoolExecutor() as executor:
            return list(executor.map(create_feature_vector, sequences))

    def create_feature_vector(sequence):
        """Create complete feature vector for a sequence"""
        features = extract_features(sequence)
        
        # Add di/tri/tetra-nucleotide frequencies
        for n in range(2,5):
            freq_dict = get_nucleotide_freq(sequence, n)
            features.extend(freq_dict.values())
        
        # Add entropy and codon bias
        features.append(get_entropy(sequence))
        features.append(get_codon_bias(sequence))
        
        return features
    return stan_model(sequence, contaminant_db)

def get_entropy(seq):
    """Calculate Shannon entropy of sequence"""
    counts = Counter(seq)
    probs = [count/len(seq) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def get_codon_bias(seq):
    """Calculate simple codon usage bias"""
    if len(seq) < 3:
        return 0
    codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
    return len(set(codons)) / len(codons)

def get_repeat_content(seq):
    """Calculate fraction of sequence in repeats"""
    repeats = 0
    for i in range(len(seq)-3):
        for j in range(2, 6):  # Look for 2-5mer repeats
            if i + 2*j <= len(seq):
                if seq[i:i+j] == seq[i+j:i+2*j]:
                    repeats += j
                    break
    return repeats / len(seq) if len(seq) > 0 else 0

def get_kmer_entropy(seq, k=3):
    """Calculate k-mer entropy to measure sequence complexity"""
    if len(seq) < k:
        return 0
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)
    probs = [count/len(kmers) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def get_palindrome_content(seq):
    """Calculate fraction of sequence in palindromes"""
    palindromes = 0
    for i in range(len(seq)):
        for j in range(4, min(11, len(seq)-i)):  # Look for 4-10bp palindromes
            subseq = seq[i:i+j]
            if subseq == subseq[::-1]:
                palindromes += j
                break
    return palindromes / len(seq) if len(seq) > 0 else 0

def get_motif_scores(seq):
    """Calculate scores for common biological motifs"""
    motifs = {
        'tata_box': ['TATAAA', 'TATATA', 'TATATAT'],
        'splice_donor': ['GT', 'GC'],
        'splice_acceptor': ['AG'],
        'kozak': ['ACCATGG', 'GCCATGG'],
        'poly_a': ['AATAAA']
    }
    scores = {}
    for motif_type, patterns in motifs.items():
        score = sum(seq.count(pattern) for pattern in patterns)
        scores[f'{motif_type}_score'] = score / (len(seq) - 1) if len(seq) > 1 else 0
    return scores

def get_frame_bias(seq):
    """Calculate reading frame bias"""
    if len(seq) < 3:
        return 0
    frames = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
    frame_counts = Counter(frames)
    total = len(frames)
    return max(count/total for count in frame_counts.values()) if total > 0 else 0

def create_feature_vector(sequence, hmm_db_path=None):
    """Create comprehensive feature vector for sequence analysis.
    
    Args:
        sequence (str): Input DNA sequence.
        
    Returns:
        list: Numerical features including:
            - Basic features (GC content, complexity)
            - Nucleotide frequencies (di/tri/tetra-nucleotides)
            - Biological features (entropy, codon bias, repeats)
            - Motif scores (TATA box, splice sites, etc.)
            - External predictions (genes, tRNAs, etc.)
            - Evolutionary features (selection, phylogeny)
            - Relationship features (orthologs, paralogs)
    
    Note:
        This is the main feature extraction function that combines
        all feature types for comprehensive sequence analysis.
    """
    features = extract_features(sequence)
    
    # Add basic nucleotide frequencies
    for n in range(2,5):
        freq_dict = get_nucleotide_freq(sequence, n)
        features.extend(freq_dict.values())
    
    # Add basic features
    features.append(get_entropy(sequence))
    features.append(get_codon_bias(sequence))
    
    # Add new biological features
    features.append(get_repeat_content(sequence))
    features.append(get_palindrome_content(sequence))
    features.append(get_frame_bias(sequence))
    features.append(get_kmer_entropy(sequence, k=3))
    features.append(get_kmer_entropy(sequence, k=4))
    
    # Add motif scores
    motif_scores = get_motif_scores(sequence)
    features.extend(motif_scores.values())
    
    # Run external predictions
    predictions = run_external_predictions(sequence)
    
    # Add RepeatMasker features
    if (predictions['repeat_masker']):
        features.extend([
            predictions['repeat_masker']['repeat_count'],
            len(predictions['repeat_masker']['repeat_types']),
            np.mean(predictions['repeat_masker']['repeat_lengths'])
        ])
    else:
        features.extend([0, 0, 0])
    
    # Add gene prediction features
    if (predictions['gene_prediction']):
        features.extend([
            predictions['gene_prediction']['gene_count'],
            predictions['gene_prediction']['exon_count'],
            np.mean(predictions['gene_prediction']['cds_lengths'])
        ])
    else:
        features.extend([0, 0, 0])
    
    # Add tRNA features
    if (predictions['trna_scan']):
        features.extend([
            predictions['trna_scan']['trna_count'],
            len(predictions['trna_scan']['anticodon_types']),
            np.mean(predictions['trna_scan']['scores'])
        ])
    else:
        features.extend([0, 0, 0])
    
    # Add phylogenetic features
    features.extend([
        predictions['phyml_score'],
        predictions['selection_score'],
        predictions['convergence_prob']
    ])
    
    # Add evolutionary relationship features
    features.extend([
        relationships['orthogroup_size'],
        relationships['paralog_count'],
        relationships['duplication_score'],
        relationships['synteny_score'],
        relationships['cluster_density']
    ])
    
    # Add HMM features if database is provided
    if hmm_db_path and os.path.exists(hmm_db_path):
        hmm_features = get_hmm_features(sequence, hmm_db_path)
        features.extend([
            hmm_features['best_hmm_score'],
            hmm_features['hmm_hit_count'],
            hmm_features['mean_hmm_score'],
            -np.log10(hmm_features['min_hmm_evalue'] + 1e-300)  # Convert e-value to score
        ])
    
    return features

def train_kmer_classifier(hmm_db_path=None):
    """Train classifier using streaming data"""
    # Initialize SGD classifier instead of RandomForest
    clf = SGDClassifier(
        loss='log_loss',  # For probability estimates
        learning_rate='optimal',
        eta0=0.01,
        random_state=42
    )
    model_handler = ModelPersistence()
    
    # Train incrementally
    for i, (X_chunk, y_chunk) in enumerate(create_kmer_dataset("biological_sequences.fasta", hmm_db_path=hmm_db_path)):
        if i == 0:
            # First chunk - fit the model with classes
            clf.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
        else:
            # Subsequent chunks - update the model
            clf.partial_fit(X_chunk, y_chunk)
        
        print(f"Processed chunk {i+1}")
    
    # Save trained model with metadata
    metadata = {
        'training_date': datetime.datetime.now().isoformat(),
        'data_source': "biological_sequences.fasta",
        'model_version': '1.0',
        'model_type': 'SGDClassifier',
        'hmm_database': hmm_db_path if hmm_db_path else 'None'
    }
    model_path = model_handler.save_model(clf, metadata)
    
    return clf, model_path

def normalize_features(features, min_val=0, max_val=1):
    """Normalize features to prevent numerical instability"""
    if len(features) == 0:
        return features
    features = np.array(features)
    feature_range = np.max(features) - np.min(features)
    if feature_range == 0:
        return np.zeros_like(features)
    return min_val + (features - np.min(features)) * (max_val - np.min(features)) / feature_range

try:
    def stan_model(sequence, contaminant_db, prior_alpha=2, prior_beta=2):
        """Modified stan_model to handle both Bloom filter and regular databases"""
        if not HAS_STAN:
            return fallback_contamination_model(sequence, contaminant_db)
        
        # Handle empty database
        if isinstance(contaminant_db, set) and len(contaminant_db) < 2:
            return False
        
        # Prepare features with scaling, handling both set and BloomFilter
        if isinstance(contaminant_db, BloomFilter):
            features = np.array([1 if sequence in contaminant_db else 0])
        else:
            # Sample if regular database is too large
            db_sample = contaminant_db
            if isinstance(contaminant_db, set) and len(contaminant_db) > 10000:
                db_sample = np.random.choice(list(contaminant_db), 10000, replace=False)
            features = np.array([1 if c in sequence else 0 for c in db_sample])
        
        features = normalize_features(features)
        
        # Adjust sampling parameters based on database size
        n_samples = min(1000, max(100, len(contaminant_db) // 10))
        
        try:
            # Build and compile Stan model with more robust priors
            stan_code = """
            data {
                int<lower=0> N;
                vector[N] features;
                real<lower=0> alpha;
                real<lower=0> beta;
            }
            parameters {
                real<lower=0,upper=1> theta;
            }
            model {
                // More robust priors for extreme cases
                theta ~ beta(alpha + 1e-6, beta + 1e-6);
                features ~ bernoulli(theta + 1e-6);
            }
            """
            model = CmdStanModel(stan_code=stan_code)
            
            # Prepare data with bounds checking
            data = {
                'N': len(features), 
                'features': np.clip(features, 0, 1),  # Ensure valid range
                'alpha': max(prior_alpha, 1e-6),  # Prevent zero values
                'beta': max(prior_beta, 1e-6)
            }
            
            # Fit model with adaptive parameters
            fit = model.sample(
                data=data,
                iter_sampling=n_samples,
                iter_warmup=min(1000, n_samples),
                max_treedepth=8,
                adapt_delta=0.9
            )
            
            # Get result with bounds checking
            theta = fit.stan_variable('theta')
            if len(theta) == 0:
                return fallback_contamination_model(sequence, contaminant_db)
            
            return np.mean(theta) > 0.5
            
        except Exception as e:
            print(f"Stan model failed: {e}, falling back to logistic regression")
            return fallback_contamination_model(sequence, contaminant_db)
            
except ImportError:
    def stan_model(sequence, contaminant_db, prior_alpha=2, prior_beta=2):
        """Fallback version when Stan is not available"""
        return fallback_contamination_model(sequence, contaminant_db)

def fallback_contamination_model(sequence, contaminant_db):
    """Modified fallback model to handle Bloom filter"""
    features = extract_features(sequence)
    if not hasattr(fallback_contamination_model, 'model'):
        # Handle both set and BloomFilter cases
        if isinstance(contaminant_db, BloomFilter):
            X = np.array([[1 if sequence in contaminant_db else 0]])
        else:
            X = np.array([extract_features(s) for s in contaminant_db])
        y = np.ones(X.shape[0])
        
        # Add negative examples
        X_neg = np.array([extract_features(s) for s in generate_random_sequences(X.shape[0])])
        X = np.vstack([X, X_neg])
        y = np.hstack([y, np.zeros(X_neg.shape[0])])
        
        fallback_contamination_model.model = LogisticRegression()
        fallback_contamination_model.model.fit(X, y)
    
    return fallback_contamination_model.model.predict_proba([features])[0][1] > 0.5

def load_phylome(phylome_file, alignment_file):
    """Load phylogenetic tree and corresponding sequence alignment"""
    try:
        tree = PhyloTree(phylome_file, format=1)
        alignment = AlignIO.read(alignment_file, "fasta")
        return tree, alignment
    except Exception as e:
        warnings.warn(f"Error loading phylome: {e}")
        return None, None

def get_conservation_score(node, alignment):
    """Calculate conservation score for sequences in a given node"""
    if not node.is_leaf():
        seqs = [alignment[alignment.index(leaf.name)].seq for leaf in node.get_leaves()]
        if not seqs:
            return 0
        # Calculate column-wise conservation
        conservation = []
        for i in range(len(seqs[0])):
            column = [seq[i] for seq in seqs]
            conservation.append(len(set(column)) / len(column))
        return 1 - (sum(conservation) / len(conservation))
    return 0

def detect_evolutionary_events(sequence, reference, max_indel_size=10):
    """Detect potential evolutionary events in sequence compared to reference"""
    events = {
        'indels': [],
        'inversions': [],
        'duplications': [],
        'pseudogene_features': []
    }
    
    # Detect indels using pairwise alignment
    alignment = pairwise2.align.globalms(sequence, reference, 
                                       2, -1, -2, -0.5, 
                                       one_alignment_only=True)[0]
    
    # Process alignment to find indels
    seq_pos = ref_pos = 0
    for i, (seq_char, ref_char) in enumerate(zip(alignment[0], alignment[1])):
        if seq_char == '-' or ref_char == '-':
            if len(events['indels']) > 0 and i == events['indels'][-1][1] + 1:
                events['indels'][-1] = (events['indels'][-1][0], i)
            else:
                events['indels'].append((i, i))
    
    # Detect inversions
    for i in range(len(sequence)-10):
        for j in range(i+10, len(sequence)):
            subseq = sequence[i:j]
            if subseq == subseq[::-1]:
                events['inversions'].append((i, j))
    
    # Detect duplications
    for i in range(len(sequence)-5):
        for j in range(i+5, len(sequence)):
            subseq = sequence[i:j]
            if sequence.count(subseq) > 1:
                events['duplications'].append((i, j))
    
    # Check for pseudogene features
    events['pseudogene_features'] = detect_pseudogene_features(sequence)
    
    return events

def detect_pseudogene_features(sequence):
    """Detect features indicative of pseudogenization"""
    features = []
    
    # Look for premature stop codons
    for i in range(0, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        if codon in ['TAA', 'TAG', 'TGA']:
            features.append(('premature_stop', i))
    
    # Look for frameshift mutations
    for i in range(len(sequence)-3):
        if sequence[i:i+3] in ['ATG']:  # Start codon
            frame_shifts = check_frame_shifts(sequence[i:])
            features.extend([('frameshift', i+pos) for pos in frame_shifts])
    
    return features

def check_frame_shifts(seq):
    """Check for potential frameshift mutations"""
    shifts = []
    canonical_length = len(seq) - (len(seq) % 3)
    
    for i in range(0, canonical_length-3, 3):
        codon = seq[i:i+3]
        next_codon = seq[i+3:i+6]
        
        # Check for insertions/deletions that disrupt reading frame
        if codon in ['ATG', 'GTG'] and next_codon not in ['ATG', 'GTG']:
            if any(base not in 'ATGC' for base in next_codon):
                shifts.append(i+3)
    
    return shifts

def extract_confident_kmers(tree, alignment, k_size=31, confidence_threshold=0.8):
    """Extract k-mers accounting for evolutionary events"""
    confident_kmers = []
    event_tolerant_kmers = defaultdict(list)
    
    for node in tree.traverse():
        if not node.is_leaf():
            seqs = [alignment[alignment.index(leaf.name)].seq for leaf in node.get_leaves()]
            if not seqs:
                continue
                
            # Get consensus and check conservation
            conservation = get_conservation_score(node, alignment)
            if conservation >= confidence_threshold:
                consensus = get_consensus_sequence(seqs)
                
                # Extract k-mers considering evolutionary events
                for i in range(len(consensus) - k_size + 1):
                    kmer = consensus[i:i+k_size]
                    if 'N' not in kmer and '-' not in kmer:
                        # Check for evolutionary events in this region
                        events = detect_evolutionary_events(kmer, consensus[i:i+k_size])
                        
                        if events['indels'] or events['inversions'] or events['duplications']:
                            event_tolerant_kmers[kmer].append((conservation, events))
                        else:
                            confident_kmers.append((kmer, conservation))
    
    # Process event-tolerant k-mers
    for kmer, data in event_tolerant_kmers.items():
        avg_conservation = sum(c for c, _ in data) / len(data)
        if avg_conservation >= confidence_threshold:
            confident_kmers.append((kmer, avg_conservation))
    
    return confident_kmers

def get_consensus_sequence(sequences):
    """Get consensus sequence handling gaps and variations"""
    if not sequences:
        return ""
        
    consensus = []
    for i in range(len(sequences[0])):
        column = [seq[i] for seq in sequences if i < len(seq)]
        counts = Counter(column)
        # Handle gaps more permissively
        if '-' in counts and counts['-'] <= len(column) // 2:
            counts.pop('-')
        consensus.append(counts.most_common(1)[0][0])
    
    return ''.join(consensus)

def load_known_biological_elements():
    """Load well-known biological sequence elements"""
    elements = {
        'ori': [],  # Origins of replication
        'trna': [],  # tRNA sequences
        'crispr': [],  # CRISPR repeats
        'promoter': [],  # Core promoter elements
        'terminator': [],  # Transcription terminators
        'rrna': []  # rRNA sequences
    }
    
    # Common bacterial ori sequences
    elements['ori'].extend([
        "ATTTAATGATCCACAG",  # E. coli oriC
        "TTATCCACAGGGCAGG",  # B. subtilis oriC
        "TTGTCCACACTGGAAG"   # M. tuberculosis oriC
    ])
    
    # Common tRNA sequence patterns
    elements['trna'].extend([
        "GGTTCGATCC",  # tRNA 3' end
        "GTGGCNNAGT",  # tRNA D-arm
        "TTCGANNTC"    # tRNA anticodon arm
    ])
    
    # CRISPR repeat sequences
    elements['crispr'].extend([
        "GTTTTAGAGCTATGCT",  # SpCas9
        "ATTCCATTTAAAAAGG",  # Common bacterial repeat
        "GTTCCATTTGAAAGGG"   # Common archaeal repeat
    ])
    
    # Core promoter elements
    elements['promoter'].extend([
        "TATAAT",  # Bacterial -10 box
        "TTGACA",  # Bacterial -35 box
        "TATAAA"   # Eukaryotic TATA box
    ])
    
    # Transcription terminators
    elements['terminator'].extend([
        "GCGCCCGGCT",  # rho-independent terminator
        "TGCCTGGCAC",  # intrinsic terminator
    ])
    
    # rRNA conserved regions
    elements['rrna'].extend([
        "GAATTGACGGAAG",  # 16S rRNA
        "CACACCGCCCGT",   # 23S rRNA
        "GCTGGCACCAGA"    # 5S rRNA
    ])
    
    return elements

def extract_known_kmers(sequence, k_size=31):
    """Extract k-mers around known biological elements"""
    known_elements = load_known_biological_elements()
    known_kmers = []
    
    for element_type, patterns in known_elements.items():
        for pattern in patterns:
            # Find all occurrences of the pattern
            start = 0
            while True:
                pos = sequence.find(pattern, start)
                if pos == -1:
                    break
                    
                # Extract k-mer centered on the pattern
                start_pos = max(0, pos - (k_size - len(pattern))//2)
                end_pos = min(len(sequence), start_pos + k_size)
                
                if end_pos - start_pos == k_size:
                    kmer = sequence[start_pos:end_pos]
                    known_kmers.append((kmer, element_type))
                
                start = pos + 1
    
    return known_kmers

# Update create_kmer_dataset to use event-aware extraction
def create_kmer_dataset(fasta_file, phylome_file=None, alignment_file=None, k_size=31, 
                       negative_sample_size=None, event_tolerance=True, 
                       include_known_elements=True, hmm_db_path=None):
    """Create dataset with evolutionary event awareness"""
    X = []
    y = []
    biological_kmers = set()
    
    if (phylome_file and alignment_file and event_tolerance):
        tree, alignment = load_phylome(phylome_file, alignment_file)
        if tree and alignment:
            confident_kmers = extract_confident_kmers(tree, alignment, k_size)
            for kmer, confidence in confident_kmers:
                features = create_feature_vector(kmer, hmm_db_path=hmm_db_path)
                # Add evolutionary event features
                events = detect_evolutionary_events(kmer, kmer)  # Self-comparison for feature extraction
                features.extend([
                    len(events['indels']),
                    len(events['inversions']),
                    len(events['duplications']),
                    len(events['pseudogene_features'])
                ])
                X.append(features)
                y.append(1)
                biological_kmers.add(kmer)
    
    # Process additional sequences from fasta file
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        for i in range(len(seq) - k_size + 1):
            kmer = seq[i:i+k_size]
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                biological_kmers.add(kmer)
                features = extract_features(kmer)
                X.append(features)
                y.append(1)  # 1 for biological
    
    # Process known biological elements
    if include_known_elements:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq)
            known_kmers = extract_known_kmers(seq, k_size)
            
            for kmer, element_type in known_kmers:
                if 'N' not in kmer and kmer not in biological_kmers:
                    biological_kmers.add(kmer)
                    features = create_feature_vector(kmer, hmm_db_path=hmm_db_path)
                    # Add element type as feature
                    element_features = [1 if et == element_type else 0 
                                     for et in load_known_biological_elements().keys()]
                    features.extend(element_features)
                    X.append(features)
                    y.append(1)
    
    def generate_random_kmers(n, existing_kmers):
        bases = ['A', 'T', 'G', 'C']
        while n > 0:
            kmer = ''.join(np.random.choice(bases, k_size))
            if kmer not in existing_kmers:
                yield kmer
                n -= 1

    def parallel_feature_extraction(sequences):
        with ProcessPoolExecutor() as executor:
            return list(executor.map(extract_features, sequences))

    # Use generator for random k-mers
    random_kmers = list(generate_random_kmers(negative_sample_size, biological_kmers))
    random_features = parallel_feature_extraction(random_kmers)
    
    X.extend(random_features)
    y.extend([0] * len(random_features))
    
    return np.array(X), np.array(y)

def predict_kmer(clf, sequence, k_size=31):
    """Predict if a k-mer is biological or artifact"""
    # Validate k-mer length
    if len(sequence) != k_size:
        raise ValueError(f"K-mer must be of length {k_size}")
    
    # Validate nucleotide characters
    valid_bases = set('ATGC')
    if not set(sequence.upper()).issubset(valid_bases):
        raise ValueError("K-mer contains invalid characters. Only A,T,G,C are allowed")
        
    features = create_feature_vector(sequence)
    prediction = clf.predict([features])[0]
    return "Biological" if prediction == 1 else "Artifact"

# Update get_feature_names to include new features
def format_feature_name(name):
    """Format feature names for better visualization"""
    # Handle nucleotide frequency names
    if 'mer_' in name:
        n, bases = name.split('mer_')
        return f"{n}-Nucleotide Freq: {bases}"
    
    # Handle special cases
    name_map = {
        'gc_content': 'GC Content',
        'complexity': 'Sequence Complexity',
        'entropy': 'Shannon Entropy',
        'codon_bias': 'Codon Usage Bias',
        'repeat_content': 'Repeat Content',
        'palindrome_content': 'Palindrome Content',
        'frame_bias': 'Reading Frame Bias',
        'kmer_entropy_3': '3-mer Entropy',
        'kmer_entropy_4': '4-mer Entropy',
        'tata_box_score': 'TATA Box Signal',
        'splice_donor_score': 'Splice Donor Site',
        'splice_acceptor_score': 'Splice Acceptor Site',
        'kozak_score': 'Kozak Sequence',
        'poly_a_score': 'Poly-A Signal',
        'indel_count': 'Indel Events',
        'inversion_count': 'Inversion Events',
        'duplication_count': 'Duplication Events',
        'pseudogene_feature_count': 'Pseudogene Features',
        'repeat_count': 'Repeat Count',
        'repeat_type_diversity': 'Repeat Type Diversity',
        'mean_repeat_length': 'Mean Repeat Length',
        'gene_count': 'Gene Count',
        'exon_count': 'Exon Count',
        'mean_cds_length': 'Mean CDS Length',
        'trna_count': 'tRNA Count',
        'anticodon_diversity': 'Anticodon Diversity',
        'mean_trna_score': 'Mean tRNA Score',
        'phylogenetic_signal': 'Phylogenetic Signal',
        'selection_pressure': 'Selection Pressure',
        'convergence_probability': 'Convergence Probability',
        'orthogroup_size': 'Orthogroup Size',
        'paralog_count': 'Paralog Count',
        'duplication_score': 'Duplication Score',
        'synteny_score': 'Synteny Score',
        'sequence_cluster_density': 'Sequence Cluster Density',
        'best_hmm_score': 'Best HMM Score',
        'hmm_hit_count': 'HMM Hit Count',
        'mean_hmm_score': 'Mean HMM Score',
        'min_hmm_evalue': 'Min HMM E-value'
    }
    return name_map.get(name, name.replace('_', ' ').title())

def get_feature_names():
    """Get descriptive feature names including evolutionary events"""
    names = ['gc_content', 'complexity']
    
    # Add nucleotide frequency names
    for n in range(2,5):
        bases = ['A', 'T', 'G', 'C']
        names.extend([f'{n}mer_{"".join(combo)}' for combo in product(bases, repeat=n)])
    
    # Add basic feature names
    names.extend(['entropy', 'codon_bias'])
    
    # Add biological feature names
    names.extend([
        'repeat_content',
        'palindrome_content',
        'frame_bias',
        'kmer_entropy_3',
        'kmer_entropy_4',
        'tata_box_score',
        'splice_donor_score',
        'splice_acceptor_score',
        'kozak_score',
        'poly_a_score'
    ])
    
    # Add evolutionary event features
    names.extend([
        'indel_count',
        'inversion_count',
        'duplication_count',
        'pseudogene_feature_count'
    ])
    
    # Add biological element type features
    names.extend([f'{element}_sequence' 
                 for element in load_known_biological_elements().keys()])
    
    # Add prediction feature names
    names.extend([
        'repeat_count',
        'repeat_type_diversity',
        'mean_repeat_length',
        'gene_count',
        'exon_count',
        'mean_cds_length',
        'trna_count',
        'anticodon_diversity',
        'mean_trna_score',
        'phylogenetic_signal',
        'selection_pressure',
        'convergence_probability'
    ])
    
    # Add evolutionary relationship feature names
    names.extend([
        'orthogroup_size',
        'paralog_count',
        'duplication_score',
        'synteny_score',
        'sequence_cluster_density'
    ])
    
    # Add HMM feature names
    names.extend([
        'best_hmm_score',
        'hmm_hit_count',
        'mean_hmm_score',
        'min_hmm_evalue'
    ])
    
    # Format all names
    return [format_feature_name(name) for name in names]

def visualize_results(clf, X_test, y_test, feature_names=None):
    """Visualize model results and feature importance."""
    if feature_names is None:
        feature_names = get_feature_names()
    
    # Plot feature importance with improved formatting
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance, x='Importance', y='Feature')
    plt.title('Feature Importance in Biological Sequence Classification', pad=20)
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Add value labels
    for i, v in enumerate(importance['Importance']):
        plt.text(v, i, f'{v:.3f}', va='center')
    
    plt.show()

def visualize_phylogenetic_kmers(tree, confident_kmers):
    """Visualize k-mer distribution across phylogenetic tree"""
    ts = TreeStyle()
    ts.show_leaf_name = True
    
    def layout(node):
        if not node.is_leaf():
            node_kmers = [k for k, _ in confident_kmers if k in node.get_leaf_names()]
            if node_kmers:
                node.img_style["size"] = 10
                node.img_style["fgcolor"] = "red"
    
    ts.layout_fn = layout
    tree.show(tree_style=ts)

def evaluate_model(clf, X_test, y_test, feature_names=None):
    """Perform comprehensive model evaluation with visualizations.
    
    Args:
        clf: Trained classifier object
        X_test: Test feature matrix
        y_test: True test labels
        feature_names (list, optional): List of feature names for plotting
        
    Returns:
        dict: Evaluation metrics including:
            - accuracy (float): Overall accuracy
            - roc_auc (float): Area under ROC curve
            - pr_auc (float): Area under Precision-Recall curve
            - class_report (dict): Per-class precision, recall, F1
            - confusion_matrix (array): Confusion matrix
    
    Note:
        Creates visualization plots for:
        - ROC curve
        - Precision-Recall curve
        - Confusion matrix
        - Feature importance
    """
    # Get predictions and probabilities
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot ROC curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Receiver Operating Characteristic')
    axes[0, 0].legend(loc="lower right")
    
    # Plot Precision-Recall curve
    axes[0, 1].plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.2f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-bio', 'Bio'],
                yticklabels=['Non-bio', 'Bio'],
                ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    
    # Plot feature importance
    if feature_names is None:
        feature_names = get_feature_names()
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)  # Show top 10
    
    sns.barplot(data=importance, x='Importance', y='Feature', ax=axes[1, 1])
    axes[1, 1].set_title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print("\nDetailed Classification Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print("\nClass-wise Metrics:")
    for label in ['0', '1']:
        print(f"\nClass {label}:")
        print(f"Precision: {class_report[label]['precision']:.3f}")
        print(f"Recall: {class_report[label]['recall']:.3f}")
        print(f"F1-score: {class_report[label]['f1-score']:.3f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'class_report': class_report,
        'confusion_matrix': conf_matrix
    }

def run_external_predictions(sequence):
    """Run external bioinformatics tools for sequence analysis.
    
    Args:
        sequence (str): Input DNA sequence.
        
    Returns:
        dict: Results from multiple tools:
            - repeat_masker: Repeat element analysis
            - gene_prediction: Gene structure predictions
            - trna_scan: tRNA predictions
            - phyml_score: Phylogenetic analysis score
            - selection_score: Natural selection analysis
            - convergence_prob: Bayesian convergence probability
    
    Note:
        Requires external tools:
        - RepeatMasker
        - AUGUSTUS
        - tRNAscan-SE
        - PhyML
        - PAML
    """
    results = {
        'repeat_masker': None,
        'gene_prediction': None,
        'trna_scan': None,
        'phyml_score': None,
        'selection_score': None,
        'convergence_prob': None
    }
    
    # Create temporary file for sequence
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
        temp_file.write(f">seq\n{sequence}\n")
        temp_file.flush()
        
        # Run RepeatMasker
        try:
            result = run_subprocess(f"RepeatMasker -noint {temp_file.name}")
            results['repeat_masker'] = parse_repeatmasker_output(result.stdout)
        except SubprocessError as e:
            logger.error("RepeatMasker analysis failed", exc_info=True)
            results['repeat_masker'] = None
        
        # Run AUGUSTUS gene prediction
        try:
            result = run_subprocess(f"augustus --species=human {temp_file.name}")
            results['gene_prediction'] = parse_augustus_output(result.stdout)
        except SubprocessError as e:
            logger.error("AUGUSTUS analysis failed", exc_info=True)
            results['gene_prediction'] = None
        
        # Run tRNAscan-SE
        try:
            result = run_subprocess(f"tRNAscan-SE {temp_file.name}")
            results['trna_scan'] = parse_trnascan_output(result.stdout)
        except SubprocessError as e:
            logger.error("tRNAscan-SE analysis failed", exc_info=True)
            results['trna_scan'] = None
    
    # Run phylogenetic analysis
    try:
        results['phyml_score'] = analyze_phylogenetic_signal(sequence)
    except Exception as e:
        logger.error("PhyML analysis failed", exc_info=True)
        results['phyml_score'] = 0.0
    
    try:
        results['selection_score'] = detect_selection_footprints(sequence)
    except Exception as e:
        logger.error("Selection analysis failed", exc_info=True)
        results['selection_score'] = 1.0
    
    try:
        results['convergence_prob'] = assess_bayesian_convergence(sequence)
    except Exception as e:
        logger.error("Convergence analysis failed", exc_info=True)
        results['convergence_prob'] = 0.5
    
    return results

def parse_repeatmasker_output(output):
    """Parse RepeatMasker output and extract features"""
    features = {
        'repeat_count': 0,
        'repeat_types': set(),
        'repeat_lengths': []
    }
    
    for line in output.split('\n'):
        if line.startswith('SW'):
            features['repeat_count'] += 1
            parts = line.split()
            if len(parts) > 10:
                features['repeat_types'].add(parts[10])
                features['repeat_lengths'].append(int(parts[6]) - int(parts[5]))
    
    return features

def parse_augustus_output(output):
    """Parse AUGUSTUS gene prediction output"""
    features = {
        'gene_count': 0,
        'exon_count': 0,
        'cds_lengths': []
    }
    
    for line in output.split('\n'):
        if '\tgene\t' in line:
            features['gene_count'] += 1
        elif '\tCDS\t' in line:
            parts = line.split('\t')
            if len(parts) > 4:
                features['cds_lengths'].append(int(parts[4]) - int(parts[3]))
    
    return features

def parse_trnascan_output(output):
    """Parse tRNAscan-SE output"""
    features = {
        'trna_count': 0,
        'anticodon_types': set(),
        'scores': []
    }
    
    for line in output.split('\n'):
        if not line.startswith('#'):
            parts = line.split()
            if len(parts) > 4:
                features['trna_count'] += 1
                features['anticodon_types'].add(parts[4])
                features['scores'].append(float(parts[8]))
    
    return features

def analyze_phylogenetic_signal(sequence, reference_db="nr"):
    """Analyze phylogenetic signal using BLAST and PhyML"""
    try:
        # BLAST search
        result_handle = NCBIWWW.qblast("blastn", reference_db, sequence)
        blast_records = NCBIWWW.parse(result_handle)
        
        # Collect homologous sequences
        homologs = []
        for record in blast_records:
            for alignment in record.alignments:
                homologs.append(alignment.seq)
        
        if not homologs:
            return 0.0
        
        # Build distance matrix
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(homologs)
        
        # Construct tree
        constructor = DistanceTreeConstructor()
        tree = constructor.build_tree(dm)
        
        # Calculate phylogenetic signal
        return calculate_phylogenetic_signal(tree)
        
    except Exception as e:
        print(f"Phylogenetic analysis failed: {e}")
        return 0.0

def detect_selection_footprints(sequence):
    """Detect footprints of selection using dN/dS ratio"""
    try:
        # Create temporary alignment file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            temp_file.write(f">seq\n{sequence}\n")
            temp_file.flush()
            
            # Run PAML
            result = run_subprocess(f"codeml {temp_file.name}")
            return parse_paml_output(result.stdout)
            
    except SubprocessError as e:
        logger.error("PAML analysis failed", exc_info=True)
        return 1.0  # Neutral evolution as default

def assess_bayesian_convergence(sequence):
    """Assess Bayesian convergence in phylogenetic trees"""
    try:
        # Create DendroPy tree from sequence
        tree = dendropy.Tree.get_from_string(
            sequence,
            schema="newick",
            preserve_underscores=True
        )
        
        # Calculate convergence probability
        prob = calculate_convergence_probability(tree)
        return prob
        
    except Exception as e:
        print(f"Convergence analysis failed: {e}")
        return 0.5  # Neutral convergence as default

def calculate_phylogenetic_signal(tree):
    """Calculate phylogenetic signal using Blomberg's K"""
    try:
        # Convert tree to matrix form
        matrix = tree.cophenetic_matrix()
        
        # Calculate Blomberg's K
        k_stat = calculate_blomberg_k(matrix)
        return k_stat
        
    except Exception as e:
        print(f"Signal calculation failed: {e}")
        return 0.0

def calculate_convergence_probability(tree):
    """Calculate Bayesian convergence probability"""
    try:
        # Get tree statistics
        tree_len = tree.length()
        node_ages = [node.age for node in tree.nodes()]
        
        # Calculate convergence probability
        prob = 1.0 - chi2_contingency(node_ages)[1]
        return prob
        
    except Exception as e:
        print(f"Convergence calculation failed: {e}")
        return 0.5

def analyze_evolutionary_relationships(sequence, database_path):
    """Analyze evolutionary relationships of a sequence.
    
    Args:
        sequence (str): Input DNA sequence.
        database_path (str): Path to FASTA file containing reference sequences.
        
    Returns:
        dict: Dictionary containing:
            - orthogroup_size (int): Number of sequences in orthogroup
            - paralog_count (int): Number of identified paralogs
            - duplication_score (float): Score indicating duplication events
            - synteny_score (float): Measure of synteny conservation
            - cluster_density (float): Density of sequence clustering
    
    Note:
        Combines multiple evolutionary analyses including:
        - Orthogroup identification
        - Paralog detection
        - Synteny analysis
        - Sequence space clustering
    """
    features = {
        'orthogroup_size': 0,
        'paralog_count': 0,
        'duplication_score': 0.0,
        'synteny_score': 0.0,
        'cluster_density': 0.0
    }
    
    try:
        # Find orthogroups using OrthoFinder-like approach
        orthogroup = identify_orthogroup(sequence, database_path)
        features['orthogroup_size'] = len(orthogroup)
        
        # Analyze paralogs and duplications
        paralogs = find_paralogs(sequence, orthogroup)
        features['paralog_count'] = len(paralogs)
        
        # Calculate duplication scores
        features['duplication_score'] = calculate_duplication_score(sequence, paralogs)
        
        # Analyze synteny
        features['synteny_score'] = analyze_synteny(sequence, orthogroup)
        
        # Calculate sequence space clustering
        features['cluster_density'] = calculate_cluster_density(sequence, orthogroup)
        
        return features
    
    except Exception as e:
        print(f"Evolutionary analysis failed: {e}")
        return features

def identify_orthogroup(sequence, database_path):
    """Identify orthogroup members using sequence similarity"""
    orthogroup = []
    
    try:
        # Load sequence database
        sequences = list(SeqIO.parse(database_path, "fasta"))
        
        # Calculate similarity matrix
        distances = calculate_distance_matrix(sequence, sequences)
        
        # Cluster sequences
        Z = linkage(distances, method='average')
        clusters = fcluster(Z, t=0.7, criterion='distance')
        
        # Get sequences in same cluster as query
        query_cluster = clusters[0]
        orthogroup = [seq for i, seq in enumerate(sequences) 
                     if clusters[i] == query_cluster]
        
    except Exception as e:
        print(f"Orthogroup identification failed: {e}")
    
    return orthogroup

def find_paralogs(sequence, orthogroup):
    """Identify paralogs within orthogroup"""
    paralogs = []
    
    try:
        # Create sequence similarity network
        G = nx.Graph()
        
        # Add nodes and edges based on sequence similarity
        for seq1 in orthogroup:
            for seq2 in orthogroup:
                if seq1 != seq2:
                    similarity = calculate_sequence_similarity(seq1, seq2)
                    if similarity > 0.3:  # Similarity threshold
                        G.add_edge(seq1.id, seq2.id, weight=similarity)
        
        # Find connected components (paralog groups)
        paralog_groups = list(nx.connected_components(G))
        
        # Get paralogs for query sequence
        for group in paralog_groups:
            if sequence in group:
                paralogs = list(group - {sequence})
                break
                
    except Exception as e:
        print(f"Paralog identification failed: {e}")
    
    return paralogs

def calculate_duplication_score(sequence, paralogs):
    """Calculate duplication score based on synteny and similarity"""
    try:
        if not paralogs:
            return 0.0
            
        similarities = []
        for paralog in paralogs:
            # Calculate sequence similarity
            sim = calculate_sequence_similarity(sequence, paralog)
            
            # Calculate synteny conservation
            syn = calculate_synteny_conservation(sequence, paralog)
            
            # Combine metrics
            similarities.append((sim + syn) / 2)
        
        return np.mean(similarities)
        
    except Exception as e:
        print(f"Duplication score calculation failed: {e}")
        return 0.0

def analyze_synteny(sequence, orthogroup):
    """Analyze syntenic relationships"""
    try:
        # Create synteny blocks
        blocks = identify_synteny_blocks(sequence, orthogroup)
        
        # Calculate synteny conservation score
        conservation = calculate_synteny_conservation_score(blocks)
        
        return conservation
        
    except Exception as e:
        print(f"Synteny analysis failed: {e}")
        return 0.0

def calculate_cluster_density(sequence, orthogroup):
    """Calculate sequence space clustering density"""
    try:
        # Convert sequences to numerical vectors
        vectors = [seq_to_vector(seq) for seq in orthogroup]
        
        # Calculate pairwise distances
        distances = pdist(vectors)
        
        # Calculate clustering coefficient
        if len(distances) > 0:
            return 1.0 / (1.0 + np.mean(distances))
        return 0.0
        
    except Exception as e:
        print(f"Cluster density calculation failed: {e}")
        return 0.0

def seq_to_vector(sequence):
    """Convert sequence to numerical vector for clustering"""
    # Use k-mer frequencies as features
    k = 3
    kmers = [''.join(p) for p in product('ATGC', repeat=k)]
    counts = Counter([''.join(sequence[i:i+k]) 
                     for i in range(len(sequence)-k+1)])
    return [counts.get(kmer, 0) for kmer in kmers]

def calculate_distance_matrix(query_sequence, sequences):
    """Calculate pairwise distances between query and database sequences"""
    distances = []
    for seq in sequences:
        distance = calculate_sequence_distance(query_sequence, str(seq.seq))
        distances.append(distance)
    return np.array(distances)

def calculate_sequence_distance(seq1, seq2):
    """Calculate normalized edit distance between sequences"""
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    matches = sum(a == b for a, b in zip(alignment[0], alignment[1]))
    return 1 - (matches / max(len(seq1), len(seq2)))

def calculate_sequence_similarity(seq1, seq2):
    """Calculate sequence similarity score"""
    return 1 - calculate_sequence_distance(seq1, seq2)

def calculate_synteny_conservation(seq1, seq2):
    """Calculate synteny conservation score between sequences"""
    # Get neighboring genes
    neighbors1 = get_neighboring_genes(seq1)
    neighbors2 = get_neighboring_genes(seq2)
    
    # Calculate conservation
    shared = len(set(neighbors1) & set(neighbors2))
    total = len(set(neighbors1) | set(neighbors2))
    
    return shared / total if total > 0 else 0

def get_neighboring_genes(sequence, window=5000):
    """Get neighboring genes within window size"""
    neighbors = []
    try:
        # Run gene prediction
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            temp_file.write(f">seq\n{sequence}\n")
            temp_file.flush()
            
            result = run_subprocess(f"augustus --species=human {temp_file.name}")
            
            for line in result.stdout.split('\n'):
                if '\tgene\t' in line:
                    parts = line.split('\t')
                    if len(parts) > 4:
                        start = int(parts[3])
                        if start <= window:  # Within window
                            neighbors.append(parts[8])  # Gene ID
                            
    except SubprocessError as e:
        logger.error("Gene prediction failed", exc_info=True)
        
    return neighbors

def identify_synteny_blocks(sequence, orthogroup):
    """Identify syntenic blocks between sequences"""
    blocks = []
    try:
        for ortho_seq in orthogroup:
            # Get gene order
            genes1 = get_neighboring_genes(sequence)
            genes2 = get_neighboring_genes(str(ortho_seq.seq))
            
            # Find conserved blocks
            i = j = 0
            current_block = []
            while i < len(genes1) and j < len(genes2):
                if genes1[i] == genes2[j]:
                    current_block.append(genes1[i])
                    i += 1
                    j += 1
                else:
                    if len(current_block) >= 3:  # Minimum block size
                        blocks.append(current_block)
                    current_block = []
                    i += 1
                    j += 1
                    
            if len(current_block) >= 3:
                blocks.append(current_block)
                
    except Exception as e:
        print(f"Synteny block identification failed: {e}")
        
    return blocks

def calculate_synteny_conservation_score(blocks):
    """Calculate conservation score from synteny blocks"""
    if not blocks:
        return 0.0
    
    # Calculate average block size and coverage
    avg_size = np.mean([len(block) for block in blocks])
    total_genes = sum(len(block) for block in blocks)
    
    # Normalize score
    return (avg_size * total_genes) / 10000  # Arbitrary scaling

def calculate_blomberg_k(matrix):
    """Calculate Blomberg's K statistic"""
    try:
        # Convert distance matrix to phylogenetic signal
        n = len(matrix)
        if n < 3:
            return 0.0
            
        # Calculate phylogenetic variance
        phylo_var = np.sum(matrix) / (n * (n-1))
        
        # Calculate trait variance
        trait_var = np.var(np.diagonal(matrix))
        
        # Calculate K statistic
        if trait_var == 0:
            return 0.0
            
        k = phylo_var / trait_var
        
        return max(0.0, min(1.0, k))  # Bound between 0 and 1
        
    except Exception as e:
        print(f"Blomberg's K calculation failed: {e}")
        return 0.0

def parse_paml_output(output):
    """Parse PAML output to get dN/dS ratio"""
    try:
        for line in output.split('\n'):
            if 'dN/dS' in line:
                return float(line.split('=')[1].strip())
        return 1.0
    except:
        return 1.0

def sequence_generator(fasta_file, chunk_size=1000):
    """Generator to stream sequences from FASTA file.
    
    Args:
        fasta_file (str): Path to FASTA file
        chunk_size (int): Number of sequences to yield at once
        
    Yields:
        list: Chunk of sequences
    """
    chunk = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        chunk.append(str(record.seq))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def kmer_generator(sequence, k_size=31):
    """Generator for k-mers from a sequence.
    
    Args:
        sequence (str): Input sequence
        k_size (int): K-mer size
        
    Yields:
        str: K-mer sequence
    """
    for i in range(len(sequence) - k_size + 1):
        kmer = sequence[i:i+k_size]
        if 'N' not in kmer:
            yield kmer

def feature_generator(sequences, k_size=31):
    """Generator for feature vectors from sequences.
    
    Args:
        sequences (iterable): Input sequences
        k_size (int): K-mer size
        
    Yields:
        tuple: (features, label)
    """
    for sequence in sequences:
        for kmer in kmer_generator(sequence, k_size):
            features = create_feature_vector(kmer)
            yield features, 1  # 1 for biological sequence

def create_kmer_dataset(fasta_file, phylome_file=None, alignment_file=None, k_size=31, 
                       negative_sample_size=None, event_tolerance=True, 
                       include_known_elements=True, chunk_size=1000, hmm_db_path=None):
    """Create dataset with streaming data handling"""
    X, y = [], []
    biological_kmers = BloomFilter(1000000, 5)  # Use Bloom filter instead of set
    
    # Process sequences in chunks
    for chunk in sequence_generator(fasta_file, chunk_size):
        # Process each sequence in the chunk
        for features, label in feature_generator(chunk, k_size):
            X.append(features)
            y.append(label)
            
        # Generate negative examples for this chunk
        chunk_size = len(chunk)
        random_sequences = generate_random_sequences(chunk_size, k_size)
        for seq in random_sequences:
            if seq not in biological_kmers:  # Use Bloom filter for membership test
                features = create_feature_vector(seq, hmm_db_path=hmm_db_path)
                X.append(features)
                y.append(0)
                
        # Process in batches to avoid memory issues
        if len(X) >= chunk_size * 2:
            X_chunk = np.array(X)
            y_chunk = np.array(y)
            yield X_chunk, y_chunk
            X, y = [], []
    
    # Yield remaining data
    if X:
        yield np.array(X), np.array(y)

def train_kmer_classifier(hmm_db_path=None):
    """Train classifier using streaming data"""
    # Initialize SGD classifier instead of RandomForest
    clf = SGDClassifier(
        loss='log_loss',  # For probability estimates
        learning_rate='optimal',
        eta0=0.01,
        random_state=42
    )
    model_handler = ModelPersistence()
    
    # Train incrementally
    for i, (X_chunk, y_chunk) in enumerate(create_kmer_dataset("biological_sequences.fasta", hmm_db_path=hmm_db_path)):
        if i == 0:
            # First chunk - fit the model with classes
            clf.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
        else:
            # Subsequent chunks - update the model
            clf.partial_fit(X_chunk, y_chunk)
        
        print(f"Processed chunk {i+1}")
    
    # Save trained model with metadata
    metadata = {
        'training_date': datetime.datetime.now().isoformat(),
        'data_source': "biological_sequences.fasta",
        'model_version': '1.0',
        'model_type': 'SGDClassifier',
        'hmm_database': hmm_db_path if hmm_db_path else 'None'
    }
    model_path = model_handler.save_model(clf, metadata)
    
    return clf, model_path

def predict_kmer(sequence, k_size=31, model_path=None):
    """Predict using saved model"""
    model_handler = ModelPersistence()
    
    try:
        # Load model (latest if path not specified)
        clf = model_handler.load_model(model_path) if model_path else model_handler.load_latest_model()
        
        # Validate input
        if len(sequence) != k_size:
            raise ValueError(f"K-mer must be of length {k_size}")
        
        valid_bases = set('ATGC')
        if not set(sequence.upper()).issubset(valid_bases):
            raise ValueError("K-mer contains invalid characters. Only A,T,G,C are allowed")
        
        # Make prediction
        features = create_feature_vector(sequence)
        prediction = clf.predict([features])[0]
        return "Biological" if prediction == 1 else "Artifact"
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

class ModelPersistence:
    """Handle model saving and loading with versioning"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, metadata=None):
        """Save model with timestamp and metadata"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"kmer_model_{timestamp}.joblib")
        
        # Prepare metadata
        meta = {
            'timestamp': timestamp,
            'n_features': len(model.coef_[0]),  # Changed from feature_importances_
            'model_type': 'SGDClassifier',
            'user_metadata': metadata or {}
        }
        
        # Save model and metadata
        dump({'model': model, 'metadata': meta}, model_path)
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_latest_model(self):
        """Load most recent model"""
        models = sorted([f for f in os.listdir(self.model_dir) if f.startswith("kmer_model_")])
        if not models:
            raise FileNotFoundError("No saved models found")
        
        latest_model = os.path.join(self.model_dir, models[-1])
        saved_data = load(latest_model)
        print(f"Loaded model from {latest_model}")
        print(f"Model metadata: {saved_data['metadata']}")
        return saved_data['model']
    
    def load_model(self, model_path):
        """Load specific model by path"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        saved_data = load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Model metadata: {saved_data['metadata']}")
        return saved_data['model']

def parse_hmm_database(hmm_db_path):
    """Parse HMM database and extract profile information.
    
    Args:
        hmm_db_path (str): Path to HMM database file
        
    Returns:
        dict: Dictionary of HMM profiles with their metadata
    """
    profiles = {}
    try:
        # Use hmmfetch --index to create index if not present
        if not os.path.exists(f"{hmm_db_path}.h3i"):
            subprocess.run(['hmmfetch', '--index', hmm_db_path], check=True)
            
        # Parse HMM database
        with open(hmm_db_path) as hmm_file:
            for record in SearchIO.parse(hmm_file, 'hmmer3-text'):
                profiles[record.id] = {
                    'length': record.seq_len,
                    'description': record.description,
                    'cutoffs': {
                        'ga': record.ga,  # Gathering threshold
                        'tc': record.tc,  # Trusted cutoff
                        'nc': record.nc   # Noise cutoff
                    } if hasattr(record, 'ga') else None
                }
        return profiles
    except Exception as e:
        logger.error(f"Error parsing HMM database: {e}")
        return {}

def scan_sequence_with_hmm(sequence, hmm_db_path):
    """Scan sequence against HMM database using HMMER.
    
    Args:
        sequence (str): Input sequence
        hmm_db_path (str): Path to HMM database
        
    Returns:
        list: List of HMM hits with scores
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:
            temp_file.write(f">seq\n{sequence}\n")
            temp_file.flush()
            
            # Run hmmscan
            result = run_subprocess(
                f"hmmscan --cut_ga --domtblout /dev/stdout {hmm_db_path} {temp_file.name}",
                timeout=60
            )
            
            # Parse results
            hits = []
            for line in result.stdout.split('\n'):
                if line and not line.startswith('#'):
                    fields = line.split()
                    if len(fields) >= 13:
                        hits.append({
                            'hmm_name': fields[0],
                            'hmm_acc': fields[1],
                            'score': float(fields[7]),
                            'e_value': float(fields[6])
                        })
            return hits
    except Exception as e:
        logger.error(f"Error running HMM scan: {e}")
        return []

def get_hmm_features(sequence, hmm_db_path):
    """Extract HMM-based features from sequence.
    
    Args:
        sequence (str): Input sequence
        hmm_db_path (str): Path to HMM database
        
    Returns:
        dict: HMM features including best hit scores and coverage
    """
    hits = scan_sequence_with_hmm(sequence, hmm_db_path)
    
    features = {
        'best_hmm_score': 0.0,
        'hmm_hit_count': len(hits),
        'mean_hmm_score': 0.0,
        'min_hmm_evalue': float('inf')
    }
    
    if hits:
        scores = [hit['score'] for hit in hits]
        evalues = [hit['e_value'] for hit in hits]
        features.update({
            'best_hmm_score': max(scores),
            'mean_hmm_score': sum(scores) / len(scores),
            'min_hmm_evalue': min(evalues)
        })
    
    return features

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    logger.info("Starting biological k-mer analysis")
    
    try:
        # Train and save model
        classifier, model_path = train_kmer_classifier()
        logger.info(f"Model trained and saved to {model_path}")
        
        # Example prediction using saved model
        test_kmer = "ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
        result = predict_kmer(test_kmer, model_path=model_path)
        logger.info(f"Test k-mer classification: {result}")
        
        # Example loading latest model and predicting
        result = predict_kmer(test_kmer)
        logger.info(f"Test k-mer classification using latest model: {result}")
        
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        raise