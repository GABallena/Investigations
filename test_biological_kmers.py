import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from biological_kmers import (
    extract_features,
    create_feature_vector,
    generate_random_sequences,
    get_entropy,
    get_codon_bias,
    get_nucleotide_freq,
    get_repeat_content,
    get_kmer_entropy,
    get_palindrome_content,
    get_frame_bias,
    train_kmer_classifier,
    predict_kmer,
    parse_repeatmasker_output,
    parse_augustus_output,
    parse_trnascan_output,
    analyze_phylogenetic_signal,
    ModelPersistence,
    feature_generator
)

class TestBiologicalKmers(unittest.TestCase):
    def setUp(self):
        # Test sequences
        self.test_seq = "ATGCATGCATGC"
        self.simple_seq = "AAAA"
        self.complex_seq = "ATGCTAGCTAGCTAG"
        self.palindrome_seq = "ATGCGCAT"

    def test_extract_features(self):
        features = extract_features(self.test_seq)
        self.assertEqual(len(features), 2)  # GC content and complexity
        self.assertAlmostEqual(features[0], 0.5)  # GC content
        self.assertAlmostEqual(features[1], 0.333, places=3)  # Complexity

    def test_create_feature_vector(self):
        # Test standard case
        vector = create_feature_vector(self.test_seq)
        self.assertIsInstance(vector, list)
        self.assertTrue(all(isinstance(x, (int, float)) for x in vector))
        
        # Test with known sequence
        known_seq = "ATGCATGC"
        vector = create_feature_vector(known_seq)
        self.assertEqual(vector[0], 0.5)  # GC content should be 0.5
        
        # Test with empty sequence
        with self.assertRaises(ValueError):
            create_feature_vector("")

    def test_generate_random_sequences(self):
        n_seqs = 5
        length = 10
        seqs = generate_random_sequences(n_seqs, length)
        self.assertEqual(len(seqs), n_seqs)
        self.assertTrue(all(len(s) == length for s in seqs))
        self.assertTrue(all(set(s).issubset('ATGC') for s in seqs))

    def test_get_entropy(self):
        # Test uniform sequence
        self.assertEqual(get_entropy(self.simple_seq), 0.0)
        # Test diverse sequence
        self.assertGreater(get_entropy(self.complex_seq), 0.0)

    def test_get_codon_bias(self):
        # Test sequence shorter than 3
        self.assertEqual(get_codon_bias("AT"), 0)
        # Test normal sequence
        self.assertLessEqual(get_codon_bias(self.test_seq), 1.0)
        self.assertGreaterEqual(get_codon_bias(self.test_seq), 0.0)

    def test_get_nucleotide_freq(self):
        # Test dinucleotide frequencies
        freq = get_nucleotide_freq(self.test_seq, 2)
        self.assertIsInstance(freq, dict)
        self.assertEqual(sum(freq.values()), 1.0)

    def test_get_repeat_content(self):
        # Test sequence with repeats
        repeat_seq = "ATGATGATG"
        self.assertGreater(get_repeat_content(repeat_seq), 0.0)
        # Test sequence without repeats
        self.assertEqual(get_repeat_content(self.simple_seq), 1.0)

    def test_get_kmer_entropy(self):
        # Test with k=3
        entropy = get_kmer_entropy(self.test_seq, k=3)
        self.assertGreaterEqual(entropy, 0.0)
        # Test sequence shorter than k
        self.assertEqual(get_kmer_entropy("AT", k=3), 0.0)

    def test_get_palindrome_content(self):
        # Test palindrome sequence
        self.assertGreater(get_palindrome_content(self.palindrome_seq), 0.0)
        # Test non-palindrome sequence
        self.assertEqual(get_palindrome_content(self.simple_seq), 0.0)

    def test_get_frame_bias(self):
        # Test sequence with frame bias
        self.assertLessEqual(get_frame_bias(self.test_seq), 1.0)
        self.assertGreaterEqual(get_frame_bias(self.test_seq), 0.0)
        # Test short sequence
        self.assertEqual(get_frame_bias("AT"), 0)

    def test_edge_cases(self):
        # Test empty sequence
        self.assertEqual(get_entropy(""), 0.0)
        self.assertEqual(get_codon_bias(""), 0.0)
        self.assertEqual(len(generate_random_sequences(0)), 0)

    def test_input_validation(self):
        # Test invalid input
        with self.assertRaises(ValueError):
            extract_features("")
        with self.assertRaises(ValueError):
            get_nucleotide_freq(self.test_seq, 0)

    def test_train_kmer_classifier(self):
        try:
            # Train classifier
            clf, model_path = train_kmer_classifier()
            
            # Validate model structure and metrics
            self.assertIsInstance(clf, SGDClassifier)
            self.assertTrue(hasattr(clf, 'coef_'))
            self.assertTrue(isinstance(model_path, str))
            
            # Test incremental learning
            test_features = create_feature_vector(self.test_seq)
            prediction = clf.predict([test_features])
            self.assertIn(prediction[0], [0, 1])
            
        except ImportError as e:
            self.skipTest(f"Skipping due to missing dependencies: {str(e)}")
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    def test_predict_kmer(self):
        try:
            # Test valid k-mer
            valid_kmer = "A" * 31  # Create valid length k-mer
            result = predict_kmer(valid_kmer)
            self.assertIn(result, ["Biological", "Artifact"])
            
            # Test invalid k-mer length
            with self.assertRaises(ValueError):
                predict_kmer("AT")
                
            # Test invalid characters
            with self.assertRaises(ValueError):
                predict_kmer("ATGCN" * 6 + "A")
                
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    def test_external_tool_validation(self):
        # Test RepeatMasker output parsing
        repeat_output = """
        SW    perc perc perc  query     position in query    matching  repeat      position in repeat
        score  div. del. ins.  sequence  begin  end          repeat    class/family  begin  end
        
        1234   11.5  6.2  0.0  Seq1     100    200         AluY      SINE/Alu      1    100
        """
        features = parse_repeatmasker_output(repeat_output)
        self.assertEqual(features['repeat_count'], 1)
        self.assertEqual(len(features['repeat_types']), 1)
        self.assertEqual(features['repeat_lengths'][0], 100)
        
        # Test AUGUSTUS output parsing
        augustus_output = """
        # start gene g1
        Seq1    AUGUSTUS    gene    1    1000    1    +    .    g1
        Seq1    AUGUSTUS    CDS     1    500     1    +    0    g1
        """
        features = parse_augustus_output(augustus_output)
        self.assertEqual(features['gene_count'], 1)
        self.assertEqual(len(features['cds_lengths']), 1)
        self.assertEqual(features['cds_lengths'][0], 499)
        
        # Test tRNAscan-SE output parsing
        trna_output = """
        Sequence    tRNA    Bounds  tRNA    Anti    Intron Bounds   Inf
        Name        #      Begin    End     Type    Codon   Begin    End     Score
        --------    ----  ----    ----    ----    -----   ----    ----    -----
        Seq1        1     1000    1072    Leu     CAA     0        0       87.6
        """
        features = parse_trnascan_output(trna_output)
        self.assertEqual(features['trna_count'], 1)
        self.assertEqual(len(features['anticodon_types']), 1)
        self.assertEqual(features['scores'][0], 87.6)

    def test_phylogenetic_analysis(self):
        test_seq = "ATGCATGCATGC"
        signal = analyze_phylogenetic_signal(test_seq)
        self.assertGreaterEqual(signal, 0.0)
        self.assertLessEqual(signal, 1.0)
        
        # Test with known conserved sequence
        conserved_seq = "ATGGCCAAGTAA"  # Common start-stop pattern
        signal = analyze_phylogenetic_signal(conserved_seq)
        self.assertGreater(signal, 0.5)

    def test_model_persistence(self):
        try:
            # Create and fit a test model using SGD instead of RandomForest
            clf = SGDClassifier(random_state=42, loss='log_loss')  # Changed to SGD
            X = np.array([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]])
            y = np.array([1, 1, 0])
            clf.fit(X, y)
            
            # Test model saving and loading
            model_handler = ModelPersistence(model_dir="test_models")
            metadata = {'test': True}
            model_path = model_handler.save_model(clf, metadata)
            
            # Load and verify model
            loaded_clf = model_handler.load_model(model_path)
            np.testing.assert_array_equal(
                loaded_clf.predict([[0.5, 0.5]]),
                clf.predict([[0.5, 0.5]])
            )
            
            # Test loading latest model
            latest_clf = model_handler.load_latest_model()
            np.testing.assert_array_equal(
                latest_clf.predict([[0.5, 0.5]]),
                clf.predict([[0.5, 0.5]])
            )
            
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")
        finally:
            # Cleanup test models directory
            import shutil
            import os
            if os.path.exists("test_models"):
                shutil.rmtree("test_models")

    def test_feature_generation(self):
        # Test feature generator
        sequences = ["ATGCATGC", "GCTAGCTA"]
        generator = feature_generator(sequences)
        features, label = next(generator)
        
        self.assertIsInstance(features, list)
        self.assertIsInstance(label, int)
        self.assertEqual(label, 1)
        
        # Test with invalid sequence
        with self.assertRaises(ValueError):
            next(feature_generator(["ATGN"]))

    def test_incremental_learning(self):
        """Test incremental learning capabilities"""
        try:
            # Create test data chunks
            X1 = np.array([[0.5, 0.5], [0.3, 0.7]])
            y1 = np.array([1, 1])
            X2 = np.array([[0.8, 0.2], [0.1, 0.9]])
            y2 = np.array([0, 0])
            
            # Initialize incremental classifier
            clf = SGDClassifier(random_state=42)
            
            # First chunk
            clf.partial_fit(X1, y1, classes=np.array([0, 1]))
            pred1 = clf.predict(X1)
            self.assertTrue(all(pred1 == y1))
            
            # Second chunk
            clf.partial_fit(X2, y2)
            pred2 = clf.predict(X2)
            self.assertTrue(all(pred2 == y2))
            
            # Test probability predictions
            probs = clf.predict_proba(X1)
            self.assertEqual(probs.shape, (2, 2))  # Two classes
            self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))
            
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
