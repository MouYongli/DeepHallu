"""
Test suite for DeepHallu package.
"""

import unittest
from deephallu import __version__
from deephallu.core import HallucinationDetector, HallucinationMitigator
from deephallu.models import StatisticalDetector, AttentionBasedMitigator
from deephallu.utils import setup_logging, calculate_metrics


class TestDeepHallu(unittest.TestCase):
    """Test cases for DeepHallu core functionality."""
    
    def test_version(self):
        """Test that version is properly defined."""
        self.assertEqual(__version__, "0.1.0")
    
    def test_detector_instantiation(self):
        """Test that detectors can be instantiated."""
        detector = StatisticalDetector()
        self.assertIsInstance(detector, HallucinationDetector)
    
    def test_mitigator_instantiation(self):
        """Test that mitigators can be instantiated."""
        mitigator = AttentionBasedMitigator()
        self.assertIsInstance(mitigator, HallucinationMitigator)
    
    def test_logging_setup(self):
        """Test logging configuration."""
        logger = setup_logging("INFO")
        self.assertIsNotNone(logger)
    
    def test_metrics_calculation(self):
        """Test metrics calculation function."""
        predictions = [{"score": 0.8}]
        ground_truth = [{"label": 1}]
        metrics = calculate_metrics(predictions, ground_truth)
        
        # Check that expected metrics are returned
        expected_keys = ["accuracy", "precision", "recall", "f1_score"]
        for key in expected_keys:
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()