#!/usr/bin/env python3
"""
Example usage of the DeepHallu framework.

This example demonstrates how to use DeepHallu for hallucination detection
and mitigation in a simple text processing scenario.
"""

from deephallu.models import StatisticalDetector, AttentionBasedMitigator
from deephallu.utils import setup_logging


def main():
    """Main example function."""
    # Set up logging
    logger = setup_logging("INFO")
    logger.info("Starting DeepHallu example")
    
    # Initialize detector and mitigator
    detector = StatisticalDetector(threshold=0.7)
    mitigator = AttentionBasedMitigator()
    
    # Example data (placeholder)
    input_text = "What is the capital of France?"
    model_output = "The capital of France is Paris."
    
    # Detect hallucinations
    logger.info("Running hallucination detection...")
    detection_result = detector.detect(input_text, model_output)
    logger.info(f"Detection result: {detection_result}")
    
    if detection_result["hallucination_score"] > detector.threshold:
        logger.warning("High hallucination risk detected!")
    else:
        logger.info("Output appears to be reliable.")
    
    logger.info("DeepHallu example completed successfully")


if __name__ == "__main__":
    main()