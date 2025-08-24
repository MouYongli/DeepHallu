"""
Utility functions for the DeepHallu framework.
"""

import json
import logging
from typing import Dict, Any, List


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for DeepHallu.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("deephallu")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")


def calculate_metrics(predictions: List[Dict[str, Any]], 
                     ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for hallucination detection.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        
    Returns:
        Dictionary of calculated metrics
    """
    # TODO: Implement comprehensive metrics calculation
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0
    }