"""
DeepHallu: A deep learning framework for hallucination detection and mitigation.

This package provides tools and algorithms for:
- Detecting hallucinations in deep learning model outputs
- Implementing mitigation strategies to reduce hallucination rates
- Evaluating model reliability across different domains

Author: Yongli Mou
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Yongli Mou"
__email__ = ""

# Import modules to make them available
from . import core
from . import models
from . import utils

__all__ = [
    "__version__",
    "__author__",
]