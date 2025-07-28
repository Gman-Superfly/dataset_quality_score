#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Quality Scorer Package

Authors: Oscar Goldman
Date: March 1, 2025
Purpose: Package initialization for Dataset Quality Scorer

This package provides tools for evaluating dataset quality using sublinear
monotonicity scoring, designed for RL-driven dataset generation.
"""

__version__ = "1.0.0"
__author__ = "Oscar Goldman"


from .dataset_quality_scorer import (
    DatasetQualityScorer,
    DatasetQualityScorerError,
    ScoreComponents,
)

from .rl_environment import (
    DatasetGenerationEnv,
    DatasetGenerationError,
)

from .data_utils import (
    generate_synthetic_data,
    plot_dataset_2d,
    plot_score_components,
    plot_diversity_scaling,
    plot_weight_sensitivity,
    plot_rl_progress,
    set_random_seeds,
    validate_tensor_data,
    DataGenerationError,
)

__all__ = [
    # Core classes
    "DatasetQualityScorer",
    "DatasetGenerationEnv",
    "ScoreComponents",
    
    # Utility functions
    "generate_synthetic_data",
    "plot_dataset_2d",
    "plot_score_components",
    "plot_diversity_scaling",
    "plot_weight_sensitivity",
    "plot_rl_progress",
    "set_random_seeds",
    "validate_tensor_data",
    
    # Exceptions
    "DatasetQualityScorerError",
    "DatasetGenerationError",
    "DataGenerationError",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
] 