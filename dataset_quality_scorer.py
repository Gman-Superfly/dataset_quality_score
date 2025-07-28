#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Quality Scorer: Optimizing Datasets for RL Teacher/Student Hypothetical

Authors: Oscar Goldman
Date: March 1, 2025
Purpose: Demonstrate and validate a sublinear monotonicity score for dataset quality,
         tailored for RL-driven dataset generation to train language models.

Description:
This module introduces the `DatasetQualityScorer`, computing S = f(Q), where
Q = w₁D + w₂E + w₃B + w₄R, evaluating datasets based on four components:
    - Diversity (D): Measures data point spread for varied examples
    - Edge-case Coverage (E): Ensures rare/challenging examples for robustness
    - Balance (B): Assesses class/category distribution evenness
    - Relevance (R): Gauges alignment with target domain reference data

The score S is sublinear (e.g., log(1 + Q) or Q^0.5) to model diminishing returns,
ideal for RL agents crafting datasets.
"""

__all__ = ['DatasetQualityScorer', 'DatasetQualityScorerError', 'ScoreComponents', 'WEIGHT_PRESETS', 'get_available_weight_presets']

from typing import Optional, Tuple, Literal, Dict, Union
import torch
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.covariance import MinCovDet
from dataclasses import dataclass


class DatasetQualityScorerError(Exception):
    """Base exception for DatasetQualityScorer errors."""
    pass


# Domain-adaptive weight presets for different use cases
WEIGHT_PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    "balanced": (0.25, 0.25, 0.25, 0.25),           # General purpose, all components equal
    "diversity_focused": (0.5, 0.2, 0.15, 0.15),   # For feature learning and representation
    "balance_critical": (0.15, 0.15, 0.6, 0.1),    # For imbalanced domains (medical, fraud)
    "relevance_heavy": (0.1, 0.15, 0.15, 0.6),     # For domain adaptation and transfer learning
    "edge_case_priority": (0.2, 0.5, 0.15, 0.15),  # For robustness testing and edge case coverage
    "production_safe": (0.2, 0.3, 0.3, 0.2),       # Conservative weights for production systems
    "research_exploration": (0.4, 0.3, 0.1, 0.2),  # For research datasets emphasizing diversity
}


def get_available_weight_presets() -> Dict[str, Tuple[float, float, float, float]]:
    """
    Get all available weight presets with their descriptions.
    
    Returns:
        Dictionary mapping preset names to weight tuples
    """
    return WEIGHT_PRESETS.copy()


@dataclass
class ScoreComponents:
    """Container for individual score components with validation."""
    diversity: float
    edge_coverage: float
    balance: float
    relevance: float
    total_score: float
    
    def __post_init__(self) -> None:
        """Validate all components are in valid range."""
        assert 0 <= self.diversity <= 1, f"Invalid diversity: {self.diversity}"
        assert 0 <= self.edge_coverage <= 1, f"Invalid edge_coverage: {self.edge_coverage}"
        assert 0 <= self.balance <= 1, f"Invalid balance: {self.balance}"
        assert 0 <= self.relevance <= 1, f"Invalid relevance: {self.relevance}"
        assert self.total_score >= 0, f"Invalid total_score: {self.total_score}"


class DatasetQualityScorer:
    """
    Computes a sublinear monotonicity score for dataset quality: S = f(Q),
    where Q = w₁*D + w₂*E + w₃*B + w₄*R.

    Features:
    - Sublinear function f(Q) for diminishing returns (log or sqrt)
    - Efficient subsampling for large datasets
    - Tailored for classification/feature-based datasets (e.g., embeddings + labels)
    """

    def __init__(
        self,
        weights: Union[Tuple[float, float, float, float], str] = "balanced",
        epsilon: float = 0.1,
        sublinear_func: Literal["log", "sqrt"] = "log",
    ) -> None:
        """
        Initialize scorer with weights, sampling tolerance, and sublinear function.

        Args:
            weights: Weight preset name or tuple (w_D, w_E, w_B, w_R), summing to ~1
                Available presets: 'balanced', 'diversity_focused', 'balance_critical',
                'relevance_heavy', 'edge_case_priority', 'production_safe', 'research_exploration'
            epsilon: Sampling error tolerance (0 < ε ≤ 1), smaller = more samples
            sublinear_func: 'log' for log(1 + Q) or 'sqrt' for Q^0.5
            
        Raises:
            DatasetQualityScorerError: If parameters are invalid
        """
        # Resolve weight presets
        if isinstance(weights, str):
            assert weights in WEIGHT_PRESETS, f"Unknown weight preset '{weights}'. Available: {list(WEIGHT_PRESETS.keys())}"
            weights = WEIGHT_PRESETS[weights]
        
        # Validate weights to ensure they are appropriate
        assert len(weights) == 4, f"Expected 4 weights, got {len(weights)}"
        assert all(0 <= w <= 1 for w in weights), f"All weights must be in [0,1], got {weights}"
        assert abs(sum(weights) - 1.0) < 1e-5, f"Weights must sum to ~1, got sum={sum(weights)}"
        
        # Ensure epsilon is within valid range for sampling
        assert 0 < epsilon <= 1, f"Epsilon must be in (0, 1], got {epsilon}"
        
        # Validate sublinear function
        assert sublinear_func in ["log", "sqrt"], f"Invalid sublinear_func: {sublinear_func}"
        
        self.weights = weights
        self.epsilon = epsilon
        self.sublinear_func = sublinear_func

    @staticmethod
    def _compute_adaptive_thresholds(data: torch.Tensor, val_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute domain-specific thresholds based on data characteristics.
        
        Args:
            data: Main dataset to analyze
            val_data: Validation dataset for additional context
            
        Returns:
            Dictionary with adaptive threshold values
        """
        assert isinstance(data, torch.Tensor), f"Expected torch.Tensor, got {type(data)}"
        assert data.size(0) > 0, "Data must have at least one sample"
        
        n_samples = data.size(0)
        n_features = data.size(1) - 1 if data.size(1) > 1 else data.size(1)
        
        # Adaptive outlier percentile: smaller datasets need less aggressive outlier detection
        # Formula: 95 - sqrt(log(n)) to range from ~85 for small data to ~92 for large data
        outlier_percentile = max(85.0, min(95.0, 95.0 - np.sqrt(np.log(max(n_samples, 2)))))
        
        # Adaptive rare label threshold: should be inversely related to dataset size
        # But not less than 1/n_samples (at least one example) or more than 10%
        rare_label_threshold = max(1.0 / n_samples, min(0.1, 2.0 / np.sqrt(n_samples)))
        
        # Adaptive diversity normalization scale based on feature dimensionality and data spread
        if n_features > 0:
            feature_data = data[:, :-1] if data.size(1) > 1 else data
            data_ranges = feature_data.max(dim=0)[0] - feature_data.min(dim=0)[0]
            avg_range = data_ranges.mean().item()
            # Use actual data spread when available, fallback to sqrt(n_features)
            diversity_scale = max(avg_range, np.sqrt(n_features) * 0.1)
        else:
            diversity_scale = 1.0
        
        # Adaptive coverage threshold based on feature scale
        if n_features > 0:
            feature_data = data[:, :-1] if data.size(1) > 1 else data
            feature_scales = torch.std(feature_data, dim=0)
            avg_scale = feature_scales.mean().item()
            coverage_threshold = max(0.5, min(2.0, avg_scale))
        else:
            coverage_threshold = 1.0
        
        return {
            "outlier_percentile": outlier_percentile,
            "rare_label_threshold": rare_label_threshold,
            "diversity_scale": diversity_scale,
            "coverage_threshold": coverage_threshold,
        }

    def _compute_diversity(self, data: torch.Tensor) -> float:
        """
        Calculate diversity as average pairwise distance, subsampled for efficiency.
        
        Args:
            data: Feature tensor of shape (n_samples, n_features)
            
        Returns:
            Diversity score in [0, 1]
            
        Raises:
            DatasetQualityScorerError: If data is invalid
        """
        assert isinstance(data, torch.Tensor), f"Expected torch.Tensor, got {type(data)}"
        assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D"
        
        n = data.size(0)
        # Return 0 if too few data points to measure spread
        if n < 2:
            return 0.0
            
        # Use Hoeffding's inequality to determine efficient sample size
        sample_size = min(n, max(10, int(np.ceil((2 / self.epsilon**2) * np.log(n)))))
        assert sample_size > 0, "Invalid sample size calculation"
        
        idx = torch.randperm(n)[:sample_size]
        sampled_data = data[idx]
        
        dists = torch.cdist(sampled_data, sampled_data, p=2)
        # Extract upper triangular part (excluding diagonal) for pairwise distances
        triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        pairwise_distances = dists[triu_mask]
        
        # Handle edge case with no pairwise distances
        if pairwise_distances.numel() == 0:
            return 0.0
            
        avg_dist = pairwise_distances.mean().item()
        
        # Use adaptive normalization based on data characteristics
        thresholds = self._compute_adaptive_thresholds(data)
        max_dist = thresholds["diversity_scale"]
        
        assert max_dist > 0, f"Invalid max distance calculation: {max_dist}"
        
        diversity = min(avg_dist / max_dist, 1.0)
        assert 0 <= diversity <= 1, f"Diversity out of bounds: {diversity}"
        
        return diversity

    def _compute_edge_coverage(self, data: torch.Tensor, val_data: torch.Tensor) -> float:
        """
        Calculate edge-case coverage for rare labels and feature outliers.
        
        Args:
            data: Training data tensor with features and labels
            val_data: Validation data tensor with features and labels
            
        Returns:
            Edge coverage score in [0, 1]
            
        Raises:
            DatasetQualityScorerError: If data is invalid
        """
        assert isinstance(data, torch.Tensor), f"Expected torch.Tensor for data, got {type(data)}"
        assert isinstance(val_data, torch.Tensor), f"Expected torch.Tensor for val_data, got {type(val_data)}"
        assert data.size(1) == val_data.size(1), "Data and val_data must have same number of features"
        assert data.size(1) >= 2, "Data must have at least 2 columns (features + labels)"
        
        # Extract labels for rarity analysis
        data_labels = data[:, -1].long()
        val_labels = val_data[:, -1].long()
        
        label_counts = torch.bincount(val_labels).float()
        assert label_counts.numel() > 0, "No labels found in validation data"
        
        # Use adaptive rare label threshold instead of mean
        thresholds = self._compute_adaptive_thresholds(val_data)
        total_samples = label_counts.sum().item()
        rare_threshold = thresholds["rare_label_threshold"] * total_samples
        rare_labels = (label_counts < rare_threshold).nonzero(as_tuple=True)[0]
        
        covered_labels = len(set(data_labels.tolist()) & set(rare_labels.tolist()))
        label_coverage = covered_labels / max(rare_labels.numel(), 1)
        
        # Detect feature outliers using robust covariance
        val_features = val_data[:, :-1].numpy()
        assert val_features.shape[0] > 0, "No validation features found"
        
        try:
            # Only proceed if we have enough samples for robust covariance
            if val_features.shape[0] < 2:
                feature_coverage = 1.0  # Complete coverage for single sample
            else:
                mcd = MinCovDet().fit(val_features)
                mahalanobis_dist = mcd.mahalanobis(val_features)
                
                # Use adaptive outlier threshold
                thresholds = self._compute_adaptive_thresholds(val_data)
                outlier_threshold = np.percentile(mahalanobis_dist, thresholds["outlier_percentile"])
                outliers = val_features[mahalanobis_dist > outlier_threshold]
                
                # Handle case with no outliers
                if outliers.shape[0] == 0:
                    feature_coverage = 1.0
                else:
                    data_features = data[:, :-1].numpy()
                    if data_features.shape[0] == 0:
                        feature_coverage = 0.0
                    else:
                        # Use adaptive coverage threshold
                        thresholds = self._compute_adaptive_thresholds(data)
                        coverage_threshold = thresholds["coverage_threshold"]
                        
                        dists = np.min(np.linalg.norm(data_features[:, np.newaxis] - outliers, axis=2), axis=1)
                        covered_outliers = np.sum(dists < coverage_threshold)
                        feature_coverage = covered_outliers / outliers.shape[0]
        except Exception as e:
            # More informative fallback
            feature_coverage = 0.5
            
        # Ensure both components are properly bounded before combining
        label_coverage = max(0.0, min(1.0, label_coverage))
        feature_coverage = max(0.0, min(1.0, feature_coverage))
        
        edge_coverage = (label_coverage + feature_coverage) / 2
        assert 0 <= edge_coverage <= 1, f"Edge coverage out of bounds: {edge_coverage}"
        
        return edge_coverage

    def _compute_balance(self, data: torch.Tensor) -> float:
        """
        Calculate balance as normalized entropy of label distribution.
        
        Args:
            data: Data tensor with last column as labels
            
        Returns:
            Balance score in [0, 1]
            
        Raises:
            DatasetQualityScorerError: If data is invalid
        """
        assert isinstance(data, torch.Tensor), f"Expected torch.Tensor, got {type(data)}"
        
        # Default to neutral score if no labels
        if data.size(1) < 2:
            return 0.5
            
        labels = data[:, -1].long()
        n_classes = labels.max().item() + 1
        
        # Maximum balance for single class
        if n_classes < 2:
            return 1.0
            
        counts = torch.bincount(labels, minlength=n_classes).float()
        assert counts.sum() > 0, "No valid labels found"
        
        probs = counts / counts.sum()
        
        # More numerically stable entropy calculation
        # Only compute log for non-zero probabilities
        mask = probs > 0
        entropy = 0.0
        if mask.any():
            log_probs = torch.log(probs[mask])
            entropy = -torch.sum(probs[mask] * log_probs)
        
        max_entropy = np.log(n_classes)
        
        assert max_entropy > 0, "Invalid max entropy calculation"
        
        balance = entropy / max_entropy
        assert 0 <= balance <= 1, f"Balance out of bounds: {balance}"
        
        return balance.item()

    def _compute_relevance(self, data: torch.Tensor, ref_data: torch.Tensor) -> float:
        """
        Calculate relevance using Wasserstein distance to reference data.
        
        Args:
            data: Feature data tensor
            ref_data: Reference feature data tensor
            
        Returns:
            Relevance score in [0, 1]
            
        Raises:
            DatasetQualityScorerError: If data is invalid
        """
        assert isinstance(data, torch.Tensor), f"Expected torch.Tensor for data, got {type(data)}"
        assert isinstance(ref_data, torch.Tensor), f"Expected torch.Tensor for ref_data, got {type(ref_data)}"
        assert data.size(1) == ref_data.size(1), "Data and ref_data must have same number of features"
        
        data_features = data.numpy()
        ref_features = ref_data.numpy()
        
        w_distances = []
        for i in range(data.size(1)):
            try:
                w_dist = wasserstein_distance(data_features[:, i], ref_features[:, i])
                w_distances.append(w_dist)
            except Exception as e:
                # Fallback distance if Wasserstein fails
                w_distances.append(1.0)
                
        assert len(w_distances) > 0, "No valid Wasserstein distances computed"
        
        avg_w_distance = np.mean(w_distances)
        
        # More robust normalization for relevance
        feature_scales = []
        for i in range(data.size(1)):
            data_std = np.std(data_features[:, i])
            ref_std = np.std(ref_features[:, i])
            # Use range if std is too small
            if data_std < 1e-8:
                data_range = np.ptp(data_features[:, i])  # peak-to-peak
                data_std = max(data_range, 1e-8)
            if ref_std < 1e-8:
                ref_range = np.ptp(ref_features[:, i])
                ref_std = max(ref_range, 1e-8)
            feature_scales.append(data_std + ref_std)
        
        max_distance = np.max(feature_scales)
        
        # Handle edge case where max_distance is 0 (all features constant)
        if max_distance < 1e-8:
            relevance = 1.0  # Maximum relevance if both datasets are constant
        else:
            relevance = 1 - min(avg_w_distance / max_distance, 1.0)
            
        assert 0 <= relevance <= 1, f"Relevance out of bounds: {relevance}"
        
        return relevance

    def _apply_sublinear_function(self, q_value: float) -> float:
        """
        Apply sublinear transformation to combined quality score.
        
        Args:
            q_value: Combined quality score Q
            
        Returns:
            Transformed score S = f(Q)
            
        Raises:
            DatasetQualityScorerError: If sublinear function is invalid
        """
        assert q_value >= 0, f"Q value must be non-negative, got {q_value}"
        
        if self.sublinear_func == "log":
            result = np.log(1 + q_value)
        elif self.sublinear_func == "sqrt":
            result = q_value**0.5
        else:
            raise DatasetQualityScorerError(f"Unsupported sublinear function: {self.sublinear_func}")
            
        assert result >= 0, f"Sublinear result must be non-negative, got {result}"
        
        return result

    def compute_score(
        self,
        data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        ref_data: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute the full quality score S = f(Q).
        
        Args:
            data: Main dataset with features and labels
            val_data: Validation dataset for edge case analysis (optional)
            ref_data: Reference dataset for relevance analysis (optional)
            
        Returns:
            Final quality score S
            
        Raises:
            DatasetQualityScorerError: If computation fails
        """
        assert isinstance(data, torch.Tensor), f"Expected torch.Tensor for data, got {type(data)}"
        assert data.size(0) > 0, "Data must have at least one sample"
        assert data.size(1) > 0, "Data must have at least one feature"
        
        features = data[:, :-1] if data.size(1) > 1 else data
        w1, w2, w3, w4 = self.weights
        
        D = self._compute_diversity(features)
        E = self._compute_edge_coverage(data, val_data) if val_data is not None else 0.5
        B = self._compute_balance(data)
        R = self._compute_relevance(features, ref_data) if ref_data is not None else 0.5
        
        # Validate component scores
        assert 0 <= D <= 1, f"Invalid diversity score: {D}"
        assert 0 <= E <= 1, f"Invalid edge coverage score: {E}"
        assert 0 <= B <= 1, f"Invalid balance score: {B}"
        assert 0 <= R <= 1, f"Invalid relevance score: {R}"
        
        Q = w1 * D + w2 * E + w3 * B + w4 * R
        assert Q >= 0, f"Combined score Q must be non-negative, got {Q}"
        
        S = self._apply_sublinear_function(Q)
        assert S >= 0, f"Final score S must be non-negative, got {S}"
        
        return S

    def compute_detailed_score(
        self,
        data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        ref_data: Optional[torch.Tensor] = None,
    ) -> ScoreComponents:
        """
        Compute detailed breakdown of all score components.
        
        Args:
            data: Main dataset with features and labels
            val_data: Validation dataset for edge case analysis (optional)
            ref_data: Reference dataset for relevance analysis (optional)
            
        Returns:
            ScoreComponents with all individual scores and total
            
        Raises:
            DatasetQualityScorerError: If computation fails
        """
        assert isinstance(data, torch.Tensor), f"Expected torch.Tensor for data, got {type(data)}"
        assert data.size(0) > 0, "Data must have at least one sample"
        assert data.size(1) > 0, "Data must have at least one feature"
        
        features = data[:, :-1] if data.size(1) > 1 else data
        w1, w2, w3, w4 = self.weights
        
        D = self._compute_diversity(features)
        E = self._compute_edge_coverage(data, val_data) if val_data is not None else 0.5
        B = self._compute_balance(data)
        R = self._compute_relevance(features, ref_data) if ref_data is not None else 0.5
        
        Q = w1 * D + w2 * E + w3 * B + w4 * R
        S = self._apply_sublinear_function(Q)
        
        return ScoreComponents(
            diversity=D,
            edge_coverage=E,
            balance=B,
            relevance=R,
            total_score=S
        ) 