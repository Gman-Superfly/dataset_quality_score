#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Utilities for Dataset Quality Scoring

Authors: Oscar Goldman
Date: March 1, 2025
Purpose: Utility functions for synthetic data generation and visualization

Description:
This module provides functions for generating synthetic datasets and creating
visualizations for the Dataset Quality Scorer system. Includes data generation
functions and plotting utilities for analysis and validation.
"""

__all__ = [
    'generate_synthetic_data', 
    'plot_dataset_2d', 
    'plot_score_components', 
    'plot_diversity_scaling',
    'plot_weight_sensitivity',
    'plot_rl_progress',
    'load_data_from_polars',
    'polars_to_torch',
    'torch_to_polars',
    'DataGenerationError'
]

from typing import Tuple, List, Optional, Dict, Any, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Optional Polars import with graceful fallback
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


class DataGenerationError(Exception):
    """Base exception for data generation errors."""
    pass


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int = 2,
    outlier_ratio: float = 0.05,
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate synthetic data with features, labels, and outliers.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_classes: Number of classes for labels
        outlier_ratio: Fraction of samples to make outliers
        random_seed: Random seed for reproducibility
        
    Returns:
        Tensor of shape (n_samples, n_features + 1) with last column as labels
        
    Raises:
        DataGenerationError: If parameters are invalid
    """
    # Input validation
    assert n_samples > 0, f"n_samples must be positive, got {n_samples}"
    assert n_features > 0, f"n_features must be positive, got {n_features}"
    assert n_classes >= 1, f"n_classes must be at least 1, got {n_classes}"
    assert 0 <= outlier_ratio <= 1, f"outlier_ratio must be in [0, 1], got {outlier_ratio}"
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    try:
        # Generate random features
        data = torch.randn(n_samples, n_features)
        assert data.shape == (n_samples, n_features), "Data shape mismatch after generation"
        
        # Generate labels based on feature sum (simple decision boundary)
        if n_classes == 2:
            labels = (data.sum(dim=1) > 0).float()
        else:
            # Multi-class: use feature sum modulo n_classes
            feature_sums = data.sum(dim=1)
            labels = (feature_sums - feature_sums.min()) / (feature_sums.max() - feature_sums.min() + 1e-8)
            labels = (labels * n_classes).long().float() % n_classes
        
        assert labels.shape == (n_samples,), f"Labels shape mismatch: {labels.shape}"
        assert torch.all(labels >= 0) and torch.all(labels < n_classes), "Invalid label values"
        
        # Create outliers by shifting some data points
        n_outliers = int(n_samples * outlier_ratio)
        if n_outliers > 0:
            outlier_indices = torch.randperm(n_samples)[:n_outliers]
            data[outlier_indices] += 5.0  # Shift outliers away from main distribution
        
        # Combine features and labels
        result = torch.cat([data, labels.unsqueeze(1)], dim=1)
        assert result.shape == (n_samples, n_features + 1), f"Result shape mismatch: {result.shape}"
        
        return result
        
    except Exception as e:
        raise DataGenerationError(f"Failed to generate synthetic data: {e}")


def plot_dataset_2d(
    data: torch.Tensor,
    title: str = "Dataset Visualization",
    feature_x: int = 0,
    feature_y: int = 1,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize dataset in 2D projection.
    
    Args:
        data: Dataset tensor with features and labels
        title: Plot title
        feature_x: Index of feature for x-axis
        feature_y: Index of feature for y-axis
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
        
    Raises:
        DataGenerationError: If plotting fails
    """
    assert isinstance(data, torch.Tensor), f"Expected torch.Tensor, got {type(data)}"
    assert data.dim() == 2, f"Data must be 2D, got {data.dim()}D"
    assert data.size(1) >= 3, f"Data must have at least 3 columns (2 features + labels), got {data.size(1)}"
    assert 0 <= feature_x < data.size(1) - 1, f"feature_x {feature_x} out of bounds"
    assert 0 <= feature_y < data.size(1) - 1, f"feature_y {feature_y} out of bounds"
    assert feature_x != feature_y, "feature_x and feature_y must be different"
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract features and labels
        x_vals = data[:, feature_x].numpy()
        y_vals = data[:, feature_y].numpy()
        labels = data[:, -1].numpy()
        
        # Create scatter plot
        scatter = ax.scatter(x_vals, y_vals, c=labels, cmap="viridis", alpha=0.6)
        
        # Customize plot
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"Feature {feature_x}", fontsize=12)
        ax.set_ylabel(f"Feature {feature_y}", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Label", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise DataGenerationError(f"Failed to create 2D plot: {e}")


def plot_score_components(
    components: Dict[str, float],
    title: str = "Dataset Quality Components",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize score components in a bar chart.
    
    Args:
        components: Dictionary of component names and scores
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
        
    Raises:
        DataGenerationError: If plotting fails
    """
    assert isinstance(components, dict), f"Expected dict, got {type(components)}"
    assert len(components) > 0, "Components dictionary cannot be empty"
    assert all(isinstance(v, (int, float)) for v in components.values()), "All component values must be numeric"
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Default colors for common components
        color_map = {
            "diversity": "#4CAF50",
            "edge_coverage": "#2196F3",
            "balance": "#FF9800",
            "relevance": "#9C27B0",
            "total_score": "#F44336"
        }
        
        names = list(components.keys())
        values = list(components.values())
        colors = [color_map.get(name.lower(), "#607D8B") for name in names]
        
        # Create bar chart
        bars = ax.bar(names, values, color=colors)
        
        # Customize plot
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, max(1.0, max(values) * 1.1))
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=10)
        
        # Rotate labels if too many components
        if len(names) > 4:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise DataGenerationError(f"Failed to create components plot: {e}")


def plot_diversity_scaling(
    sample_sizes: List[int],
    diversity_scores: List[float],
    title: str = "Diversity Score vs. Sample Size",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot diversity score scaling with sample size.
    
    Args:
        sample_sizes: List of sample sizes
        diversity_scores: List of corresponding diversity scores
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
        
    Raises:
        DataGenerationError: If plotting fails
    """
    assert len(sample_sizes) == len(diversity_scores), "Sample sizes and scores must have same length"
    assert len(sample_sizes) > 0, "Must have at least one data point"
    assert all(isinstance(s, int) and s > 0 for s in sample_sizes), "All sample sizes must be positive integers"
    assert all(isinstance(d, (int, float)) for d in diversity_scores), "All diversity scores must be numeric"
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create line plot
        ax.plot(sample_sizes, diversity_scores, marker="o", color="#03A9F4", linewidth=2, markersize=6)
        
        # Customize plot
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Sample Size", fontsize=12)
        ax.set_ylabel("Diversity Score", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(0, max(sample_sizes) * 1.05)
        ax.set_ylim(0, max(diversity_scores) * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise DataGenerationError(f"Failed to create diversity scaling plot: {e}")


def plot_weight_sensitivity(
    weight_configs: List[str],
    scores: List[float],
    title: str = "Score Sensitivity to Weight Configurations",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot score sensitivity to different weight configurations.
    
    Args:
        weight_configs: List of weight configuration names
        scores: List of corresponding scores
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
        
    Raises:
        DataGenerationError: If plotting fails
    """
    assert len(weight_configs) == len(scores), "Weight configs and scores must have same length"
    assert len(weight_configs) > 0, "Must have at least one configuration"
    assert all(isinstance(s, (int, float)) for s in scores), "All scores must be numeric"
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar chart
        bars = ax.bar(weight_configs, scores, color="#4CAF50", alpha=0.7)
        
        # Customize plot
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, max(scores) * 1.1)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(scores) * 0.01,
                   f"{score:.3f}", ha='center', va='bottom', fontsize=10)
        
        # Rotate labels if needed
        if len(weight_configs) > 3:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise DataGenerationError(f"Failed to create weight sensitivity plot: {e}")


def plot_rl_progress(
    dataset_sizes: List[int],
    rewards: List[float],
    title: str = "RL Agent Progress",
    figsize: Tuple[int, int] = (10, 6),
    max_size_line: Optional[int] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot RL agent's progress over time.
    
    Args:
        dataset_sizes: List of dataset sizes at each step
        rewards: List of corresponding rewards
        title: Plot title
        figsize: Figure size tuple
        max_size_line: Optional vertical line to mark maximum size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
        
    Raises:
        DataGenerationError: If plotting fails
    """
    assert len(dataset_sizes) == len(rewards), "Dataset sizes and rewards must have same length"
    assert len(dataset_sizes) > 0, "Must have at least one data point"
    assert all(isinstance(s, int) and s >= 0 for s in dataset_sizes), "All dataset sizes must be non-negative integers"
    assert all(isinstance(r, (int, float)) for r in rewards), "All rewards must be numeric"
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create line plot
        ax.plot(dataset_sizes, rewards, marker="o", color="#03A9F4", 
               linewidth=2, markersize=4, alpha=0.7, label="RL Reward (S)")
        
        # Add max size line if specified
        if max_size_line is not None:
            ax.axvline(x=max_size_line, color="gray", linestyle="--", 
                      alpha=0.5, label=f"Max Size ({max_size_line})")
        
        # Customize plot
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Dataset Size", fontsize=12)
        ax.set_ylabel("Reward (Score)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        
        # Set reasonable axis limits
        ax.set_xlim(0, max(dataset_sizes) * 1.05)
        reward_range = max(rewards) - min(rewards)
        ax.set_ylim(min(rewards) - reward_range * 0.1, max(rewards) + reward_range * 0.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise DataGenerationError(f"Failed to create RL progress plot: {e}")


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    assert isinstance(seed, int), f"Seed must be integer, got {type(seed)}"
    assert seed >= 0, f"Seed must be non-negative, got {seed}"
    
    torch.manual_seed(seed)
    np.random.seed(seed)


def validate_tensor_data(data: torch.Tensor, min_samples: int = 1, min_features: int = 1) -> None:
    """
    Validate tensor data format and constraints.
    
    Args:
        data: Tensor to validate
        min_samples: Minimum number of samples required
        min_features: Minimum number of features required
        
    Raises:
        DataGenerationError: If data is invalid
    """
    assert isinstance(data, torch.Tensor), f"Expected torch.Tensor, got {type(data)}"
    assert data.dim() == 2, f"Data must be 2D, got {data.dim()}D"
    assert data.size(0) >= min_samples, f"Need at least {min_samples} samples, got {data.size(0)}"
    assert data.size(1) >= min_features, f"Need at least {min_features} features, got {data.size(1)}"
    assert torch.all(torch.isfinite(data)), "Data contains non-finite values"


def load_data_from_polars(
    file_path: str,
    feature_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None,
    convert_to_torch: bool = True
) -> Union[torch.Tensor, 'pl.DataFrame']:
    """
    Load data from various formats using Polars for efficient data handling.
    
    Args:
        file_path: Path to data file (CSV, Parquet, JSON, etc.)
        feature_columns: List of feature column names (None for all except label)
        label_column: Name of label column (None if no labels)
        convert_to_torch: Whether to convert to PyTorch tensor
        
    Returns:
        PyTorch tensor or Polars DataFrame with features and optionally labels
        
    Raises:
        DataGenerationError: If Polars is not available or loading fails
    """
    if not POLARS_AVAILABLE:
        raise DataGenerationError(
            "Polars is not installed. Install with 'pip install polars>=0.20.0'"
        )
    
    assert isinstance(file_path, str), f"Expected string file path, got {type(file_path)}"
    
    try:
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pl.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pl.read_json(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pl.read_excel(file_path)
        else:
            # Try CSV as default
            df = pl.read_csv(file_path)
        
        assert df.height > 0, f"Loaded data is empty: {file_path}"
        assert df.width > 0, f"Loaded data has no columns: {file_path}"
        
        # Select feature columns
        if feature_columns is not None:
            # Validate all feature columns exist
            missing_cols = set(feature_columns) - set(df.columns)
            assert len(missing_cols) == 0, f"Missing feature columns: {missing_cols}"
            feature_df = df.select(feature_columns)
        else:
            # Use all columns except label column
            if label_column is not None:
                assert label_column in df.columns, f"Label column '{label_column}' not found"
                feature_df = df.select([col for col in df.columns if col != label_column])
            else:
                feature_df = df
        
        # Add label column if specified
        if label_column is not None:
            assert label_column in df.columns, f"Label column '{label_column}' not found"
            combined_df = feature_df.with_columns(df.select(label_column))
        else:
            combined_df = feature_df
        
        if not convert_to_torch:
            return combined_df
        
        # Convert to PyTorch tensor
        return polars_to_torch(combined_df)
        
    except Exception as e:
        raise DataGenerationError(f"Failed to load data from {file_path}: {e}")


def polars_to_torch(df: 'pl.DataFrame') -> torch.Tensor:
    """
    Convert Polars DataFrame to PyTorch tensor.
    
    Args:
        df: Polars DataFrame with numeric data
        
    Returns:
        PyTorch tensor of shape (n_rows, n_cols)
        
    Raises:
        DataGenerationError: If conversion fails
    """
    if not POLARS_AVAILABLE:
        raise DataGenerationError("Polars is not available")
    
    assert isinstance(df, pl.DataFrame), f"Expected Polars DataFrame, got {type(df)}"
    assert df.height > 0, "DataFrame is empty"
    assert df.width > 0, "DataFrame has no columns"
    
    try:
        # Convert to numpy first, then to torch
        # Polars .to_numpy() is efficient for numeric data
        numpy_array = df.to_numpy()
        
        # Ensure all data is numeric
        assert numpy_array.dtype.kind in ['i', 'u', 'f'], f"Non-numeric data detected: {numpy_array.dtype}"
        
        tensor = torch.from_numpy(numpy_array).float()
        
        # Validate result
        assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D"
        assert torch.all(torch.isfinite(tensor)), "Converted data contains non-finite values"
        
        return tensor
        
    except Exception as e:
        raise DataGenerationError(f"Failed to convert Polars DataFrame to torch: {e}")


def torch_to_polars(
    tensor: torch.Tensor, 
    column_names: Optional[List[str]] = None
) -> 'pl.DataFrame':
    """
    Convert PyTorch tensor to Polars DataFrame.
    
    Args:
        tensor: PyTorch tensor of shape (n_rows, n_cols)
        column_names: Optional column names (defaults to "feature_0", "feature_1", etc.)
        
    Returns:
        Polars DataFrame
        
    Raises:
        DataGenerationError: If conversion fails
    """
    if not POLARS_AVAILABLE:
        raise DataGenerationError("Polars is not available")
    
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
    assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D"
    assert torch.all(torch.isfinite(tensor)), "Tensor contains non-finite values"
    
    try:
        # Convert to numpy
        numpy_array = tensor.detach().cpu().numpy()
        
        # Generate column names if not provided
        if column_names is None:
            column_names = [f"feature_{i}" for i in range(tensor.size(1))]
        else:
            assert len(column_names) == tensor.size(1), f"Column names length mismatch: {len(column_names)} vs {tensor.size(1)}"
        
        # Create Polars DataFrame
        df = pl.DataFrame(numpy_array, schema=column_names)
        
        assert df.height == tensor.size(0), "Row count mismatch after conversion"
        assert df.width == tensor.size(1), "Column count mismatch after conversion"
        
        return df
        
    except Exception as e:
        raise DataGenerationError(f"Failed to convert torch tensor to Polars: {e}") 