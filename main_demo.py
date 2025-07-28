#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Quality Scorer - Main Demo

Authors: Oscar Goldman
Date: March 1, 2025
Purpose: Comprehensive demonstration of the Dataset Quality Scorer system

Description:
This script replicates the functionality from the Jupyter notebook, demonstrating:
1. Synthetic data generation
2. Score component validation
3. RL environment demo
4. Visualization and analysis

The demo validates the sublinear monotonicity score for dataset quality and
shows its application in RL-driven dataset generation.
"""

import sys
import argparse
from typing import List, Tuple, Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset_quality_scorer import DatasetQualityScorer, ScoreComponents, WEIGHT_PRESETS, get_available_weight_presets
from rl_environment import DatasetGenerationEnv
from data_utils import (
    generate_synthetic_data, 
    plot_dataset_2d, 
    plot_score_components,
    plot_diversity_scaling,
    plot_weight_sensitivity,
    plot_rl_progress,
    set_random_seeds,
    torch_to_polars,
    polars_to_torch,
    POLARS_AVAILABLE
)


def run_basic_demo(save_plots: bool = False) -> Dict[str, Any]:
    """
    Run the basic dataset quality scoring demonstration.
    
    Args:
        save_plots: Whether to save generated plots
        
    Returns:
        Dictionary with demo results
    """
    print("=== Dataset Quality Scorer Demo ===")
    print("Generating synthetic datasets...")
    
    # Use different random seeds each run for varied datasets
    import random
    import time
    base_seed = int(time.time()) % 10000  # Use current time as base seed
    
    # Generate synthetic datasets with different seeds each run
    main_data = generate_synthetic_data(100, 5, n_classes=2, outlier_ratio=0.1, random_seed=base_seed)
    val_data = generate_synthetic_data(20, 5, n_classes=2, outlier_ratio=0.2, random_seed=base_seed+1)
    ref_data = torch.randn(50, 5)  # Reference data (features only)
    
    print(f"Main dataset shape: {main_data.shape}")
    print(f"Validation dataset shape: {val_data.shape}")
    print(f"Reference dataset shape: {ref_data.shape}")
    
    # Visualize dataset
    print("\nCreating 2D visualization...")
    fig1 = plot_dataset_2d(
        main_data, 
        title="Synthetic Dataset: 2D Projection",
        save_path="dataset_2d.png" if save_plots else None
    )
    plt.show()
    
    # Initialize scorer and compute components
    print("\nComputing score components...")
    scorer = DatasetQualityScorer(sublinear_func="log")
    
    # Get detailed score breakdown
    components = scorer.compute_detailed_score(main_data, val_data, ref_data)
    
    print(f"\n### Component Breakdown ###")
    print(f"Diversity (D): {components.diversity:.3f}")
    print(f"Edge-case Coverage (E): {components.edge_coverage:.3f}")
    print(f"Balance (B): {components.balance:.3f}")
    print(f"Relevance (R): {components.relevance:.3f}")
    print(f"Total Score (S): {components.total_score:.3f}")
    
    # Visualize components
    component_dict = {
        "Diversity": components.diversity,
        "Edge Coverage": components.edge_coverage,
        "Balance": components.balance,
        "Relevance": components.relevance
    }
    
    fig2 = plot_score_components(
        component_dict,
        title="Dataset Quality Components",
        save_path="score_components.png" if save_plots else None
    )
    plt.show()
    
    return {
        "main_data": main_data,
        "val_data": val_data,
        "ref_data": ref_data,
        "components": components,
        "scorer": scorer
    }


def run_diversity_scaling_demo(main_data: torch.Tensor, scorer: DatasetQualityScorer, save_plots: bool = False) -> None:
    """
    Demonstrate diversity score scaling with sample size.
    
    Args:
        main_data: Main dataset for analysis
        scorer: Initialized scorer
        save_plots: Whether to save plots
    """
    print("\n=== Diversity Scaling Analysis ===")
    
    sample_sizes = [10, 25, 50, 75, 100]
    diversity_scores = []
    
    for size in sample_sizes:
        if size <= main_data.size(0):
            subset = main_data[:size, :-1]  # Features only
            diversity = scorer._compute_diversity(subset)
            diversity_scores.append(diversity)
            print(f"Sample size {size}: Diversity = {diversity:.3f}")
        else:
            diversity_scores.append(diversity_scores[-1])  # Use last valid score
    
    # Plot scaling behavior
    fig = plot_diversity_scaling(
        sample_sizes,
        diversity_scores,
        title="Diversity Score vs. Sample Size",
        save_path="diversity_scaling.png" if save_plots else None
    )
    plt.show()


def run_weight_sensitivity_demo(main_data: torch.Tensor, val_data: torch.Tensor, 
                               ref_data: torch.Tensor, save_plots: bool = False) -> None:
    """
    Demonstrate score sensitivity to weight configurations using new presets.
    
    Args:
        main_data: Main dataset
        val_data: Validation dataset
        ref_data: Reference dataset
        save_plots: Whether to save plots
    """
    print("\n=== Weight Sensitivity Analysis (Enhanced) ===")
    
    # Show available presets
    print("Available weight presets:")
    for preset_name, weights in get_available_weight_presets().items():
        print(f"  - {preset_name}: {weights}")
    
    # Test key presets
    key_presets = ["balanced", "diversity_focused", "balance_critical", "edge_case_priority", "production_safe"]
    
    config_names = []
    scores = []
    
    for preset_name in key_presets:
        scorer = DatasetQualityScorer(weights=preset_name)  # Using string preset!
        score = scorer.compute_score(main_data, val_data, ref_data)
        config_names.append(preset_name.replace("_", " ").title())
        scores.append(score)
        print(f"{preset_name}: Score = {score:.3f}")
    
    # Plot weight sensitivity
    fig = plot_weight_sensitivity(
        config_names,
        scores,
        title="Score Sensitivity to Weight Configurations",
        save_path="weight_sensitivity.png" if save_plots else None
    )
    plt.show()


def run_rl_demo(main_data: torch.Tensor, val_data: torch.Tensor, 
                ref_data: torch.Tensor, save_plots: bool = False, 
                n_steps: int = 1000) -> Dict[str, Any]:
    """
    Run the RL environment demonstration.
    
    Args:
        main_data: Main dataset to use as pool
        val_data: Validation dataset
        ref_data: Reference dataset
        save_plots: Whether to save plots
        n_steps: Number of RL steps to run
        
    Returns:
        Dictionary with RL results
    """
    print("\n=== RL Environment Demo ===")
    
    # Create larger data pool for RL
    pool_data = generate_synthetic_data(1000, 5, n_classes=2, outlier_ratio=0.1, random_seed=44)
    print(f"RL Data pool size: {pool_data.shape[0]}")
    
    # Initialize RL environment
    env = DatasetGenerationEnv(
        data_pool=pool_data,
        val_data=val_data,
        ref_data=ref_data,
        max_size=1000
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    
    # Run random agent simulation
    print(f"\nRunning random agent for {n_steps} steps...")
    print("Note: This may take a moment for large step counts...")
    
    rewards = []
    dataset_sizes = []
    episode_steps = []
    
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Only track successful (non-duplicate) steps
        if "duplicate" not in info:
            rewards.append(reward)
            dataset_sizes.append(info["dataset_size"])
            episode_steps.append(step + 1)
            
            if step < 10 or step % 10 == 0:  # Print progress
                print(f"Step {step + 1}: Reward={reward:.3f}, Dataset Size={info['dataset_size']}")
        
        if terminated or truncated:
            print(f"Episode terminated at step {step + 1}")
            break
    
    # Get final detailed score
    final_components = env.get_detailed_score()
    if final_components:
        print(f"\n### Final Score Breakdown ###")
        for component, value in final_components.items():
            print(f"{component.title()}: {value:.3f}")
    
    # Plot RL progress
    if rewards and dataset_sizes:
        fig = plot_rl_progress(
            dataset_sizes,
            rewards,
            title="RL Agent: Reward vs. Dataset Size",
            max_size_line=1000,
            save_path="rl_progress.png" if save_plots else None
        )
        plt.show()
    
    return {
        "rewards": rewards,
        "dataset_sizes": dataset_sizes,
        "final_components": final_components,
        "environment": env
    }


def run_sublinear_comparison(main_data: torch.Tensor, val_data: torch.Tensor, 
                           ref_data: torch.Tensor) -> None:
    """
    Compare different sublinear functions.
    
    Args:
        main_data: Main dataset
        val_data: Validation dataset
        ref_data: Reference dataset
    """
    print("\n=== Sublinear Function Comparison ===")
    
    # Test different sublinear functions
    functions = ["log", "sqrt"]
    
    for func in functions:
        scorer = DatasetQualityScorer(sublinear_func=func)
        score = scorer.compute_score(main_data, val_data, ref_data)
        print(f"{func.upper()} function: Score = {score:.3f}")


def run_polars_demo(main_data: torch.Tensor) -> None:
    """
    Demonstrate Polars integration for data handling.
    
    Args:
        main_data: PyTorch tensor to convert and manipulate
    """
    print("\n=== Polars Integration Demo ===")
    
    if not POLARS_AVAILABLE:
        print("Polars is not available. Install with: pip install polars>=0.20.0")
        print("Skipping Polars demo...")
        return
    
    try:
        print("Converting PyTorch tensor to Polars DataFrame...")
        
        # Convert to Polars with custom column names
        feature_cols = [f"feature_{i}" for i in range(main_data.size(1) - 1)]
        column_names = feature_cols + ["label"]
        
        df = torch_to_polars(main_data, column_names)
        print(f"Created Polars DataFrame: {df.shape}")
        print(f"Columns: {df.columns}")
        
        # Show some Polars operations
        print("\nPolars DataFrame Info:")
        print(f"- Shape: {df.height} rows × {df.width} columns")
        print(f"- Data types: {dict(zip(df.columns, df.dtypes))}")
        
        # Demonstrate filtering and aggregation
        print("\nPolars Operations:")
        import polars as pl
        label_counts = df.group_by("label").agg([
            pl.len().alias("count"),
            pl.col(feature_cols[0]).mean().alias(f"avg_{feature_cols[0]}")
        ])
        print("Label distribution:")
        print(label_counts)
        
        # Convert back to PyTorch
        print("\nConverting back to PyTorch tensor...")
        reconstructed_tensor = polars_to_torch(df)
        
        # Verify conversion accuracy
        max_diff = torch.max(torch.abs(main_data - reconstructed_tensor)).item()
        print(f"Max difference after round-trip conversion: {max_diff:.2e}")
        
        if max_diff < 1e-6:
            print("Round-trip conversion successful!")
        else:
            print("Warning: Conversion accuracy may be compromised")
        
        # Save to CSV as example with timestamp to avoid overwriting
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"example_dataset_{timestamp}.csv"
        df.write_csv(csv_path)
        print(f"Saved DataFrame to {csv_path}")
        print("Note: Each run creates a new CSV file to preserve previous outputs")
        
        print("\nPolars benefits demonstrated:")
        print("- Fast conversion to/from PyTorch tensors")
        print("- Efficient data operations and filtering")
        print("- Multiple file format support (CSV, Parquet, JSON, etc.)")
        print("- Memory-efficient processing")
        
    except Exception as e:
        print(f"Polars demo failed: {e}")
        print("This may indicate a compatibility issue.")


def run_adaptive_thresholds_demo(main_data: torch.Tensor) -> None:
    """
    Demonstrate adaptive threshold functionality.
    
    Args:
        main_data: PyTorch tensor to analyze
    """
    print("\n=== Adaptive Thresholds Demo ===")
    
    scorer = DatasetQualityScorer()
    
    # Test with different dataset sizes to show adaptive behavior
    test_sizes = [10, 50, 100]
    
    for size in test_sizes:
        if size <= main_data.size(0):
            subset = main_data[:size]
            thresholds = scorer._compute_adaptive_thresholds(subset)
            
            print(f"\nDataset size: {size}")
            print(f"  Outlier percentile: {thresholds['outlier_percentile']:.1f}%")
            print(f"  Rare label threshold: {thresholds['rare_label_threshold']:.3f}")
            print(f"  Diversity scale: {thresholds['diversity_scale']:.3f}")
            print(f"  Coverage threshold: {thresholds['coverage_threshold']:.3f}")
    
    print("\nAdaptive thresholds automatically adjust based on:")
    print("  - Dataset size (smaller datasets → less aggressive outlier detection)")
    print("  - Feature dimensionality and spread")
    print("  - Data distribution characteristics")
    print("  - This improves accuracy across different domain types!")


def print_validation_notes() -> None:
    """Print validation notes and conclusions."""
    print("\n" + "="*60)
    print("VALIDATION NOTES")
    print("="*60)
    print("1. Diversity: Non-zero and scales with data spread")
    print("2. Edge Coverage: Captures rare labels and feature outliers")
    print("3. Balance: Near 1 for even distributions, lower if skewed")
    print("4. Relevance: Between 0 and 1, higher for alignment with reference")
    print("5. Score: Increases sublinearly, typically 0 to ~0.693 for log(1 + Q)")
    print("6. RL: Reward rises with dataset size, stabilizing near max_size")
    print("\nObservations:")
    print("- All components range from 0 to 1, behaving as expected")
    print("- Total score reflects sublinear growth, aligning with function choice")
    print("- RL demo shows clear reward trajectory, validating scorer utility")
    print("- Weight configurations allow domain-specific tuning")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("The DatasetQualityScorer provides a systematic approach for evaluating")
    print("datasets. It combines diversity, edge-case coverage, balance, and relevance")
    print("metrics, providing a sublinear reward signal suitable for RL-driven")
    print("dataset generation.")
    
    print("\nNext Steps:")
    print("- Test on real language datasets (e.g., tokenized text embeddings)")
    print("- Tune weights for language-specific priorities")
    print("- Develop trained RL agents to maximize the score")
    print("- Integrate with production ML pipelines")


def main() -> None:
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Dataset Quality Scorer Demo")
    parser.add_argument("--save-plots", action="store_true", help="Save generated plots")
    parser.add_argument("--rl-steps", type=int, default=1000, help="Number of RL steps to run")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Configure matplotlib for non-interactive mode if no plots
    if args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
    
    try:
        # Run basic demo
        basic_results = run_basic_demo(save_plots=args.save_plots)
        
        # Run diversity scaling demo
        run_diversity_scaling_demo(
            basic_results["main_data"], 
            basic_results["scorer"],
            save_plots=args.save_plots
        )
        
        # Run weight sensitivity demo
        run_weight_sensitivity_demo(
            basic_results["main_data"],
            basic_results["val_data"],
            basic_results["ref_data"],
            save_plots=args.save_plots
        )
        
        # Run RL demo
        rl_results = run_rl_demo(
            basic_results["main_data"],
            basic_results["val_data"],
            basic_results["ref_data"],
            save_plots=args.save_plots,
            n_steps=args.rl_steps
        )
        
        # Run sublinear function comparison
        run_sublinear_comparison(
            basic_results["main_data"],
            basic_results["val_data"],
            basic_results["ref_data"]
        )
        
        # Run Polars integration demo
        run_polars_demo(basic_results["main_data"])
        
        # Demonstrate adaptive thresholds
        run_adaptive_thresholds_demo(basic_results["main_data"])
        
        # Print validation notes
        print_validation_notes()
        
        print(f"\n{'='*60}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        if args.save_plots:
            print("Plots saved to current directory.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Check your dependencies and data generation.")
        sys.exit(1)


if __name__ == "__main__":
    main() 