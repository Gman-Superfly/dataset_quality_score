#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL Environment for Dataset Generation

Authors: Oscar Goldman  
Date: March 1, 2025
Purpose: Gymnasium-compatible RL environment for dataset generation using DatasetQualityScorer

Description:
This module provides a reinforcement learning environment where agents can select
data points from a pool to construct optimal datasets. The environment uses the
DatasetQualityScorer as a reward signal and enforces unique selection constraints.
"""

__all__ = ['DatasetGenerationEnv', 'DatasetGenerationError']

from typing import Optional, Tuple, Dict, Any, Set
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dataset_quality_scorer import DatasetQualityScorer, DatasetQualityScorerError


class DatasetGenerationError(Exception):
    """Base exception for DatasetGeneration environment errors."""
    pass


class DatasetGenerationEnv(gym.Env):
    """
    RL environment for dataset generation with enhanced observation space.
    
    The agent selects unique data points from a pool to maximize dataset quality
    as measured by DatasetQualityScorer. The environment provides statistical
    observations about the current dataset state.
    """

    def __init__(
        self,
        data_pool: torch.Tensor,
        val_data: torch.Tensor,
        ref_data: torch.Tensor,
        max_size: int = 50,
        scorer_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        scorer_epsilon: float = 0.1,
        scorer_sublinear_func: str = "log",
    ) -> None:
        """
        Initialize environment with data pool and parameters.
        
        Args:
            data_pool: Pool of available data points to select from
            val_data: Validation data for edge case analysis
            ref_data: Reference data for relevance computation
            max_size: Maximum dataset size before episode termination
            scorer_weights: Weights for DatasetQualityScorer components
            scorer_epsilon: Sampling tolerance for scorer
            scorer_sublinear_func: Sublinear function type ('log' or 'sqrt')
            
        Raises:
            DatasetGenerationError: If initialization parameters are invalid
        """
        super().__init__()
        
        # Validate inputs
        assert isinstance(data_pool, torch.Tensor), f"Expected torch.Tensor for data_pool, got {type(data_pool)}"
        assert isinstance(val_data, torch.Tensor), f"Expected torch.Tensor for val_data, got {type(val_data)}"
        assert isinstance(ref_data, torch.Tensor), f"Expected torch.Tensor for ref_data, got {type(ref_data)}"
        assert data_pool.dim() == 2, f"data_pool must be 2D, got {data_pool.dim()}D"
        assert val_data.dim() == 2, f"val_data must be 2D, got {val_data.dim()}D"
        assert ref_data.dim() == 2, f"ref_data must be 2D, got {ref_data.dim()}D"
        assert data_pool.size(0) > 0, "data_pool must have at least one sample"
        assert max_size > 0, f"max_size must be positive, got {max_size}"
        assert max_size <= data_pool.size(0), f"max_size ({max_size}) cannot exceed pool size ({data_pool.size(0)})"
        
        self.data_pool = data_pool
        self.val_data = val_data
        self.ref_data = ref_data
        self.max_size = max_size
        
        # Initialize scorer with validation
        try:
            self.scorer = DatasetQualityScorer(
                weights=scorer_weights,
                epsilon=scorer_epsilon,
                sublinear_func=scorer_sublinear_func
            )
        except Exception as e:
            raise DatasetGenerationError(f"Failed to initialize scorer: {e}")
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(data_pool))
        
        # Observation space: [mean_features, var_features] 
        self.observation_space = spaces.Box(
            low=-float("inf"), 
            high=float("inf"), 
            shape=(2 * data_pool.size(1),),
            dtype=np.float32
        )
        
        # State tracking
        self.selected_indices: Set[int] = set()
        self.episode_step: int = 0
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.selected_indices = set()
        self.episode_step = 0
        
        initial_obs = self._get_observation()
        info = {
            "dataset_size": 0,
            "score": 0.0,
            "episode_step": self.episode_step
        }
        
        return initial_obs, info

    def step(
        self, 
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of RL, selecting data points.
        
        Args:
            action: Index of data point to select from pool
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            
        Raises:
            DatasetGenerationError: If action is invalid
        """
        assert isinstance(action, (int, np.integer)), f"Action must be integer, got {type(action)}"
        assert 0 <= action < self.action_space.n, f"Action {action} out of bounds [0, {self.action_space.n})"
        
        self.episode_step += 1
        
        # Penalize duplicate actions to encourage exploration
        if action in self.selected_indices:
            info = {
                "duplicate": True,
                "dataset_size": len(self.selected_indices),
                "score": 0.0,
                "episode_step": self.episode_step
            }
            return self._get_observation(), -0.1, False, False, info
        
        # Add new selection
        self.selected_indices.add(action)
        
        # Compute reward using scorer
        try:
            selected_data = self.data_pool[list(self.selected_indices)]
            score = self.scorer.compute_score(selected_data, self.val_data, self.ref_data)
            reward = score
        except Exception as e:
            # Fallback reward if scoring fails
            reward = 0.0
            score = 0.0
            
        # Check termination conditions
        terminated = len(self.selected_indices) >= self.max_size
        truncated = False  # Could add time limits here if needed
        
        # Generate observation and info
        obs = self._get_observation(selected_data if len(self.selected_indices) > 0 else None)
        info = {
            "dataset_size": len(self.selected_indices),
            "score": score,
            "episode_step": self.episode_step,
            "reward": reward
        }
        
        assert reward is not None, "Reward must not be None"
        assert isinstance(obs, np.ndarray), f"Observation must be numpy array, got {type(obs)}"
        assert obs.shape == self.observation_space.shape, f"Observation shape mismatch: {obs.shape} vs {self.observation_space.shape}"
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self, selected_data: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generate observation based on current dataset statistics.
        
        Args:
            selected_data: Currently selected data points (optional)
            
        Returns:
            Observation array with dataset statistics
            
        Raises:
            DatasetGenerationError: If observation generation fails
        """
        # Handle cases with insufficient data to avoid variance errors
        if selected_data is None or selected_data.size(0) <= 1:
            return np.zeros(2 * self.data_pool.size(1), dtype=np.float32)
        
        try:
            mean = selected_data.mean(dim=0).numpy().astype(np.float32)
            var = selected_data.var(dim=0).numpy().astype(np.float32)
            
            # Handle potential NaN/inf values
            mean = np.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
            var = np.nan_to_num(var, nan=0.0, posinf=1e6, neginf=0.0)
            
            obs = np.concatenate([mean, var])
            
            assert obs.shape == (2 * self.data_pool.size(1),), f"Observation shape mismatch: {obs.shape}"
            assert np.all(np.isfinite(obs)), "Observation contains non-finite values"
            
            return obs
            
        except Exception as e:
            raise DatasetGenerationError(f"Failed to generate observation: {e}")

    def get_selected_dataset(self) -> Optional[torch.Tensor]:
        """
        Get the currently selected dataset.
        
        Returns:
            Selected data points as tensor, or None if no selections made
        """
        if len(self.selected_indices) == 0:
            return None
            
        return self.data_pool[list(self.selected_indices)]

    def get_current_score(self) -> float:
        """
        Get the current dataset quality score.
        
        Returns:
            Current quality score, or 0.0 if no data selected
        """
        if len(self.selected_indices) == 0:
            return 0.0
            
        try:
            selected_data = self.data_pool[list(self.selected_indices)]
            return self.scorer.compute_score(selected_data, self.val_data, self.ref_data)
        except Exception:
            return 0.0

    def get_detailed_score(self) -> Optional[Dict[str, float]]:
        """
        Get detailed breakdown of current score components.
        
        Returns:
            Dictionary with score components, or None if no data selected
        """
        if len(self.selected_indices) == 0:
            return None
            
        try:
            selected_data = self.data_pool[list(self.selected_indices)]
            components = self.scorer.compute_detailed_score(selected_data, self.val_data, self.ref_data)
            return {
                "diversity": components.diversity,
                "edge_coverage": components.edge_coverage,
                "balance": components.balance,
                "relevance": components.relevance,
                "total_score": components.total_score
            }
        except Exception:
            return None

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current environment state.
        
        Args:
            mode: Rendering mode ('human' for text output)
            
        Returns:
            String representation if mode is not 'human'
        """
        selected_count = len(self.selected_indices)
        current_score = self.get_current_score()
        
        render_str = (
            f"Dataset Generation Environment\n"
            f"Selected: {selected_count}/{self.max_size} data points\n"
            f"Current Score: {current_score:.4f}\n"
            f"Episode Step: {self.episode_step}\n"
            f"Pool Size: {self.data_pool.size(0)}\n"
        )
        
        if mode == "human":
            print(render_str)
            return None
        else:
            return render_str

    def close(self) -> None:
        """Clean up environment resources."""
        # Clear references to potentially large tensors
        self.selected_indices.clear()
        # Note: We don't delete the data tensors as they might be used elsewhere 