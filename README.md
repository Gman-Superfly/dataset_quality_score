# Dataset Quality Scorer

A configurable system for evaluating simple dataset quality using sublinear scoring functions, designed for reinforcement learning-driven dataset generation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The Dataset Quality Scorer provides a systematic approach to evaluating datasets for machine learning applications. It implements a multi-component scoring system with sublinear reward functions suitable for reinforcement learning agents tasked with dataset generation and curation.

### Core Features

- **Multi-Component Evaluation**: Analyzes datasets across four key dimensions
- **Sublinear Scoring**: Implements logarithmic and square root functions for diminishing returns
- **Configurable Weights**: Domain-specific presets and custom weight configurations  
- **Adaptive Thresholds**: Data-driven parameter adjustment for improved accuracy
- **RL Integration**: Gymnasium-compatible environment for dataset generation
- **Polars Support**: Efficient data processing with modern data library integration

## Architecture

The scorer computes a quality score **S = f(Q)** where:
- **Q = w₁D + w₂E + w₃B + w₄R** (weighted combination of components)
- **f(Q)** is a sublinear function (log or sqrt)

### Components

1. **Diversity (D)**: Measures data point spread using subsampled pairwise distances
2. **Edge-case Coverage (E)**: Evaluates representation of rare labels and feature outliers
3. **Balance (B)**: Assesses class/category distribution evenness via normalized entropy
4. **Relevance (R)**: Measures alignment with reference data using Wasserstein distance

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- Polars 0.20+ (replaces pandas for data handling)
- Matplotlib for visualization
- Gymnasium for RL environment

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd dataset_quality_score

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from dataset_quality_scorer import DatasetQualityScorer
from data_utils import generate_synthetic_data

# Generate sample data
data = generate_synthetic_data(100, 5, n_classes=2)
val_data = generate_synthetic_data(20, 5, n_classes=2)
ref_data = torch.randn(50, 5)

# Initialize scorer with preset weights
scorer = DatasetQualityScorer(
    weights="balanced",  # or custom tuple (0.25, 0.25, 0.25, 0.25)
    sublinear_func="log"  # or "sqrt"
)

# Compute score
score = scorer.compute_score(data, val_data, ref_data)
print(f"Dataset Quality Score: {score:.3f}")

# Get detailed breakdown
components = scorer.compute_detailed_score(data, val_data, ref_data)
print(f"Diversity: {components.diversity:.3f}")
print(f"Edge Coverage: {components.edge_coverage:.3f}")
print(f"Balance: {components.balance:.3f}")
print(f"Relevance: {components.relevance:.3f}")
```

### Weight Presets

The system includes several predefined weight configurations:

```python
from dataset_quality_scorer import get_available_weight_presets

# View all available presets
presets = get_available_weight_presets()
print(presets)

# Use domain-specific presets
diversity_scorer = DatasetQualityScorer("diversity_focused")    # (0.5, 0.2, 0.15, 0.15)
balance_scorer = DatasetQualityScorer("balance_critical")       # (0.15, 0.15, 0.6, 0.1)
production_scorer = DatasetQualityScorer("production_safe")     # (0.2, 0.3, 0.3, 0.2)
```

Available presets:
- `balanced`: Equal weighting across all components
- `diversity_focused`: Emphasizes data spread for representation learning
- `balance_critical`: Prioritizes class balance for imbalanced domains
- `relevance_heavy`: Focuses on domain alignment for transfer learning
- `edge_case_priority`: Emphasizes rare cases for robustness testing
- `production_safe`: Conservative weights for production systems
- `research_exploration`: Optimized for research dataset creation

### RL Environment

```python
from rl_environment import DatasetGenerationEnv

# Create environment
env = DatasetGenerationEnv(
    data_pool=pool_data,    # Large pool of candidate data
    val_data=val_data,      # Validation set
    ref_data=ref_data,      # Reference distribution
    max_size=1000           # Maximum dataset size
)

# Reset environment
obs, info = env.reset()

# Run agent
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    
    if done:
        break

print(f"Final score: {env.get_current_score():.3f}")
```

### Polars Integration

```python
from data_utils import load_data_from_polars, torch_to_polars

# Load data from various formats
data_tensor = load_data_from_polars(
    "dataset.csv",
    feature_columns=["feature_1", "feature_2", "feature_3"],
    label_column="label"
)

# Convert PyTorch tensors to Polars DataFrames
df = torch_to_polars(data_tensor, column_names=["feat1", "feat2", "feat3", "label"])

# Use Polars for data operations
filtered_df = df.filter(pl.col("label") == 1)
stats = df.select([pl.col("feat1").mean(), pl.col("feat2").std()])
```

## Configuration

### Adaptive Thresholds

The system automatically adjusts thresholds based on dataset characteristics:

```python
# Thresholds adapt to:
# - Dataset size (smaller datasets use less aggressive outlier detection)
# - Feature dimensionality and spread
# - Data distribution characteristics

scorer = DatasetQualityScorer()
thresholds = scorer._compute_adaptive_thresholds(data)
print(f"Outlier percentile: {thresholds['outlier_percentile']:.1f}%")
print(f"Rare label threshold: {thresholds['rare_label_threshold']:.3f}")
```

### Demo Scripts

Run comprehensive demonstrations:

```bash
# Full demo with plots
python main_demo.py --save-plots

# Demo without plots (for servers)
python main_demo.py --no-plots

# Custom RL steps  
python main_demo.py --rl-steps 1000
```

**Note on Dataset Generation:**
- Each demo run generates **different synthetic datasets** using time-based random seeds
- This demonstrates the scorer's behavior across varied data distributions
- CSV outputs are saved with timestamps (`example_dataset_YYYYMMDD_HHMMSS.csv`) to preserve results from each run
- For reproducible results in research, you can modify the code to use fixed seeds

## Project Structure

```
dataset_quality_score/
├── dataset_quality_scorer.py    # Core scorer implementation
├── rl_environment.py           # RL environment for dataset generation
├── data_utils.py              # Data generation and utilities (with Polars support)
├── main_demo.py               # Demonstration script
├── requirements.txt           # Python dependencies
├── README.md                 # Documentation
├── LICENSE                   # MIT license
└── DatasetQualityScore_(sublinear).ipynb  # Original research notebook
```

## API Reference

### DatasetQualityScorer

Main class for computing dataset quality scores.

**Constructor:**
```python
DatasetQualityScorer(
    weights: Union[str, Tuple[float, float, float, float]] = "balanced",
    epsilon: float = 0.1,
    sublinear_func: Literal["log", "sqrt"] = "log"
)
```

**Methods:**
- `compute_score(data, val_data=None, ref_data=None) -> float`
- `compute_detailed_score(data, val_data=None, ref_data=None) -> ScoreComponents`

### DatasetGenerationEnv

Gymnasium-compatible RL environment for dataset generation.

**Methods:**
- `reset(seed=None, options=None) -> Tuple[np.ndarray, Dict]`
- `step(action) -> Tuple[np.ndarray, float, bool, bool, Dict]`
- `get_current_score() -> float`
- `get_detailed_score() -> Optional[Dict[str, float]]`

### Data Utilities

**Polars Integration:**
- `load_data_from_polars(file_path, feature_columns=None, label_column=None)`
- `polars_to_torch(df) -> torch.Tensor`
- `torch_to_polars(tensor, column_names=None) -> pl.DataFrame`

**Visualization:**
- `plot_dataset_2d(data, title="Dataset Visualization")`
- `plot_score_components(components, title="Score Components")`
- `plot_rl_progress(dataset_sizes, rewards, title="RL Progress")`

## Performance Characteristics

### Computational Complexity

- **Diversity**: O(log n) with intelligent subsampling
- **Edge Coverage**: O(n) for outlier detection
- **Balance**: O(n) for entropy calculation  
- **Relevance**: O(nm) for Wasserstein distance

### Typical Score Ranges

- **Individual Components**: [0, 1]
- **Combined Score Q**: [0, 1] 
- **Final Score S**:
  - Log function: [0, ~0.693]
  - Sqrt function: [0, 1]

## Use Cases

### Research Applications

- RL-driven dataset generation for training data selection
- Active learning prioritization for annotation
- Data augmentation quality assessment
- Transfer learning domain alignment evaluation

### Production Applications

- Automated dataset quality monitoring in ML pipelines
- Data acquisition strategy optimization
- Model performance prediction indicators
- A/B testing for dataset variants

## Implementation Notes

### Validation Approach

The implementation follows systematic validation principles:

1. **Input Validation**: Parameter checking with descriptive assertions
2. **Component Bounds**: Scores maintained within [0, 1] ranges
3. **Mathematical Consistency**: Sublinear properties verified
4. **Edge Cases**: Handling of edge cases (empty data, single samples)
5. **Error Recovery**: Graceful fallbacks for computation failures

### Numerical Stability Improvements

Recent enhancements include:

- Adaptive threshold computation based on dataset characteristics
- Improved diversity normalization using actual data ranges
- More stable entropy calculation avoiding log(0) issues
- Robust outlier detection with conservative percentiles
- Enhanced Wasserstein distance normalization

## Limitations

### Current Constraints

- Feature independence assumption in relevance computation
- Requires labeled data for balance and edge coverage components
- Performance scales with dataset size for some components
- Validation primarily on synthetic datasets

### Known Considerations

- Weight selection requires domain knowledge or empirical tuning
- Reference dataset quality directly impacts relevance scores
- Edge case definitions may not generalize across all domains
- Outlier detection sensitivity varies with data distribution

## Contributing

No just make your own this is a fun test example for my own use, if you really want to follow these guidelines

1. **Vvalidation** - Validate all inputs and assumptions
2. **Type annotations** - Use complete type hints
3. **Focused functions** - Maintain single responsibility principle
4. **Error handling** - Implement graceful failure modes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

LMAO

## Contact

- **Author**: Oscar Goldman
- **Issues**: Please use the GitHub issue tracker for bug reports and feature requests

---

*This system provides a systematic approach to simple dataset quality evaluation. Users should validate the scoring behavior on their specific domains and adjust weights accordingly.*
