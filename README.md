# K-means Clustering Implementation

A high-performance K-means clustering implementation using PyTorch with support for GPU acceleration, distributed training, and comprehensive visualization.

## Features

- **High Performance**: GPU-accelerated clustering with batch processing
- **Multiple Distance Metrics**: Euclidean and cosine distance support
- **Balanced Clustering**: Optional balanced clustering using auction algorithm
- **Comprehensive Logging**: Detailed experiment logging and metrics
- **Rich Visualizations**: 2D plots, cluster distributions, and high-dimensional data visualization
- **Easy Configuration**: YAML-based configuration system

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Default Configuration

```bash
./run_kmeans.sh
```

### 3. Run with Custom Parameters

```bash
# Run on GPU with 5 clusters
./run_kmeans.sh --device cuda --clusters 5

# Run with custom batch size
./run_kmeans.sh --batch-size 2048

# Use custom config file
./run_kmeans.sh config/my_config.yaml --clusters 10
```

## Configuration

Edit `config/kmeans_config.yaml` to customize:

- **Data**: Sample size, features, clusters
- **Model**: Distance metric, balanced clustering, iterations
- **Training**: Device, batch size, distributed training
- **Output**: Visualization, metrics, logging

## Output Structure

```
outputs/
├── plots/           # Visualization plots
├── kmeans_model.pkl # Trained model
├── metrics.json     # Clustering metrics
└── kmeans.log       # Experiment logs
```

## Examples

### Basic Usage
```bash
./run_kmeans.sh
```

### GPU Training
```bash
./run_kmeans.sh --device cuda --clusters 8
```

### Large Dataset
```bash
./run_kmeans.sh --batch-size 5000 --clusters 20
```

## Command Line Options

- `--device`: Set device (cpu/cuda/auto)
- `--clusters`: Number of clusters
- `--batch-size`: Batch size for processing
- `--help`: Show help message

## Files Structure

```
kmeans/
├── config/
│   └── kmeans_config.yaml    # Configuration file
├── src/
│   ├── kmeans.py             # K-means implementation
│   ├── main.py               # Main execution script
│   ├── logger.py             # Logging module
│   └── visualization.py      # Visualization module
├── run_kmeans.sh             # Shell runner script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
``` 