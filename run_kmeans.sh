#!/bin/bash

# K-means Clustering Runner Script
# Usage: ./run_kmeans.sh [config_file] [options]

CONFIG_FILE="config/kmeans_config.yaml"
ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            ARGS="$ARGS --device $2"
            shift 2
            ;;
        --clusters)
            ARGS="$ARGS --n_clusters $2"
            shift 2
            ;;
        --batch-size)
            ARGS="$ARGS --batch_size $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [config_file] [options]"
            echo "Options: --device, --clusters, --batch-size, --help"
            exit 0
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

# Create output directories
mkdir -p outputs/plots outputs/logs data

# Run K-means clustering
echo "Starting K-means clustering..."
python src/main.py --config "$CONFIG_FILE" $ARGS 