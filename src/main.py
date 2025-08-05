import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import make_blobs

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.kmeans import KMeans
from src.logger import ClusteringLogger
from src.visualization import ClusteringVisualizer





def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup device (CPU/GPU) based on configuration"""
    device_config = config['training']['device']
    
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    return device


def generate_synthetic_data(config):
    """Generate synthetic data for clustering"""
    data_config = config['data']
    
    # Generate synthetic data
    X, y_true = make_blobs(
        n_samples=data_config['n_samples'],
        n_features=data_config['n_features'],
        centers=data_config['n_clusters'],
        cluster_std=data_config['noise'],
        random_state=data_config['random_state']
    )
    
    # Convert to torch tensor
    X = torch.tensor(X, dtype=torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.long)
    
    return X, y_true


def load_data(config):
    """Load data from file or generate synthetic data"""
    data_config = config['data']
    
    if data_config['data_path'] and os.path.exists(data_config['data_path']):
        data = torch.load(data_config['data_path'])
        X = data['X']
        y_true = data.get('y_true', None)
    else:
        X, y_true = generate_synthetic_data(config)
    
    return X, y_true


def setup_distributed_training(config):
    """Setup distributed training if enabled"""
    if not config['training']['distributed']:
        return False
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logging.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
    return True


def compute_clustering_metrics(X, y_pred, y_true=None):
    """Compute clustering evaluation metrics"""
    metrics = {}
    
    # Silhouette score (higher is better)
    try:
        silhouette_avg = silhouette_score(X.cpu().numpy(), y_pred.cpu().numpy())
        metrics['silhouette_score'] = silhouette_avg
    except Exception as e:
        logging.warning(f"Could not compute silhouette score: {e}")
        metrics['silhouette_score'] = None
    
    # Calinski-Harabasz score (higher is better)
    try:
        calinski_score = calinski_harabasz_score(X.cpu().numpy(), y_pred.cpu().numpy())
        metrics['calinski_harabasz_score'] = calinski_score
    except Exception as e:
        logging.warning(f"Could not compute Calinski-Harabasz score: {e}")
        metrics['calinski_harabasz_score'] = None
    
    # Cluster sizes
    unique, counts = torch.unique(y_pred, return_counts=True)
    cluster_sizes = dict(zip(unique.cpu().numpy().astype(int), counts.cpu().numpy().astype(int)))
    metrics['cluster_sizes'] = cluster_sizes
    metrics['cluster_size_std'] = torch.std(counts.float()).item()
    
    # If true labels are available, compute additional metrics
    if y_true is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        nmi = normalized_mutual_info_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        
        metrics['adjusted_rand_score'] = ari
        metrics['normalized_mutual_info_score'] = nmi
    
    return metrics





def save_results(model, metrics, config, logger):
    """Save model and metrics"""
    # Save model
    if config['output']['save_model']:
        os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
        model.save(config['output']['model_path'])
        logger.save_model_info(config['output']['model_path'])
    
    # Save metrics
    if config['output']['compute_metrics']:
        logger.save_metrics(metrics)


def main():
    """Main function for K-means clustering"""
    parser = argparse.ArgumentParser(description='K-means Clustering')
    parser.add_argument('--config', type=str, default='config/kmeans_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device setting (cpu/cuda/auto)')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Override number of clusters')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config['training']['device'] = args.device
    if args.n_clusters:
        config['model']['n_clusters'] = args.n_clusters
        config['data']['n_clusters'] = args.n_clusters
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Setup logging and visualization
    logger = ClusteringLogger(config)
    visualizer = ClusteringVisualizer(config)
    
    # Log experiment start
    config_override = {}
    if args.device:
        config_override['device'] = args.device
    if args.n_clusters:
        config_override['n_clusters'] = args.n_clusters
    if args.batch_size:
        config_override['batch_size'] = args.batch_size
    
    logger.log_experiment_start(config_override)
    
    # Setup device
    device = setup_device(config)
    
    try:
        # Load or generate data
        X, y_true = load_data(config)
        
        # Initialize K-means model
        model = KMeans(
            n_clusters=config['model']['n_clusters'],
            balanced=config['model']['balanced'],
            iter_limit=config['model']['iter_limit'],
            device=device
        )
        
        model_info = f"K-means with {config['model']['n_clusters']} clusters, distance={config['model']['distance']}, balanced={config['model']['balanced']}"
        logger.log_training_start(model_info)
        
        # Train the model
        start_time = datetime.now()
        cluster_centers = model.fit(
            X,
            distance=config['model']['distance'],
            tol=float(config['model']['tolerance']),
            tqdm_flag=config['training']['verbose'],
            batch_size=config['training']['batch_size']
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.log_training_complete(100, training_time, 0.001)  # Simplified
        
        # Predict cluster assignments
        prediction_start = datetime.now()
        y_pred = model.kmeans_predict(
            X,
            distance=config['model']['distance'],
            batch_size=config['training']['batch_size']
        )
        prediction_time = (datetime.now() - prediction_start).total_seconds()
        logger.log_prediction_complete(prediction_time)
        
        # Compute metrics
        if config['output']['compute_metrics']:
            metrics = compute_clustering_metrics(X, y_pred, y_true)
            logger.log_metrics(metrics)
            logger.log_cluster_distribution(metrics.get('cluster_sizes', {}))
        
        # Visualize results
        if config['output']['plot_results']:
            visualizer.visualize_all(X, y_pred, cluster_centers, metrics)
            logger.save_visualization_info(config['output']['plot_path'])
        
        # Save results
        save_results(model, metrics, config, logger)
        
        # Log experiment summary
        logger.log_experiment_summary()
        logger.log_experiment_complete()
        
    except Exception as e:
        logger.log_error(f"Error during clustering: {e}", e)
        raise
    
    finally:
        # Cleanup
        logger.cleanup()


if __name__ == "__main__":
    main()
