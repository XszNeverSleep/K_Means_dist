import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ClusteringLogger:
    """Logging class for K-means clustering experiments"""
    
    def __init__(self, config):
        self.config = config
        self.log_file = config['logging']['log_file']
        self.metrics_path = config['output']['metrics_path']
        self.save_logs = config['logging']['save_logs']
        
        # Create directories
        if self.save_logs:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Store experiment info
        self.experiment_info = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': {},
            'training_history': []
        }
    
    def _setup_logger(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['logging']['log_level'].upper())
        
        # Create logger
        logger = logging.getLogger('KMeansClustering')
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler (detailed)
        if self.save_logs:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        # Console handler (simple)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_experiment_start(self, config_override=None):
        """Log experiment start information"""
        self.logger.info("=" * 60)
        self.logger.info("K-MEANS CLUSTERING EXPERIMENT STARTED")
        self.logger.info("=" * 60)
        
        # Log configuration
        self.logger.info("Configuration:")
        if config_override:
            self.logger.info(f"  Config overrides: {config_override}")
        
        # Log key parameters
        model_config = self.config['model']
        data_config = self.config['data']
        training_config = self.config['training']
        
        self.logger.info(f"  Number of clusters: {model_config['n_clusters']}")
        self.logger.info(f"  Distance metric: {model_config['distance']}")
        self.logger.info(f"  Balanced clustering: {model_config['balanced']}")
        self.logger.info(f"  Data samples: {data_config['n_samples']}")
        self.logger.info(f"  Data features: {data_config['n_features']}")
        self.logger.info(f"  Device: {training_config['device']}")
        self.logger.info(f"  Batch size: {training_config['batch_size']}")
        self.logger.info(f"  Distributed: {training_config['distributed']}")
        
        # Store experiment info
        self.experiment_info['config_override'] = config_override
    
    def log_training_start(self, model_info):
        """Log training start information"""
        self.logger.info("-" * 40)
        self.logger.info("TRAINING STARTED")
        self.logger.info("-" * 40)
        self.logger.info(f"Model initialized: {model_info}")
        self.start_time = datetime.now()
    
    def log_training_progress(self, iteration, center_shift, elapsed_time=None):
        """Log training progress"""
        if iteration % 10 == 0 or iteration <= 5:  # Log every 10 iterations or first 5
            time_str = f" ({elapsed_time:.2f}s)" if elapsed_time else ""
            self.logger.info(f"Iteration {iteration:3d}: center_shift = {center_shift:.6f}{time_str}")
    
    def log_training_complete(self, final_iteration, total_time, final_shift):
        """Log training completion"""
        self.logger.info("-" * 40)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("-" * 40)
        self.logger.info(f"Total iterations: {final_iteration}")
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info(f"Final center shift: {final_shift:.6f}")
        
        # Store training info
        self.experiment_info['training_summary'] = {
            'total_iterations': final_iteration,
            'total_time': total_time,
            'final_center_shift': final_shift,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
    
    def log_prediction_complete(self, prediction_time):
        """Log prediction completion"""
        self.logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
    
    def log_metrics(self, metrics):
        """Log clustering metrics"""
        self.logger.info("-" * 40)
        self.logger.info("CLUSTERING METRICS")
        self.logger.info("-" * 40)
        
        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, dict):
                    self.logger.info(f"{key}:")
                    for sub_key, sub_value in value.items():
                        self.logger.info(f"  {sub_key}: {sub_value}")
                else:
                    self.logger.info(f"{key}: {value}")
        
        # Store metrics
        self.experiment_info['metrics'] = metrics
    
    def log_cluster_distribution(self, cluster_sizes):
        """Log cluster size distribution"""
        if not cluster_sizes:
            return
        
        self.logger.info("-" * 40)
        self.logger.info("CLUSTER SIZE DISTRIBUTION")
        self.logger.info("-" * 40)
        
        sizes = list(cluster_sizes.values())
        self.logger.info(f"Total clusters: {len(cluster_sizes)}")
        self.logger.info(f"Min cluster size: {min(sizes)}")
        self.logger.info(f"Max cluster size: {max(sizes)}")
        self.logger.info(f"Mean cluster size: {np.mean(sizes):.1f}")
        self.logger.info(f"Std cluster size: {np.std(sizes):.1f}")
        
        # Log individual cluster sizes
        for cluster_id, size in sorted(cluster_sizes.items()):
            self.logger.info(f"  Cluster {cluster_id}: {size} points")
    
    def log_experiment_complete(self):
        """Log experiment completion"""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
    
    def log_error(self, error_msg, exception=None):
        """Log error information"""
        self.logger.error("-" * 40)
        self.logger.error("ERROR OCCURRED")
        self.logger.error("-" * 40)
        self.logger.error(f"Error message: {error_msg}")
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
            self.logger.error(f"Exception type: {type(exception).__name__}")
    
    def log_warning(self, warning_msg):
        """Log warning information"""
        self.logger.warning(f"WARNING: {warning_msg}")
    
    def log_info(self, info_msg):
        """Log general information"""
        self.logger.info(info_msg)
    
    def log_debug(self, debug_msg):
        """Log debug information"""
        self.logger.debug(debug_msg)
    
    def save_metrics(self, metrics):
        """Save metrics to JSON file"""
        if not self.config['output']['compute_metrics']:
            return
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = self._make_serializable(metrics)
        
        # Add experiment info
        output_data = {
            'experiment_info': self.experiment_info,
            'metrics': metrics_serializable,
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            self.logger.info(f"Metrics saved to {self.metrics_path}")
        except TypeError as e:
            # Fallback: convert all keys to strings
            def convert_keys_to_str(obj):
                if isinstance(obj, dict):
                    return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_keys_to_str(item) for item in obj]
                else:
                    return obj
            
            output_data = convert_keys_to_str(output_data)
            with open(self.metrics_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            self.logger.info(f"Metrics saved to {self.metrics_path}")
    
    def save_model_info(self, model_path):
        """Log model saving information"""
        self.logger.info(f"Model saved to {model_path}")
    
    def save_visualization_info(self, plot_path):
        """Log visualization saving information"""
        self.logger.info(f"Visualizations saved to {plot_path}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {str(self._make_serializable(key)): self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    def get_experiment_summary(self):
        """Get a summary of the experiment"""
        summary = {
            'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': self.experiment_info['timestamp'],
            'config': self.config,
            'training_summary': self.experiment_info.get('training_summary', {}),
            'metrics': self.experiment_info.get('metrics', {})
        }
        return summary
    
    def log_experiment_summary(self):
        """Log a summary of the experiment"""
        summary = self.get_experiment_summary()
        
        self.logger.info("-" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("-" * 60)
        self.logger.info(f"Experiment ID: {summary['experiment_id']}")
        self.logger.info(f"Timestamp: {summary['timestamp']}")
        
        if 'training_summary' in summary:
            training = summary['training_summary']
            self.logger.info(f"Training iterations: {training.get('total_iterations', 'N/A')}")
            self.logger.info(f"Training time: {training.get('total_time', 'N/A'):.2f}s")
            self.logger.info(f"Final center shift: {training.get('final_center_shift', 'N/A'):.6f}")
        
        if 'metrics' in summary:
            metrics = summary['metrics']
            if metrics.get('silhouette_score'):
                self.logger.info(f"Silhouette score: {metrics['silhouette_score']:.3f}")
            if metrics.get('calinski_harabasz_score'):
                self.logger.info(f"Calinski-Harabasz score: {metrics['calinski_harabasz_score']:.3f}")
            if metrics.get('cluster_size_std'):
                self.logger.info(f"Cluster size std: {metrics['cluster_size_std']:.1f}")
    
    def cleanup(self):
        """Cleanup logging resources"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 