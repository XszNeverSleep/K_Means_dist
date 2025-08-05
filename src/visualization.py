import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE


class ClusteringVisualizer:
    """Simplified visualization class for K-means clustering results"""
    
    def __init__(self, config):
        self.config = config
        self.plot_path = Path(config['output']['plot_path'])
        self.plot_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_tsne(self, X, y_pred, cluster_centers):
        """Plot t-SNE visualization of clustering results"""
        if X.shape[1] <= 2:
            print("Data is 2D or less, skipping t-SNE plot")
            return
        
        print("Creating t-SNE visualization...")
        
        # Convert tensors to numpy if needed
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if torch.is_tensor(cluster_centers):
            cluster_centers = cluster_centers.cpu().numpy()
        
        # Combine data and centers for t-SNE
        combined_data = np.vstack([X, cluster_centers])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
        combined_reduced = tsne.fit_transform(combined_data)
        
        # Split back into data and centers
        X_reduced = combined_reduced[:len(X)]
        centers_reduced = combined_reduced[len(X):]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of data points
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='viridis', alpha=0.6, s=20)
        
        # Plot cluster centers
        plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
        
        plt.title('K-means Clustering Results (t-SNE)', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter, label='Cluster')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path / 'clustering_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE plot saved to {self.plot_path / 'clustering_tsne.png'}")
    
    def visualize_all(self, X, y_pred, cluster_centers, metrics):
        """Generate t-SNE visualization only"""
        print(f"Creating visualizations in {self.plot_path}")
        
        # Only create t-SNE plot
        self.plot_tsne(X, y_pred, cluster_centers)
        
        print(f"Visualization completed") 