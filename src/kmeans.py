import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
import pickle
from utils import auction_lap, pairwise_euclidean, pairwise_cosine

    
class KMeans(object):
    def __init__(self, n_clusters=None, cluster_centers=None, balanced=False, iter_limit=100, device=torch.device('cpu')):
        self.n_clusters = n_clusters
        self.cluster_centers = cluster_centers
        self.balanced = balanced
        self.device=device
        self.iter_limit = iter_limit

    @classmethod
    def load(cls, path_to_file):
        with open(path_to_file, 'rb') as f:
            saved = pickle.load(f)
        return cls(saved['n_clusters'], saved['cluster_centers'], torch.device('cpu'), saved['balanced'])
    
    def save(self, path_to_file):
        with open(path_to_file, 'wb+') as f :
            pickle.dump(self.__dict__, f)    

    def initialize(self, X):
        """
        initialize cluster centers
        :param X: (torch.tensor) matrix
        :param n_clusters: (int) number of clusters
        :return: (np.array) initial state
        """
        num_samples = len(X)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        initial_state = X[indices]
        return initial_state
    
    def fit_dist(
        self,
        X,
        distance='euclidean',
        tol=1e-4,
        tqdm_flag=True,
        online=False,
        iter_k=0,
        rank=0,
        batch_size=10240,
    ):
        print(f'running k-means on {self.device}..')
        
        if distance == 'euclidean':
            pairwise_distance_function = pairwise_euclidean
        elif distance == 'cosine':
            pairwise_distance_function = pairwise_cosine
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()

        # transfer to device
        X = X.to(self.device)

        # initialize
        if not online or (online and iter_k == 0):
            self.cluster_centers = torch.zeros(self.n_clusters, X.shape[-1], device=self.device)
            if dist.get_rank() == 0:
                self.cluster_centers = self.initialize(X).to(self.device)
                
        self.cluster_centers = self.cluster_centers.to(self.device)
        iteration = 0
        
        # Only create progress bar on rank 0
        if rank == 0:
            pbar = tqdm(total=self.iter_limit, desc='K-means iterations')
        
        while True:
            dist.broadcast(self.cluster_centers, src=0)
            choice_clusters = []
            
            # Add inner progress bar for batch processing (only on rank 0)
            total_batches = (X.shape[0] + batch_size - 1) // batch_size
            if rank == 0:
                batch_pbar = tqdm(total=total_batches, desc=f'Processing {iter_k} batches (iteration {iteration+1})', 
                                leave=False)
            
            for start_idx in range(0, X.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X.shape[0])
                X_batch = X[start_idx:end_idx]
                dis = pairwise_distance_function(X_batch, self.cluster_centers, device=self.device)
                choice_cluster = torch.argmin(dis, dim=1)
                choice_clusters.append(choice_cluster)
                if rank == 0:
                    batch_pbar.update(1)
            
            if rank == 0:
                batch_pbar.close()
                
            choice_cluster = torch.cat(choice_clusters, dim=0)
            
            # Optimize: reuse distance matrix for balanced clustering
            if self.balanced:
                # Reconstruct distance matrix from batch results to avoid recomputation
                distance_matrix = torch.cat([pairwise_distance_function(X[start_idx:min(start_idx+batch_size, X.shape[0])], 
                                                self.cluster_centers, device=self.device) 
                                        for start_idx in range(0, X.shape[0], batch_size)], dim=0)
                choice_cluster = auction_lap(-distance_matrix)

            torch.distributed.barrier()
            initial_state_pre = self.cluster_centers.clone()
            
            for index in range(self.n_clusters):
                selected = torch.nonzero(choice_cluster == index).squeeze()
                selected = torch.index_select(X, 0, selected)
                new_center = selected.mean(dim=0)
                dist.all_reduce(new_center, op=dist.ReduceOp.SUM)
                new_center /= dist.get_world_size()
                self.cluster_centers[index] = new_center

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.cluster_centers - initial_state_pre) ** 2, dim=1)
                ))
                
            iteration = iteration + 1
            
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({'center_shift': f'{center_shift.item():.6f}'})

            if center_shift < tol or iteration >= self.iter_limit:
                break
        
        if rank == 0:
            pbar.close()
        
        return self.cluster_centers.cpu()

    def fit(
        self,
        X,
        distance='euclidean',
        tol=1e-4,
        tqdm_flag=True,
        batch_size=10240,  # 增加 batch_size 参数
    ):

        print(f'Running k-means on {self.device}')

        if distance == 'euclidean':
            pairwise_distance_function = pairwise_euclidean
        elif distance == 'cosine':
            pairwise_distance_function = pairwise_cosine
        else:
            raise NotImplementedError

        # 转换为浮点数
        X = X.float()

        # 转移到设备
        X = X.to(self.device)

        # 初始化聚类中心
        self.cluster_centers = self.initialize(X).to(self.device)

        iteration = 0
        if tqdm_flag:
            tqdm_meter = tqdm(desc='[running kmeans]')
        while True:
            choice_clusters = []
            for start_idx in range(0, X.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X.shape[0])
                X_batch = X[start_idx:end_idx]
                dis = pairwise_distance_function(X_batch, self.cluster_centers, device=self.device)
                choice_cluster = torch.argmin(dis, dim=1)
                choice_clusters.append(choice_cluster)
            choice_cluster = torch.cat(choice_clusters, dim=0)

            initial_state_pre = self.cluster_centers.clone()
            for index in range(self.n_clusters):
                selected = torch.nonzero(choice_cluster == index).squeeze()
                if len(selected) > 0:
                    selected = torch.index_select(X, 0, selected)
                    new_center = selected.mean(dim=0)
                    self.cluster_centers[index] = new_center

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.cluster_centers - initial_state_pre) ** 2, dim=1)
                )
            )
            iteration += 1
            if tqdm_flag:
                tqdm_meter.update(1)
            if center_shift < tol or iteration >= self.iter_limit:
                break

        if tqdm_flag:
            tqdm_meter.close()
        return self.cluster_centers

    def kmeans_predict(
        self,
        X,
        distance='euclidean',
        batch_size=10240,  # Add batch_size parameter
    ):
        """
        predict using cluster centers
        :param X: (torch.tensor) matrix
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param batch_size: (int) batch size for processing large datasets
        :return: (torch.tensor) cluster ids
        """
        print(f'predicting on {self.device}..')

        if distance == 'euclidean':
            pairwise_distance_function = pairwise_euclidean
        elif distance == 'cosine':
            pairwise_distance_function = pairwise_cosine
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()

        # transfer to device
        self.cluster_centers = self.cluster_centers.to(self.device)
        X = X.to(self.device)

        # Process in batches with progress bar
        choice_clusters = []
        total_batches = (X.shape[0] + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc='Predicting clusters') as pbar:
            for start_idx in range(0, X.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X.shape[0])
                X_batch = X[start_idx:end_idx]
                dis = pairwise_distance_function(X_batch, self.cluster_centers)
                choice_cluster = torch.argmin(dis, dim=1)
                choice_clusters.append(choice_cluster)
                pbar.update(1)
        
        choice_cluster = torch.cat(choice_clusters, dim=0)
        return choice_cluster.cpu()