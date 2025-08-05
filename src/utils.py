import torch

def pairwise_distance(data1, data2, device=torch.device('cpu')):
    """Calculate pairwise Euclidean distance using broadcasting"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis

def pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    """Optimized pairwise Euclidean distance calculation"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)
    
    d = torch.sum(data1**2, dim=1, keepdim=True) + \
            torch.sum(data2**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', data1, data2.T)

    return d

def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    """Calculate pairwise cosine distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

def auction_lap(job_and_worker_to_score, return_token_to_worker=True):
    """
    Solving the balanced linear assignment problem with auction algorithm.
    Arguments:
        - job_and_worker_to_score -> N x M euclidean distances between N data points and M cluster centers
    Returns:
        - assignment -> balanced assignment between jobs and workers
    """
    eps = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50
    eps.clamp_min_(1e-04)
    
    # Remove redundant assertion
    if torch.isnan(job_and_worker_to_score).any():
        raise Exception("NaN distance")
        
    worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous()
    num_workers, num_jobs = worker_and_job_to_score.size()
    jobs_per_worker = num_jobs // num_workers
    value = torch.clone(worker_and_job_to_score)
    bids = torch.zeros((num_workers, num_jobs),
                        dtype=worker_and_job_to_score.dtype,
                        device=worker_and_job_to_score.device,
                        requires_grad=False)
    counter = 0
    index = None
    cost = torch.zeros((1,num_jobs,),
                        dtype=worker_and_job_to_score.dtype,
                        device=worker_and_job_to_score.device,
                        requires_grad=False)
    while True:
        top_values, top_index = value.topk(jobs_per_worker + 1, dim=1)
        # Each worker bids the difference in value between that job and the k+1th job
        bid_increments = top_values[:,:-1] - top_values[:,-1:]  + eps
        assert bid_increments.size() == (num_workers, jobs_per_worker)
        bids.zero_()
        bids.scatter_(dim=1, index=top_index[:,:-1], src=bid_increments)

        if counter < 100 and index is not None:
            # If we were successful on the last round, put in a minimal bid to retain
            # the job only if noone else bids. After N iterations, keep it anyway.
            bids.view(-1)[index] = eps
            
        if counter > 1000:
            bids.view(-1)[jobs_without_bidder] = eps
            
        # Find jobs that was a top choice for some worker
        jobs_with_bidder = (bids > 0).any(0).nonzero(as_tuple=False).squeeze(1)
        jobs_without_bidder = (bids == 0).all(0).nonzero(as_tuple=False).squeeze(1)

        # Find the highest bidding worker per job
        high_bids, high_bidders = bids[:, jobs_with_bidder].max(dim=0)
        if high_bidders.size(0) == num_jobs:
            # All jobs were bid for
            break
        
        # Make popular items more expensive
        cost[:, jobs_with_bidder] += high_bids
        value = worker_and_job_to_score - cost

        # Hack to make sure that this item will be in the winning worker's top-k next time
        index = (high_bidders * num_jobs) + jobs_with_bidder
        value.view(-1)[index] = worker_and_job_to_score.view(-1)[index]
        counter += 1
    
    if return_token_to_worker:
        return high_bidders
    _, sorting = torch.sort(high_bidders)
    assignment = jobs_with_bidder[sorting]
    assert len(assignment.unique()) == num_jobs

    return assignment.view(-1)