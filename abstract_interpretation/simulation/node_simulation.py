from utils import datasets, models
import torch
from tqdm import tqdm

dataset = 'pubmed'
LN = 100000
robust_gcn = False
global_p = 10
iterations = 20000
device = 'cuda:0'

def perturb(X, l, g, N, F):
    """
    Perturbs the feature vector of nodes in a graph.

    Parameters
    ----------
    X : torsor: feature matrix
    l : int: local perturbation
    g : int: global perturbation
    N : int: number of nodes
    F : int: number of features
    """

    # generate l features to perturb for each node
    nodes = (torch.arange(0, N) * LN).view(-1, 1).to(device)
    feat_perturb = (nodes + torch.randint(0, F, (N, l)).to(device)).flatten()
    selection = torch.randperm(feat_perturb.shape[0])[:g]
    feat_perturb = feat_perturb[selection]
    x, y = (feat_perturb / LN).round().long(), torch.remainder(feat_perturb, LN).long()
    X_new = X.clone().detach()
    X_new[x, y] = 1 - X_new[x, y]
    return X_new


def main():
    (A, X, z, N, D, edge_index) = datasets.get_dataset('./robust-gcn-structure/datasets/', dataset)
    gcn = models.get_model('robust-gcn-structure/pretrained_weights', dataset, robust_gcn)
    
    X = X.to(device)
    edge_index = edge_index.to(device)
    gcn = gcn.to(device)
    original_result = gcn(X, edge_index).argmax(dim=1)
    record = torch.zeros(N)
    changed = 0
    
    for p in range(1, global_p + 1):
        print(f'perturbation {p}')
        
        for i in tqdm(range(iterations)):
            i += 1
            X_p = perturb(X, p, p, N, X.shape[1])
            perturb_result = gcn(X_p, edge_index).argmax(dim=1)
            changed_nodes = (perturb_result != original_result).nonzero()
            record[changed_nodes] = 1
            new_changed = record.sum()
            if i % 10000 == 0:
                new_robustness = 1 - new_changed / N
                print(f'iteration {i}, current robustness: {new_robustness}')
        print(f'final robustness upper bound for perturbation {p}: {1 - record.sum() / N}')
        

if __name__ == '__main__':
    main()