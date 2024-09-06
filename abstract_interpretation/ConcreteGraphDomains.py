import torch
from torch_geometric.typing import Adj


class BinaryNodeFeatureDomain:
    edge_index: Adj
    q: int
    x: torch.Tensor

    def __init__(self, edge_list: Adj, q: int, x: torch.Tensor, l:int = -1):
        """
        edge_list: (Adj) the list of edges of shape [2, E]
        q: global perturbation space on node features
        x: node features of shape [N, R]
        l: local perturbation space
        """
        self.edge_index = edge_list
        self.q = q
        self.x = x
        if l == -1:
            self.l = q
        else:
            self.l = l

