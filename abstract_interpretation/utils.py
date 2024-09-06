from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch

from robust_gcn.robust_gcn.robust_gcn import RobustGCNModel


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith(".npz"):
        file_name += ".npz"
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        if "attr_data" in loader:
            attr_matrix = sp.csr_matrix(
                (
                    loader["attr_data"],
                    loader["attr_indices"],
                    loader["attr_indptr"],
                ),
                shape=loader["attr_shape"],
            )
        else:
            attr_matrix = None

        labels = loader.get("labels")

    return adj_matrix, attr_matrix, labels


def safe_topk(
    tensor: torch.Tensor, k: int, dim: int, largest: bool, size=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if size is None or size < 0:
        size = tensor.shape[dim]
    if k < size:
        return torch.topk(tensor, k, dim, largest=largest)
    else:
        return torch.topk(tensor, size, dim, largest=largest)


def sparse_collect(src: torch.Tensor, index: torch.Tensor, num_nodes: int):
    output_dim = src.shape[-1]
    collect = [
        src[torch.where(index == i)].reshape(-1, output_dim)
        for i in range(num_nodes)
    ]
    return collect
