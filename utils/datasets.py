import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter

from abstract_interpretation.utils import load_npz
from utils import models


def get_dataset(folder, dataset):
    A, X, z = load_npz(f"{folder}/{dataset}.npz")
    A = A + A.T
    A[A > 1] = 1
    A.setdiag(0)

    edge_index, _ = torch_geometric.utils.from_scipy_sparse_matrix(A)

    X = (X > 0).astype("float32")
    X = torch.tensor(X.todense())
    z = z.astype("int64")
    z = torch.tensor(z)
    N, D = X.shape

    return A, X, z, N, D, edge_index


def cal_node_degree(edge_index, num_nodes=None):
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    _, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce="sum")
    return deg


def gcn_normalize(edge_index, deg=None, num_nodes=None):
    if deg is not None:
        num_nodes = deg.shape[0]

    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]

    if deg is None:
        deg = cal_node_degree(edge_index, num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

    edge_attr = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_attr


def prepare_model_dataset(dataset: str, robust_gcn: bool):
    (A, X, z, N, D, edge_index) = get_dataset(
        "./robust-gcn-structure/datasets/", dataset
    )

    gcn = models.get_model(
        "robust-gcn-structure/pretrained_weights",
        dataset,
        robust_gcn,
        normalize=False,
    )

    edge_index, edge_attrs = gcn_norm(edge_index)
    node_id = torch.LongTensor(range(X.shape[0]))
    labels = gcn(X, edge_index, edge_attrs).argmax(dim=1)

    data = Data(
        X,
        edge_index,
        edge_attrs,
        z,
        labels=labels,
        node_id=node_id,
    )

    return gcn, data
