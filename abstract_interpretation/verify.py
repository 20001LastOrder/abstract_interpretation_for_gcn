from abstract_interpretation.node_interval import NodeIntervalElement, NodeInterval, BinaryNodeFeatureDomain
from abstract_interpretation.node_deeppoly import NodeDeeppolyVerifier
from abstract_interpretation.ConcreteGraphDomains import BinaryNodeFeatureDomain
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch
from tqdm import tqdm

def pre_verification(gcn, edge_index, X):
    """
    Pre-processing for verification.
    Args:
        gcn: The GCN model.
        edge_index: edge index of the graph
        X: input node features
    """
    layers = list(gcn.children())
    # turn off normalize in GCNConv, manual normalization is provided
    for layer in layers:
        if isinstance(layer, GCNConv):
            layer.normalize = False
    
    # get normalized edge weights
    edge_index, edge_weights = gcn_norm(edge_index, num_nodes=X.shape[0])
    # get original predicted labels
    predicted_logits = gcn(X, edge_index, edge_weights)

    return layers, edge_index, edge_weights, predicted_logits

@torch.no_grad()
def verify_interval(gcn, X, edge_index, l, q, approx='topk', labels=None):
    """
    Verify the GCN model using the interval domain.
    Args:
        gcn: The GCN model, currently, three types of layers are supported (GCNConv, ReLU and Linear)
        X: input node features
        edge_index: edge index of the graph
        l: local perturbation budget
        q: global perturbation budget
        approx: approximation method, currently, 'topk' and 'max' are supported
        labels: the labels of the nodes, if None, the labels are predicted by the model
    """
    layers, edge_index, edge_weights, predicted_logits = pre_verification(gcn, edge_index, X)
    if labels is None:
        labels = predicted_logits.argmax(dim=1)

    concrete_domain = BinaryNodeFeatureDomain(edge_index, q, X, l)
    abstract_domain = NodeInterval()
    abstract_ele = abstract_domain.transform(concrete_domain, layers, approx=approx, edge_weights=edge_weights)
    node_robustness = []
    lb = abstract_ele.lb()
    ub = abstract_ele.ub()
    for node in range(X.shape[0]):
        label = labels[node]
        if lb[node][label] > max([value for i, value in enumerate(ub[node]) if i != label]):
            node_robustness.append(True)
        else:
            node_robustness.append(False)
    return np.array(node_robustness)

@torch.no_grad()
def verify_deeppoly(gcn, X, edge_index, l, q, approx='topk', hop_neighbors=2, labels=None,
                    batch_size=8, device='cpu', verbose=False):
    """
    Verify the GCN model using the Deeppoly domain.
    Args:
        gcn: The GCN model, currently, three types of layers are supported (GCNConv, ReLU and Linear)
        X: input node features
        edge_index: edge index of the graph
        l: local perturbation budget
        q: global perturbation budget
        approx: approximation method, currently, 'topk' and 'max' are supported
        hop_neighbors: the number of hop of neighbors to be considered
        labels: the labels of the nodes, if None, the labels are predicted by the model
        batch_size: batch size for verification
        device: device for verification
        verbose: whether to print the progress
    """
    layers, edge_index, edge_weights, predicted_logits = pre_verification(gcn, edge_index, X)

    concrete_domain = BinaryNodeFeatureDomain(edge_index, q, X, l)
    verifier = NodeDeeppolyVerifier(layers, concrete_domain, approx=approx, edge_weights=edge_weights)
    num_classes = predicted_logits.shape[1]
    if labels is None:
        labels = predicted_logits.argmax(dim=1)

    data = Data(X.cpu(), edge_index.cpu(), edge_weights.cpu(), labels.cpu(), node_id = torch.LongTensor(range(X.shape[0])))
    loader = NeighborLoader(
        data,
        num_neighbors=[-1 for _ in range(hop_neighbors)],
        batch_size=batch_size,
        directed=False
    )

    node_robustness = []
    node_non_robustness = []

    if verbose:
        loader = tqdm(loader)

    for batch in loader:
        batch = batch.to(device)
        labels = batch.y[:batch.batch_size]
        nodes = list(range(batch.batch_size))
        diff_lb, (coord_x, coord_y) = lb_deeppoly_batch(verifier, batch, num_classes, device=device)
        batch_robustness, batch_non_robustness = uncertify_node_deeppoly_batch(nodes, diff_lb, coord_x, 
                                                                               coord_y, batch, gcn)
        node_robustness.extend(batch_robustness)
        node_non_robustness.extend(batch_non_robustness)
    return np.array(node_robustness), np.array(node_non_robustness)

@torch.no_grad()
def uncertify_node_deeppoly_batch(nodes, lbs, coord_x, coord_y, batch, gcn):
    """
    Try to uncertify the nodes using the Deeppoly domain by checking the most influential perturbation
    Args:
        nodes: the nodes (N)
        lbs: the lower bounds (N x (C - 1))
        coord_x: the x coordinates of the perturbation (N x (C - 1) x p)
        coord_y: the y coordinates of the perturbation (N x (C - 1) x p)
        batch: the batch
        gcn: the GCN model
    """
    node_robustness = []
    node_non_robustness = []
    for i in range(len(nodes)):
        certified, uncertified = uncertify_node_deeppoly(nodes[i], lbs[i, :], coord_x[i, :, :], coord_y[i, :, :], batch, gcn)
        node_robustness.append(certified)
        node_non_robustness.append(uncertified)
    return node_robustness, node_non_robustness

@torch.no_grad()
def uncertify_node_deeppoly(node, lbs, coord_x, coord_y, batch, gcn):
    """
    Try to uncertify a single node in batch using the Deeppoly domain by checking the most influential perturbation
    Args:
        node: the node id (int)
        lbs: the lower bounds (C - 1)
        coord_x: the x coordinates of the perturbation ((C - 1) x p)
        coord_y: the y coordinates of the perturbation ((C - 1) x p)
        batch: the batch
        gcn: the GCN model
    Return:
        certified: whether the node is certified
        uncertified: whether the node is uncertified with a counter example
    """
    if (lbs > 0).all():
        return True, False
    original_label = batch.y[node]
    uncertified = False

    for j, lb in enumerate(lbs):
        if lb > 0:
            continue
        x = batch.x.detach().clone()
        x[coord_x[j, :], coord_y[j, :]] = 1 - x[coord_x[j, :], coord_y[j, :]]
        out = gcn(x, batch.edge_index, batch.edge_attr)
        out = out.argmax(dim=1)
        if out[node] != original_label:
            uncertified = True 
            break
    return False, uncertified


def lb_deeppoly_batch(verifier, batch, num_classes, device='cpu'):
    """
    Verify the GCN model using the Deeppoly domain of one batch by getting the lower bounds
    Args:
        verifier: the verifier
        batch: the batch
        num_classes: the number of classes
        device: device for verification
    """
    nodes = list(range(batch.batch_size))
    node_labels = batch.y[:batch.batch_size]
    return verifier.get_lb_backward(nodes, node_labels, num_classes, batch, device)
    