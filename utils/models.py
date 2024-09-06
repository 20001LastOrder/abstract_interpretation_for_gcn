import torch
import torch.nn as nn
from torch.nn import Dropout, ReLU
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GCNConv


def get_weights(folder, dataset, is_robust=False):
    weight_path = f"{folder}/{dataset}"
    if is_robust:
        weight_path = f"{weight_path}_robust_gcn.pkl"
    else:
        weight_path = f"{weight_path}_gcn.pkl"

    state_dict = torch.load(weight_path, map_location="cpu")

    weights = [
        v for k, v in state_dict.items() if "weight" in k and "conv" in k
    ]
    biases = [v for k, v in state_dict.items() if "bias" in k and "conv" in k]

    W1, W2 = [w.cpu().detach() for w in weights]
    b1, b2 = [b.cpu().detach() for b in biases]

    return (W1, W2), (b1, b2)


def get_model(
    folder, dataset, is_robust=False, normalize=True, load_weights=True
):
    # these weights are used to get parameter configurations of the model
    (W1, W2), (b1, b2) = get_weights(folder, dataset, is_robust)

    layer_1 = GCNConv(
        in_channels=W1.shape[0], out_channels=W1.shape[1], normalize=normalize
    )
    layer_2 = GCNConv(
        in_channels=W2.shape[0], out_channels=W2.shape[1], normalize=normalize
    )

    # load the weights if necessary
    if load_weights:
        layer_1.lin.weight = torch.nn.Parameter(W1.T)
        layer_1.bias = torch.nn.Parameter(b1)
        layer_2.lin.weight = torch.nn.Parameter(W2.T)
        layer_2.bias = torch.nn.Parameter(b2)
    else:
        layer_1.lin.weight = torch.nn.Parameter(
            nn.Parameter(
                nn.init.xavier_normal_(torch.zeros(W1.shape[1], W1.shape[0]))
            )
        )
        layer_1.bias = torch.nn.Parameter(
            nn.Parameter(nn.init.normal_(torch.zeros(b1.shape[0])))
        )
        layer_2.lin.weight = torch.nn.Parameter(
            nn.Parameter(
                nn.init.xavier_normal_(torch.zeros(W2.shape[1], W2.shape[0]))
            )
        )
        layer_2.bias = torch.nn.Parameter(
            nn.Parameter(nn.init.normal_(torch.zeros(b2.shape[0])))
        )

    if normalize:
        gcn = Sequential(
            "x, edge_index",
            [
                (layer_1, "x, edge_index -> x"),
                (ReLU(), "x -> x"),
                (layer_2, "x, edge_index -> x"),
            ],
        )
    else:
        gcn = Sequential(
            "x, edge_index, edge_weights",
            [
                (layer_1, "x, edge_index, edge_weights -> x"),
                (ReLU(), "x -> x"),
                # (Dropout(0.6), 'x -> x'),
                (layer_2, "x, edge_index, edge_weights -> x"),
            ],
        )

    return gcn


def create_GCN_layers(layers, normalize=False):
    result_layers = []
    for layer in layers:
        weights = layer.weights
        bias = layer.bias
        result_layer = GCNConv(
            in_channels=weights.shape[0],
            out_channels=weights.shape[1],
            normalize=normalize,
        )
        result_layer.lin.weight = torch.nn.Parameter(weights.T)
        result_layer.bias = torch.nn.Parameter(bias)
        result_layers.append(result_layer)
        result_layers.append(ReLU())

    # ignore the last ReLU
    return result_layers[:-1]
