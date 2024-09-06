from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import relu
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SGConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv, gcn_norm
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor, matmul

from abstract_interpretation.abstract_domain import (
    AbstractDomain,
    AbstractElement,
)
from abstract_interpretation.ConcreteGraphDomains import (
    BinaryNodeFeatureDomain,
)
from abstract_interpretation.utils import safe_topk


def sparse_collect(src: torch.Tensor, index: torch.Tensor, num_nodes: int):
    output_dim = src.shape[-1]
    collect = []
    for i in range(num_nodes):
        result = src[torch.where(index == i)]
        result = result.view(-1, output_dim)
        collect.append(result)
    return collect


class NodeIntervalElement(AbstractElement):
    _ub: torch.Tensor
    _lb: torch.Tensor
    edge_index: Adj

    def __init__(
        self, lb: torch.Tensor, ub: torch.Tensor, edge_index: Adj, node_id=None
    ):
        self._lb = lb
        self._ub = ub
        self.edge_index = edge_index
        self.node_id = node_id

    def ub(self):
        return self._ub

    def lb(self):
        return self._lb

    def inside(self, point: torch.tensor, eps=1e-5):
        return (
            torch.ge(point, self._lb - eps).all()
            and torch.le(point, self._ub + eps).all()
        )


class AbstractSGConv(MessagePassing):
    def __init__(self, concrete_gcn: Union[GCNConv, SGConv], **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.linear_layer = concrete_gcn.lin
        self.bias = concrete_gcn.bias
        self.add_self_loops = concrete_gcn.add_self_loops

    def reset_parameters(self):
        self.linear_layer.reset_parameters()

    def forward(
        self, abstract_ele: NodeIntervalElement, edge_weights=None
    ) -> NodeIntervalElement:
        edge_index = abstract_ele.edge_index
        lb, ub = abstract_ele.lb(), abstract_ele.ub()
        num_nodes = lb.shape[0]

        if edge_weights is None:
            if isinstance(edge_index, Tensor):
                # edge_index is the edge list
                edge_index, edge_weights = gcn_norm(
                    edge_index,
                    num_nodes=num_nodes,
                    add_self_loops=self.add_self_loops,
                )
            elif isinstance(edge_index, SparseTensor):
                # edge_index is a sparse adj matrix
                edge_index, edge_weights = gcn_norm(
                    edge_index,
                    num_nodes=num_nodes,
                    add_self_loops=self.add_self_loops,
                )

        # propagate the lower and upper bound
        ub = self.propagate(edge_index, x=ub, edge_weights=edge_weights)
        lb = self.propagate(edge_index, x=lb, edge_weights=edge_weights)

        # only check for the batch size, the ub should be always larger than
        # the lb
        if not torch.all((ub >= lb - 1e-4)[:8, :]):
            idx = (ub < lb - 1e-4)[:8, :].nonzero(as_tuple=True)
            print(idx)
        assert torch.all(
            (ub >= lb - 1e-4)[:8, :]
        ), "the upper bound should be always be larger than the lower bound"

        # apply affine transformation
        lb, ub = self.lin(lb, ub)
        # if not torch.all((ub >= lb - 1e-4)[:8, :]):
        #     idx = (ub < lb - 1e-4)[:8, :].nonzero(as_tuple=True)
        #     print(idx)
        assert torch.all(
            (ub >= lb - 1e-4)[:8, :]
        ), "the upper bound should be always be larger than the lower bound"

        return NodeIntervalElement(
            lb, ub, edge_index, node_id=abstract_ele.node_id
        )

    def lin(self, lb: Tensor, ub: Tensor) -> Tuple[Tensor, Tensor]:
        weight = self.linear_layer.weight.T
        bias = 0 if self.bias is None else self.bias
        pos = (weight.detach() > 0) * weight
        neg = (weight.detach() < 0) * weight
        new_lb = lb @ pos + ub @ neg + bias
        new_ub = ub @ pos + lb @ neg + bias
        return new_lb, new_ub

    def message_and_aggregate(
        self, adj_t: SparseTensor, x: Tensor = None
    ) -> Tensor:
        if x is None:
            raise ValueError("The value of x cannot be None")
        return matmul(adj_t, x, reduce=self.aggr)

    def message(self, x_j: Tensor, edge_weights: Tensor = None) -> Tensor:
        if edge_weights is None:
            raise ValueError("The value of the edge_weights list is needed")
        return edge_weights.view(-1, 1) * x_j

    def edge_update(self) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        in_channels, out_channels = self.linear_layer.weight.shape
        return f"{self.__class__.__name__}({in_channels}, " f"{out_channels})"


class AbstractLinear(torch.nn.Module):
    def __init__(self, linear_layer: torch.nn.Linear):
        super().__init__()
        self.linear_layer = linear_layer

    def forward(
        self, abstract_ele: NodeIntervalElement, edge_weights=None
    ) -> NodeIntervalElement:
        lb, ub = abstract_ele.lb(), abstract_ele.ub()
        weight = self.linear_layer.weight.T
        bias = 0 if self.linear_layer.bias is None else self.linear_layer.bias
        pos = (weight > 0) * weight
        neg = (weight < 0) * weight
        new_lb = lb @ pos + ub @ neg + bias
        new_ub = ub @ pos + lb @ neg + bias
        return NodeIntervalElement(new_lb, new_ub, abstract_ele.edge_index)


class AbstractRelu(torch.nn.Module):
    def __init__(self, other):
        super().__init__()

    def forward(
        self, abstract_ele: NodeIntervalElement, edge_weights=None
    ) -> NodeIntervalElement:
        lb = abstract_ele.lb()
        ub = abstract_ele.ub()
        return NodeIntervalElement(relu(lb), relu(ub), abstract_ele.edge_index)


class NodeInterval(AbstractDomain):
    # Implement the abstact domain for binary node features proposed in
    #  Abstract Interpretation based Robustness
    # Certification for Graph Convolutional Networks
    abstraction_map = {
        GCNConv: AbstractSGConv,
        SGConv: AbstractSGConv,
        torch.nn.Linear: AbstractLinear,
        torch.nn.ReLU: AbstractRelu,
    }

    def to_abstract(
        self,
        concrete_domain: BinaryNodeFeatureDomain,
        h=None,
        w=None,
        include_unperturbed=True,
        approx="max",
        purtab=None,
        norm=None,
        node_feature_mask: torch.BoolTensor = None,
    ) -> NodeIntervalElement:
        """
        Convert the concrete domain to the abstract domain by applying the
        first layer of the GCN model and convert the feature from binary
        domain to continuous domain.

        Args:
        concrete_domain (BinaryNodeFeatureDomain): the concrete domain of
            graph perpurbation space
        h (Tensor): the hidden node representation after one GCN layer of
            shape (N, H1)
        w (Tensor): the weights of the first GCN layer of shape (H0, H1)
        include_unperturbed (Tensor): should the resulting abstraction include
            the unperturbed version of the graph?
        """
        if h is None or w is None:
            raise ValueError(
                "Value of h and w are required to compute the abstract domain"
            )

        edge_index = concrete_domain.edge_index
        edge_index_org = edge_index
        x = concrete_domain.x
        q = concrete_domain.q
        local_p = concrete_domain.l

        # linear transform the possible node features perturbation but keep
        # all channels
        if purtab is None:
            purtab = (1 - x) + (-1 * x)

        # mask the features that are fixed
        if node_feature_mask is not None:
            purtab = purtab * node_feature_mask

        purtab = purtab.unsqueeze(2) * w

        if norm is None:
            edge_index, norm = gcn_norm(edge_index, num_nodes=x.shape[0])

        # normalize the adjacency matrix using the GCN normalization method as
        # in the paper
        if approx == "max":
            lower, _ = purtab.min(dim=1)
            upper, _ = purtab.max(dim=1)
            adj_t = SparseTensor.from_edge_index(edge_index, norm).t()
            upper_bound = matmul(adj_t, upper, reduce="max") * q
            lower_bound = matmul(adj_t, lower, reduce="min") * q
        else:

            def calculate_bound(lower=True):
                # for some reason this step fails on CUDA
                topk_purtab = safe_topk(
                    purtab, local_p, dim=1, largest=not lower
                )[0]

                #  only consider the first k for lower / upper bound
                message = norm.reshape(-1, 1, 1) * topk_purtab[edge_index[0]]

                # collect neighbours for each node
                nodes = sparse_collect(message, edge_index[1], x.shape[0])

                # TODO this step is very memory itensive, maybe consider
                # change this later
                # padding based on the node with largest neighbour so that we
                # can do batch calculation
                nodes = pad_sequence(nodes, batch_first=True)

                topk_purtab = safe_topk(nodes, q, dim=1, largest=not lower)[0]

                if lower:
                    valid_purtab = torch.where(
                        topk_purtab < 0,
                        topk_purtab,
                        torch.zeros(1, device=topk_purtab.device),
                    )
                else:
                    valid_purtab = torch.where(
                        topk_purtab > 0,
                        topk_purtab,
                        torch.zeros(1, device=topk_purtab.device),
                    )
                return valid_purtab.sum(dim=1)

            # top k based method
            purtab = purtab.cpu()
            norm = norm.cpu()
            edge_index = edge_index.cpu()
            # purtab = torch.sort(purtab, dim=1)[0]
            lower_bound = calculate_bound(True)
            upper_bound = calculate_bound(False)
            lower_bound = lower_bound.to(h.device)
            upper_bound = upper_bound.to(h.device)

        upper_bound = (
            torch.maximum(h, h + upper_bound)
            if include_unperturbed
            else h + upper_bound
        )
        lower_bound = (
            torch.minimum(h, h + lower_bound)
            if include_unperturbed
            else h + lower_bound
        )
        assert torch.all(upper_bound > lower_bound - 1e-4)
        abstract_ele = NodeIntervalElement(
            lower_bound, upper_bound, edge_index_org
        )

        return abstract_ele

    def transform(
        self,
        ele: BinaryNodeFeatureDomain,
        layers: List[torch.nn.Module],
        approx="max",
        edge_weights=None,
        node_feature_mask: torch.BoolTensor = None,
    ) -> Optional[AbstractElement]:
        """
        Transform the concrete domain to the abstract domain by applying the
        GCN model and convert the feature from binary domain to continuous
        domain.

        Args:
        ele (BinaryNodeFeatureDomain): the concrete domain of graph
            perpurbation space
        layers (List[torch.nn.Module]): the list of GCN layers
        approx (str): the approximation method to use for the abstract domain
            calculation, either "max" or "topk"
        edge_weights (Tensor): the edge weights of the graph
        node_feature_mask (Tensor): the mask of the node features that are
            fixed
        """
        if len(layers) == 0:
            return None
        h = layers[0](ele.x, ele.edge_index, edge_weights)
        abs_ele = self.to_abstract(
            ele,
            h,
            layers[0].lin.weight.T,
            approx=approx,
            norm=edge_weights,
            node_feature_mask=node_feature_mask,
        )

        for layer in layers[1:]:
            abs_ele = self.transform_single_layer(abs_ele, layer, edge_weights)

        return abs_ele

    def transform_single_layer(
        self,
        ele: NodeIntervalElement,
        layer: List[torch.nn.Module],
        edge_weights: torch.FloatTensor = None,
    ) -> AbstractElement:
        abstractor = self.abstraction_map[type(layer)]
        abstract_layer = abstractor(layer)
        return abstract_layer(ele, edge_weights)

    def to_concrete(self, ele: NodeIntervalElement):
        pass


def derive_node_certification(
    data: Data,
    lb: Iterable[Iterable[float]],
    ub: Iterable[Iterable[float]],
    N: int,
) -> List[float]:
    """
    Derive node certification from the lower and upper bounds of the graph
    perturbation

    Args:
        data (Data): the graph to certify
        lb (Iterable[Iterable[float]]): the lower bounds of the graph
            perturbation
        ub (Iterable[Iterable[float]]): the upper bounds of the graph
            perturbation
        N (int): the number of nodes to certify
    """

    certification = []

    for node in range(N):
        label = data.labels[node]
        max_up = max([value for i, value in enumerate(ub[node]) if i != label])

        node_cert = lb[node][label] - max_up

        certification.append(node_cert.item())

    return certification


def certify_gcn_full(
    gcn: torch.nn.Module,
    data: Data,
    global_budget: int,
    device: str = "cpu",
    approx: str = "max",
    return_node_result: bool = False,
    node_feature_mask: torch.BoolTensor = None,
) -> Union[float, Tuple[float, List[float]]]:
    """
    Certify a GCN model using the full graph perturbation method

    Args:
        gcn (torch.nn.Module): the GCN model to certify
        data (Data): the graph to certify
        global_budget (int): the global budget of the graph perturbation
        device (str): the device to run the certification on
        approx (str): the approximation method to use, either "max" or "topk"
        return_node_result (bool): whether to return the node-wise
            certification
        node_feature_mask (Tensor): the mask of the node features that are
            fixed
    """
    gcn = gcn.to(device)
    data = data.to(device)

    N = data.x.shape[0]
    abstract_domain = NodeInterval()
    layers = list(gcn.children())

    concrete_domain = BinaryNodeFeatureDomain(
        data.edge_index, global_budget, data.x
    )
    abstract_ele = abstract_domain.transform(
        concrete_domain,
        layers,
        approx=approx,
        edge_weights=data.edge_attr,
        node_feature_mask=node_feature_mask,
    )

    certification = derive_node_certification(
        data, abstract_ele.lb(), abstract_ele.ub(), N
    )

    result = sum([node_cert > 0 for node_cert in certification]) / N
    result = result

    if return_node_result:
        return result, certification
    else:
        return result


def certify_gcn_batch(
    gcn: torch.nn.Module,
    loader: DataLoader,
    global_budget: int,
    device: str = "cpu",
    approx: str = "max",
    return_node_result: bool = False,
    node_feature_mask: torch.BoolTensor = None,
) -> Union[float, Tuple[float, List[float]]]:
    """
    Certify a GCN model using batch

    Args:
        gcn (torch.nn.Module): the GCN model to certify
        loader (DataLoader): the graph to certify
        global_budget (int): the global budget of the graph perturbation
        device (str): the device to run the certification on
        approx (str): the approximation method to use, either "max" or "topk"
        return_node_result (bool): whether to return the node-wise
            certification
        node_feature_mask (Tensor): the mask of the node features that are
            fixed
    """

    gcn = gcn.to(device)
    abstract_domain = NodeInterval()
    layers = list(gcn.children())

    certification = []
    N = 0
    for data in loader:
        data = data.to(device)
        batch_size = data.batch_size

        N += batch_size

        concrete_domain = BinaryNodeFeatureDomain(
            data.edge_index, global_budget, data.x
        )
        abstract_ele = abstract_domain.transform(
            concrete_domain,
            layers,
            approx=approx,
            edge_weights=data.edge_attr,
            node_feature_mask=node_feature_mask,
        )

        batch_certification = derive_node_certification(
            data, abstract_ele.lb(), abstract_ele.ub(), batch_size
        )

        certification += batch_certification

    result = sum([node_cert > 0 for node_cert in certification]) / N

    if return_node_result:
        return result, certification
    else:
        return result
