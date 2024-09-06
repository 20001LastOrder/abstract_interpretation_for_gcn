from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import relu
from torch_geometric.nn import SGConv
from torch_geometric.nn.conv.gcn_conv import GCNConv, gcn_norm
from torch_sparse import SparseTensor

from abstract_interpretation import node_interval
from abstract_interpretation.abstract_domain import (
    AbstractDomain,
    AbstractElement,
)
from abstract_interpretation.ConcreteGraphDomains import (
    BinaryNodeFeatureDomain,
)
from abstract_interpretation.node_interval import (
    NodeInterval,
    NodeIntervalElement,
)
from abstract_interpretation.utils import safe_topk


class NodeDeeppolyEle(AbstractElement):
    def __init__(
        self,
        lower_coef: Tensor,
        lower_const: Optional[Tensor],
        upper_coef: Tensor,
        upper_const: Optional[Tensor],
        edge_index,
        edge_attr=None,
        node_id=None,
    ):
        """
        Here, we assume all the features are binary: either 0 or 1 for the
        node entry layer
        """
        self.lower_coef = lower_coef
        self.lower_const = lower_const
        self.upper_coef = upper_coef
        self.upper_const = upper_const
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_id = node_id

    def ub(self, x=None, q=0, center=None):
        """
        Return the upper bound changes of the values
        """
        purtab = (1 - x) + (-1 * x)
        if center is None:
            center = (self.upper_coef * x).sum() + self.upper_const.sum()
        values = (self.upper_coef * purtab).flatten()
        values, _ = torch.topk(values, k=q)
        return center + (values * (values > 0)).sum()

    def lb(self, x=None, q=0, center=None):
        purtab = (1 - x) + (-1 * x)
        if center is None:
            center = (self.lower_coef * x).sum() + self.lower_const.sum()
        values = (self.lower_coef * purtab).flatten()
        values, _ = torch.topk(values, k=q, largest=False)
        return center + (values * (values < 0)).sum()

    def lb_batch(
        self, x=None, global_perturb=0, local_perturb=-1, center=None
    ):
        # coef is in shape [c, N, f]
        # const is in shape [c, N]
        # l is local budget
        device = x.device
        purtab = (1 - x) + (-1 * x)
        if center is None:
            center = (self.lower_coef * x).sum((2, 3)) + self.lower_const.sum(
                2
            )

        values = self.lower_coef * purtab
        if local_perturb != -1:
            values, coord_y = torch.topk(
                values, k=local_perturb, largest=False, dim=3
            )
        else:
            coord_y = torch.tensor(
                range(values.shape[3]), device=device
            ).expand(values.shape)
        # generate a tensor with node ids i.e. [[0,0,0...0,0], [1,1,1,...1]]
        coord_x = (
            torch.tensor(range(coord_y.shape[2]), device=device)
            .unsqueeze(1)
            .expand(coord_y.shape)
            .flatten(start_dim=2)
        )
        coord_y = coord_y.flatten(start_dim=2)
        values = values.flatten(start_dim=2)
        values, idx = safe_topk(values, k=global_perturb, largest=False, dim=2)
        # TODO: Remove indexing with positive effect to make the upperbound
        # more precise
        # idx[values > 0] = -1
        return center + (values * (values < 0)).sum(2), (
            torch.gather(coord_x, 2, idx),
            torch.gather(coord_y, 2, idx),
        )


class AbstractSGConv(node_interval.AbstractSGConv):
    def __init__(self, concrete_gcn, **kwargs):
        """
        For this linear transformation, each output is bound exactly by the
        expression i.e. exp(input) <= output <= exp(output)
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(concrete_gcn, **kwargs)
        self.linear_layer = concrete_gcn.lin
        self.bias = concrete_gcn.bias
        self.weights = self.linear_layer.weight.T
        self.concrete_gcn = concrete_gcn

    def forward_pre_calculation(self, ele, edge_index=None, edge_weights=None):
        # TODO: Integrate edge_weights into the abstract domain
        return super().forward(ele, edge_weights)

    def lin(self, lb: Tensor, ub: Tensor):
        bias = 0 if self.bias is None else self.bias
        pos = (self.weights.detach() > 0) * self.weights
        neg = (self.weights.detach() < 0) * self.weights

        new_lb = lb @ pos + ub @ neg + bias
        new_ub = ub @ pos + lb @ neg + bias
        return new_lb, new_ub

    def backward_deeppoly(
        self, edge_index, edge_weights, exp: Tensor, consts: Tensor
    ):
        """
        Process the expression backward
        exp: backward node feature matrix [c, N, F]
        """
        # process linear lays
        # append the bias to linear weights
        weights = self.linear_layer.weight.T
        consts = consts + (exp * self.bias).sum(3)
        # exp = (weights @ exp.T).T
        exp = exp @ weights.T

        # process gcn layers, (unconvolve, using the transposed edge index)
        torch.permute(exp, [2, 0, 1, 3])
        edge_index = edge_index[[1, 0]]
        exp = self.propagate(edge_index, x=exp, edge_weights=edge_weights)
        torch.permute(exp, [1, 2, 0, 3])

        return exp, consts

    def backward(self, ele: NodeDeeppolyEle):
        """
        Process the expression backward (for one end expression)
        exp: backward node feature matrix
        const: the constant term for each node
        """
        # process linear lays
        # append the bias to linear weights
        # exp shape: [c, N, F], const shape[c, N]
        upper_exp, upper_const = ele.upper_coef, ele.upper_const
        lower_exp, lower_const = ele.lower_coef, ele.lower_const
        edge_index = ele.edge_index
        edge_weights = ele.edge_attr

        upper_exp, upper_const = self.backward_deeppoly(
            edge_index, edge_weights, upper_exp, upper_const
        )
        lower_exp, lower_const = self.backward_deeppoly(
            edge_index, edge_weights, lower_exp, lower_const
        )

        return NodeDeeppolyEle(
            lower_exp,
            lower_const,
            upper_exp,
            upper_const,
            edge_index,
            edge_weights,
            ele.node_id,
        )


class AbstractLinear(node_interval.AbstractLinear):
    def __init__(self, linear, **kwargs):
        """
        For this linear transformation, each output is bound exactly by the
            expression i.e. exp(input) <= output <= exp(output)
        """
        super().__init__(linear, **kwargs)
        self.bias = linear.bias
        self.weights = linear.weight.T

    def forward_pre_calculation(self, ele, edge_index=None, edge_weights=None):
        return super().forward(ele, edge_weights)

    def backward_deeppoly(self, exp: Tensor, consts: Tensor):
        """
        Process the expression backward
        exp: backward node feature matrix [c, N, F]
        """
        # process linear lays
        # append the bias to linear weights
        weights = self.linear_layer.weight.T
        consts = consts + (exp * self.bias).sum(3)
        # exp = (weights @ exp.T).T
        exp = exp @ weights.T

        return exp, consts

    def backward(self, ele: NodeDeeppolyEle):
        """
        Process the expression backward (for one end expression)
        exp: backward node feature matrix
        const: the constant term for each node
        """
        # process linear lays
        # append the bias to linear weights
        # exp shape: [c, N, F], const shape[c, N]
        upper_exp, upper_const = ele.upper_coef, ele.upper_const
        lower_exp, lower_const = ele.lower_coef, ele.lower_const
        edge_index = ele.edge_index
        edge_weights = ele.edge_attr

        upper_exp, upper_const = self.backward_deeppoly(upper_exp, upper_const)
        lower_exp, lower_const = self.backward_deeppoly(lower_exp, lower_const)

        return NodeDeeppolyEle(
            lower_exp,
            lower_const,
            upper_exp,
            upper_const,
            edge_index,
            edge_weights,
            ele.node_id,
        )


class AbstractRelu(torch.nn.Module):
    def __init__(self, other=None):
        super().__init__()
        self.upper_coefs = None
        self.upper_consts = None
        self.lower_coefs = None
        self.lower_consts = None
        self.lam = None

    def forward_pre_calculation(
        self,
        abstract_ele: NodeIntervalElement,
        edge_index=None,
        edge_weights=None,
    ) -> NodeIntervalElement:
        """
        Input x_j: output x_i
        """
        upper_bounds, lower_bounds = abstract_ele.ub(), abstract_ele.lb()
        # print(lower_bounds[0, 23])
        # print(upper_bounds[0, 23])

        self.upper_coefs = torch.zeros_like(
            upper_bounds, device=upper_bounds.device
        )
        self.upper_consts = torch.zeros_like(
            upper_bounds, device=upper_bounds.device
        )
        self.lower_coefs = torch.zeros_like(
            upper_bounds, device=upper_bounds.device
        )
        self.lower_consts = torch.zeros_like(
            upper_bounds, device=upper_bounds.device
        )

        # case 1: the lower bound is larger than zero (x_j <= x_i <= x_j)
        self.lower_coefs = (lower_bounds > 0) * torch.ones_like(
            lower_bounds, device=upper_bounds.device
        )
        self.upper_coefs = (lower_bounds > 0) * torch.ones_like(
            upper_bounds, device=upper_bounds.device
        )

        # case 2: the upper bound is smaller than zero (0<=x_i<=0): default,
        #   no action
        # case 3: the lower bound is smaller than zero and the upper bound is
        #   larger than zero
        # 0 <= x_j <= u_ix_i / (u_i - l_i) - l_iu_i / (u_i - l_i)
        #   when |L| > |U| : only need to update the upper coefs
        selection = torch.logical_and((lower_bounds < 0), (upper_bounds > 0))
        selected_upper = selection * upper_bounds
        selected_lower = selection * lower_bounds + (
            ~selection
        )  # add one to other values to avoid devision by zero
        self.upper_coefs += selected_upper / (selected_upper - selected_lower)
        self.upper_consts += -(selected_upper * selected_lower) / (
            selected_upper - selected_lower
        )

        lower_lam = self.lam if self.lam is not None else 0
        selection_lower = torch.logical_and(
            selection, lower_bounds.abs() >= upper_bounds.abs()
        )
        self.lower_coefs = (
            self.lower_coefs + lower_lam * selection_lower.float()
        )

        # case 4: the lower bound is smaller than zero and the upper bound is
        #   larger than zero
        # x_i <= x_j <= u_ix_i / (u_i - l_i) - l_iu_i / (u_i - l_i)
        #   when |L| < |U|: update the lower coefs
        #   (upper has been updated in case 3)
        selection_upper = torch.logical_and(
            selection, lower_bounds.abs() < upper_bounds.abs()
        )
        self.lower_coefs = self.lower_coefs + selection_upper.float()

        # return normal interval bounds
        lb = abstract_ele.lb()
        ub = abstract_ele.ub()

        # the lb needs to be separated because now the lower bound may be
        #   negative
        new_lb = relu(lb) + selection * lb
        return NodeIntervalElement(new_lb, relu(ub), abstract_ele.edge_index)

    def forward(self, ele: NodeIntervalElement, edge_weights=None):
        """
        forward the value with upper and lower neural network
        the constant is ignored
        """
        ub, lb = ele.ub(), ele.lb()
        # print(ele.node_id[0])
        # print(lb[0, 23])
        # print(ub[0, 23])
        node_id = ele.node_id
        upper_coefs = self.upper_coefs[node_id, :]
        lower_coefs = self.lower_coefs[node_id, :]
        upper_consts = self.upper_consts[node_id, :]
        lower_consts = self.lower_consts[node_id, :]

        ub = upper_coefs * ub + upper_consts
        lb = lower_coefs * lb + lower_consts
        if not torch.all(ub >= lb - 1e-4):
            idx = (ub < lb - 1e-4).nonzero(as_tuple=True)
            print(idx)
            # print(lb[0, 23])
            # print(ub[0, 23])

        # only check for the batch size
        assert torch.all((ub >= lb - 1e-4)[:8, :])

        return NodeIntervalElement(lb, ub, ele.edge_index, node_id=node_id)

    def backward(self, ele: NodeDeeppolyEle):
        upper_exp, upper_const = ele.upper_coef, ele.upper_const
        lower_exp, lower_const = ele.lower_coef, ele.lower_const
        edge_index = ele.edge_index
        node_id = ele.node_id

        upper_coefs = self.upper_coefs[node_id, :]
        lower_coefs = self.lower_coefs[node_id, :]
        upper_consts = self.upper_consts[node_id, :]
        lower_consts = self.lower_consts[node_id, :]

        # calculate the new upper exp and upper consts
        upper_pos = (upper_exp > 0) * upper_exp
        upper_neg = (upper_exp < 0) * upper_exp
        upper_exp = upper_pos * upper_coefs + upper_neg * lower_coefs
        upper_const += (
            upper_pos * upper_consts + upper_neg * lower_consts
        ).sum(dim=3)

        # calculate the new lower exp and lower consts
        lower_pos = (lower_exp > 0) * lower_exp
        lower_neg = (lower_exp < 0) * lower_exp
        lower_exp = lower_pos * lower_coefs + lower_neg * upper_coefs
        lower_const += (
            lower_pos * lower_consts + lower_neg * upper_consts
        ).sum(dim=3)
        return NodeDeeppolyEle(
            lower_exp,
            lower_const,
            upper_exp,
            upper_const,
            edge_index,
            ele.edge_attr,
            node_id,
        )


class AbstractFeatureNormalization:
    # Abstract layer for feature normalization (row-wise)
    # The normalization is equivalent to multiply the value by a constant
    #   given the features

    def __init__(self):
        self.upper_coef = None
        self.lower_coef = None

    def forward_pre_calculation(self, ele: BinaryNodeFeatureDomain):
        x = ele.x
        local_perturb = ele.l

        # calculate the sum of each row as normalization factor
        norm = x.sum(dim=1, keepdim=True)
        num_features = x.shape[1]
        lb = x - x
        ones = torch.ones_like(x, device=x.device)
        ones_norm = torch.ones_like(norm, device=norm.device)
        ub = ones / torch.max(ones_norm, norm - local_perturb)

        self.lower_coef = ones / torch.min(
            num_features * ones_norm, norm + local_perturb
        )
        self.upper_coef = ub.detach().clone()

        return NodeIntervalElement(lb, ub, ele.edge_index)

    def backward(self, ele: NodeDeeppolyEle):
        upper_exp = ele.upper_coef
        lower_exp = ele.lower_coef

        upper_coefs = self.upper_coef[ele.node_id]
        lower_coefs = self.lower_coef[ele.node_id]

        upper_pos = (upper_exp > 0) * upper_exp
        upper_neg = (upper_exp < 0) * upper_exp
        upper_exp = upper_pos * upper_coefs + upper_neg * lower_coefs

        lower_pos = (lower_exp > 0) * lower_exp
        lower_neg = (lower_exp < 0) * lower_exp
        lower_exp = lower_pos * lower_coefs + lower_neg * upper_coefs

        ele.upper_coef = upper_exp
        ele.lower_coef = lower_exp

        return ele


ABSTRACTION_MAP = {
    GCNConv: AbstractSGConv,
    SGConv: AbstractSGConv,
    torch.nn.ReLU: AbstractRelu,
    torch.nn.Linear: AbstractLinear,
}


def normalize(x):
    return x / x.sum(dim=1, keepdim=True)


class NodeDeepployDomain(AbstractDomain):
    @torch.no_grad()
    def to_abstract(
        self,
        ele: BinaryNodeFeatureDomain,
        layers=None,
        abstract_layers=None,
        approx="max",
        feature_norm=None,
        edge_weights=None,
        node_feature_mask=None,
    ):
        """
        Args:
        - ele (BinaryNodeFeatureDomain): the input node feature domain
        - layers (list): the list of layers
        - abstract_layers (list): the list of abstract layers
        - approx (str): the approximation method
        """
        if len(layers) == 0:
            return None

        interval_domain = NodeInterval()

        if feature_norm is None:
            # if no feature normalization, run the normal interval
            #   approximation with perturbation
            h = layers[0](ele.x, ele.edge_index, edge_weights)
            abs_ele = interval_domain.to_abstract(
                ele,
                h,
                layers[0].lin.weight.T,
                approx=approx,
                norm=edge_weights,
                node_feature_mask=node_feature_mask,
            )
        else:
            # first run feature normalization before run the rest
            h = layers[0](normalize(ele.x), ele.edge_index, edge_weights)
            abs_ele = feature_norm.forward_pre_calculation(ele)
            # simplify -(abs_ele.ub() * ele.x) + (abs_ele.ub() * (1 - ele.x))
            perturb = -abs_ele.ub() * (2 * ele.x - 1)
            abs_ele = interval_domain.to_abstract(
                ele,
                h,
                layers[0].lin.weight.T,
                approx=approx,
                purtab=perturb,
                node_feature_mask=node_feature_mask,
            )

        abstract_layers = abstract_layers[1:]
        for i, layer in enumerate(abstract_layers):
            abs_ele = layer.forward_pre_calculation(
                abs_ele, ele.edge_index, edge_weights
            )

        return abs_ele

    def transform(self, ele: AbstractElement, layers, edge_weights=None):
        for i, layer in enumerate(layers):
            ele = layer(ele, edge_weights)
        return ele

    def transform_backward(self, ele: AbstractElement, layers):
        for layer in reversed(layers):
            ele = layer.backward(ele)
        return ele

    def transform_single_layer(self, ele: AbstractElement, layer):
        return layer(ele)

    def to_concrete(self, ele: AbstractElement):
        pass


class NodeDeeppolyVerifier:
    def __init__(
        self,
        layers,
        concrete_domain,
        approx="topk",
        feature_norm=False,
        edge_weights=None,
        lam=None,
        node_feature_mask=None,
    ):
        """
        Args:
            layers (list): list of gnn layers
            concrete_domain (NodeDeeppolyDomain): concrete domain for node
                perturbation
            approx (str): approximation method (topk or max)
            feature_norm (AbstractFeatureNormalization): feature normalization
                weather to use feature normalization layer
            edge_weights (torch.tensor): weather use pre-computed edge weights
                (this should be done for mini-batches) or compute edge weights
                automatically
        """
        layers = [layer for layer in layers if type(layer) in ABSTRACTION_MAP]
        self.abstract_layers = [
            ABSTRACTION_MAP[type(layer)](layer) for layer in layers
        ]
        self.lam = lam
        # assign lambda for relu
        for layer in self.abstract_layers:
            if type(layer) == AbstractRelu:
                layer.lam = self.lam

        self.feature_norm = feature_norm
        self.abs_norm = AbstractFeatureNormalization()
        self.abstract_domain = NodeDeepployDomain()

        self.abstract_domain.to_abstract(
            concrete_domain,
            layers,
            self.abstract_layers,
            approx,
            feature_norm=self.abs_norm if feature_norm else None,
            edge_weights=edge_weights,
            node_feature_mask=node_feature_mask,
        )

        self.x = concrete_domain.x.detach().clone()
        self.global_budget = concrete_domain.q
        self.local_perturb = concrete_domain.l
        self.edge_index = concrete_domain.edge_index

    def get_lower(self, lb_value: torch.tensor):
        lb_center = lb_value.item()
        lower_coef = self.get_linear_combination(lb_value)
        symbolic_ele = NodeDeeppolyEle(lower_coef, None, None, None, None)
        return symbolic_ele.lb(self.x, self.global_budget, lb_center)

    def get_bound(self, lb_value: torch.Tensor, ub_value: torch.Tensor):
        lb_center = lb_value.item()
        ub_center = ub_value.item()
        lower_coef = self.get_linear_combination(lb_value)
        upper_coef = self.get_linear_combination(ub_value)
        symbolic_ele = NodeDeeppolyEle(lower_coef, None, upper_coef, None)
        lb = symbolic_ele.lb(self.x, self.global_budget, lb_center)
        ub = symbolic_ele.ub(self.x, self.global_budget, ub_center)
        return lb, ub

    def backward_ele(
        self, nodes, target_classes, num_classes, data, device
    ) -> NodeDeeppolyEle:
        batch_size = len(nodes)
        num_nodes = data.x.shape[0]
        exp = torch.zeros(
            (batch_size, num_classes, num_classes), device=device
        )

        minus = torch.eye(num_classes, device=device)
        exp[range(batch_size), :, target_classes] = 1
        exp = exp - minus
        choosed_labels = (exp != 0).any(2)
        exp = exp[choosed_labels].reshape(batch_size, -1, num_classes)

        comb = torch.zeros(
            [batch_size, num_classes - 1, num_nodes, num_classes],
            device=device,
        )
        comb[range(batch_size), :, nodes, :] = exp

        lower_const = torch.zeros(
            (batch_size, num_classes - 1, num_nodes), device=device
        )
        upper_const = torch.zeros(
            (batch_size, num_classes - 1, num_nodes), device=device
        )
        ele = NodeDeeppolyEle(
            comb,
            lower_const,
            comb,
            upper_const,
            data.edge_index,
            data.edge_attr,
            data.node_id,
        )

        layers = (
            [self.abs_norm] + self.abstract_layers
            if self.feature_norm
            else self.abstract_layers
        )
        ele = self.abstract_domain.transform_backward(ele, layers)
        return ele

    def get_lb_backward(
        self, nodes, target_classes, num_classes, data, device
    ):
        ele = self.backward_ele(
            nodes, target_classes, num_classes, data, device
        )
        return ele.lb_batch(
            data.x, self.global_budget, local_perturb=self.local_perturb
        )

    def forward(self, h, edge_index, node_id, edge_weights):
        """does not support feature norm yet
        Args:
            h (torch.tensor): node features after the first GCN layer
        """
        # use the abstract domain to propagate the rest of the layers
        layers = self.abstract_layers[1:]
        ele = NodeIntervalElement(h, h, edge_index, node_id=node_id)

        ele = self.abstract_domain.transform(ele, layers, edge_weights)
        return ele

    def get_label_lb(self, nodes, target_classes, num_classes, data, device):
        batch_size = len(nodes)
        num_nodes = data.x.shape[0]
        # only perform abstract interpretation for 1 class
        #   (i.e. the target class)
        exp = torch.zeros((batch_size, 1, num_classes), device=device)

        exp[range(batch_size), :, target_classes] = 1

        comb = torch.zeros(
            [batch_size, 1, num_nodes, num_classes], device=device
        )
        comb[range(batch_size), :, nodes, :] = exp

    def get_linear_combination(self, value: torch.Tensor):
        self.x.grad = None
        value.backward(retain_graph=True)
        assert self.x.grad is not None
        return self.x.grad
