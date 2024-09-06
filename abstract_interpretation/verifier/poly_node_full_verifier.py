"""
Provide a full verification for nodes on a graph using GCN and polyhedra
abstract interpretation. It calculates a perturbation matrix on how many
node features can be added or deleted while the node stay robust.
"""
from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import Data

from abstract_interpretation.ConcreteGraphDomains import (
    BinaryNodeFeatureDomain,
)
from abstract_interpretation.node_deeppoly import (
    NodeDeeppolyEle,
    NodeDeeppolyVerifier,
)
from abstract_interpretation.verifier.base import BaseVerifier
from abstract_interpretation.verifier.verification_budget import (
    VerificationPerturbBudget,
)


class PolyNodeFullVerifier(BaseVerifier):
    def __init__(
        self,
        gcn: torch.nn.Module,
        data: Data,
        budget: VerificationPerturbBudget,
        gloabl_budget: int,
        approx: str = "topk",
    ):
        """
        Args:
            gcn (torch.Module): GCN model
            data (Data): graph data
            budget (VerificationPerturbBudget): budget for verification,
                including the number of added and removed features
            gloabl_budget (int): the global budget for the perturbation
            approx (str, optional): approximation method for the verification.
        """
        self.gcn = gcn
        self.data = data
        self.budget = budget
        self.approx = approx

        self.layers = list(gcn.children())
        self.global_budget = gloabl_budget

        concrete_domain = BinaryNodeFeatureDomain(
            data.edge_index, self.global_budget, data.x, l=self.global_budget
        )

        feature_mask = None
        if self.budget.feature_addition == 0:
            feature_mask = torch.ones_like(data.x)
        elif self.budget.feature_removal == 0:
            feature_mask = 1 - data.x

        self.verifier = NodeDeeppolyVerifier(
            self.layers,
            concrete_domain,
            approx=approx,
            edge_weights=data.edge_attr,
            node_feature_mask=feature_mask,
        )

        self.num_classes = data.y.max().item() + 1
        self.num_nodes = data.x.shape[0]

    def verify_batch(
        self, batch: Data
    ) -> Dict[int, Dict[Tuple[int, int], bool]]:
        """
        Verify a batch of nodes

        Args:
            batch (Data): batch of nodes to verify

        Returns:
            Dict[int, Dict[Tuple[int, int], bool]]: verification results
                from node id to a dictionary of (feature addition, feature
                deletion) to if the node is robust
        """

        nodes = list(range(batch.batch_size))
        node_labels = batch.labels[: batch.batch_size]
        ele = self.verifier.backward_ele(
            nodes, node_labels, self.num_classes, batch, batch.x.device
        )

        results = {}
        for node in nodes:
            node_id = batch.node_id[node].item()
            results[node_id] = self.verify_node(node, ele, batch)

        return results

    def verify_node(
        self, node: int, ele: NodeDeeppolyEle, data: Data
    ) -> Dict[Tuple[int, int], bool]:
        """
        Calculate the perturbation matrix for a node

        Args:
            node (int): node id
            ele (NodeDeeppolyEle): Polyhedra bounding for the node
            data (Data): graph data

        Returns:
            Dict[Tuple[int, int], bool]: a dictionary of (feature addition,
                feature deletion) to if the node is robust
        """
        node_lb_expression = ele.lower_coef[node]
        node_lb_constant = ele.lower_const[node]

        num_expressions = node_lb_expression.shape[0]
        result = {}
        for i in range(num_expressions):
            expression = node_lb_expression[i]
            constant = node_lb_constant[i]
            result = update_dict(
                result,
                self.verify_lower_bound(expression, constant.sum(), data.x),
            )
        return result

    def verify_lower_bound(
        self, expression: torch.Tensor, constant: torch.Tensor, x: torch.Tensor
    ) -> Dict[Tuple[int, int], bool]:
        """
        Verify one lower bound expression for a node

        Args:
            expression (torch.Tensor): the expression of the node
            constant (torch.Tensor): the constants of the node
            x (torch.Tensor): the node features

        Returns:
            Dict[Tuple[int, int], bool]: a dictionary of (feature addition,
                feature deletion) to if the node is robust
        """
        center = ((expression * x).sum() + constant).item()

        addition_change, removal_change = get_feature_changes(
            expression, constant, x
        )

        # sort the feature addition and removal from most negative
        addition_change, _ = torch.sort(addition_change)
        removal_change, _ = torch.sort(removal_change)

        result = {}
        for feature_addition in range(self.global_budget + 1):
            feature_removal = self.global_budget - feature_addition
            if feature_addition > self.budget.feature_addition:
                continue

            if feature_removal > self.budget.feature_removal:
                continue

            # perturb the node for the specified feature addition and removal
            feature_addition_change = addition_change[:feature_addition]
            feature_removal_change = removal_change[:feature_removal]

            # only consider the negative impact as we want to make the lower
            # bound smaller
            feature_addition_change = (
                (feature_addition_change * (feature_addition_change < 0))
                .sum()
                .item()
            )
            feature_removal_change = (
                (feature_removal_change * (feature_removal_change < 0))
                .sum()
                .item()
            )

            # calculate if the node is robust
            is_robust = (
                center + feature_addition_change + feature_removal_change > 0
            )

            result[(feature_addition, feature_removal)] = is_robust

        return result


def get_feature_changes(
    expression: torch.Tensor, constants: torch.Tensor, x: torch.Tensor
):
    """
    Get the feature changes for a node

    Args:
        expression (torch.Tensor): the expression of the node
        constants (torch.Tensor): the constants of the node
        x (torch.Tensor): the node features

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the feature addition and removal
    """
    # get the perturbation matrix
    perturb = (1 - x) + (-1 * x)

    # get the feature addition and removal
    change = perturb * expression

    addition_change = (change * (1 - x)).flatten()
    removal_change = (change * x).flatten()

    return addition_change, removal_change


def update_dict(a: Dict[Any, bool], b: Dict[Any, bool]) -> Dict[Any, bool]:
    """
    Update a dictionary with another dictionary, if a key
    in b does not exist in a, add it to a, otherwise use
    the result of "boolean and" as the value
    """
    for key, value in b.items():
        if key not in a:
            a[key] = value
        else:
            a[key] = a[key] and value

    return a
