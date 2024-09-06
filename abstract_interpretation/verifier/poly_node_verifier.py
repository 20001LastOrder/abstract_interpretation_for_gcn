"""
Perform verification for node features using polyhedra abstract interpretation.
"""

import copy
import logging
from typing import Callable, Iterator, List, Tuple

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from abstract_interpretation.ConcreteGraphDomains import (
    BinaryNodeFeatureDomain,
)
from abstract_interpretation.node_deeppoly import (
    NodeDeeppolyEle,
    NodeDeeppolyVerifier,
)
from abstract_interpretation.utils import safe_topk
from abstract_interpretation.verifier.base import BaseVerifier
from abstract_interpretation.verifier.verification_budget import (
    VerificationBudget,
)

logger = logging.getLogger(__name__)


class PolyNodeVerifier(BaseVerifier):
    def __init__(
        self,
        gcn: torch.nn.Module,
        data: Data,
        global_perturb: int,
        local_perturb: int,
        budget: VerificationBudget,
        approx: str = "topk",
        adversarial_callback: Callable = None,
        count_adversarial: bool = False,
    ):
        """
        Args:
            gcn: the GCN model to be verified
            global_perturb: the global budget for the perturbation
            local_perturb: the budget for the perturbation for each node
            budget: the budget for the verification for each node
            approax: the approaximation method for the interval abstract
                interpretation either "topk" or "max"
        """
        self.gcn = gcn
        self.global_perturb = global_perturb
        self.local_perturb = local_perturb
        self.budget = budget
        self.approx = approx

        self.layers = list(gcn.children())
        concrete_domain = BinaryNodeFeatureDomain(
            data.edge_index, global_perturb, data.x, l=local_perturb
        )
        self.verifier = NodeDeeppolyVerifier(
            self.layers,
            concrete_domain,
            approx=approx,
            edge_weights=data.edge_attr,
        )
        self.num_classes = data.y.max().item() + 1
        self.num_nodes = data.x.shape[0]
        self.adverssarial_callback = adversarial_callback
        self.count_adversarial = count_adversarial

    def verify_batch(self, batch: Data) -> List[int]:
        """
        Verify a subgraph represented as a batch data
        Args:
            batch: the batch of graphs to be verified
        """
        nodes = list(range(batch.batch_size))
        node_labels = batch.labels[: batch.batch_size]
        ele = self.verifier.backward_ele(
            nodes, node_labels, self.num_classes, batch, batch.x.device
        )

        results = []
        for node in nodes:
            results.append(self.verify_node(node, ele, batch))

        return results

    def verify_node(self, node: int, ele: NodeDeeppolyEle, data: Data) -> int:
        """
        Verify a single node
        Args:
            node: the node to be verified
            ele: the abstract element for the subgraph containing the node
            data: the data of the graph
        Return:
            1 if the node is verified, -1 if the node is not verified,
            0 if uncertain
        """
        orignal_class = data.labels[node]

        node_lb_expression = ele.lower_coef[node]
        node_lb_constant = ele.lower_const[node]
        node_ub_expression = ele.upper_coef[node]
        node_ub_constant = ele.upper_const[node]

        count = 0
        certification_results = []
        for i in range(self.num_classes):
            if i == orignal_class:
                continue

            ub_result = self.verify_ub_expression(
                node,
                node_ub_expression[count],
                node_ub_constant[count].sum().item(),
                data,
            )

            if ub_result:
                logger.info(
                    f"Node {data.node_id[node]} is uncertified by upper bound"
                )
                return -1

            result = self.verify_lb_expression(
                node,
                node_lb_expression[count],
                node_lb_constant[count].sum().item(),
                i,
                data,
            )
            count += 1

            if result == -1:
                return result
            certification_results.append(result)

        for result in certification_results:
            if result == 0:
                # at least diff for one class is uncertain
                return 0

        return 1

    def verify_ub_expression(
        self,
        node: int,
        expression: torch.FloatTensor,
        constant: float,
        data: Data,
    ) -> bool:
        """
        Verify a single node with given upper bound expression to certify the
          appearance of adversarial examples
        Args:
            node: the node to be verified
            expression: the upper bound expression of shape
                (num_nodes * num_classes)
            constant: the constant of the upper bound expression
            data: the data of the graph
        """
        perturb = (1 - data.x) + (-1 * data.x)
        center = ((expression * data.x).sum() + constant).item()
        change = expression * perturb

        change = safe_topk(change, self.local_perturb, dim=1, largest=False)[
            0
        ].flatten()

        top_change = safe_topk(
            change, self.global_perturb, dim=0, largest=False
        )[0]
        top_change = top_change[top_change < 0]

        result = top_change.sum() + center
        return result <= 0

        # use these to get the actual features changed
        # sort the feature changes from the most negative ones
        # sorted_features = torch.argsort(change)
        # local_budgets = [self.local_perturb for _ in range(num_nodes)]

        # result = identify_adversairal_ub(
        #     sorted_features,
        #     change,
        #     local_budgets,
        #     self.global_perturb,
        #     num_features,
        #     center,
        # )

        # if len(result) > 0:
        #     # an adversarial identified
        #     return True

        # return False

    def verify_lb_expression(
        self,
        node: int,
        expression: torch.FloatTensor,
        constant: float,
        target_class: int,
        data: Data,
    ) -> int:
        """
        Verify a single node with a given lower bound expression
            Args:
                node: the node to be verified
                expression: the lower bound expression of shape
                    (num_nodes * num_classes)
                target_class: the target class to be verified
                data: the data of the graph
            Return:
                1 if the node is verified, -1 if the node is not verified,
                0 if uncertain
        """
        perturb = (1 - data.x) + (-1 * data.x)

        center = ((expression * data.x).sum() + constant).item()
        num_nodes = data.x.shape[0]

        change = (expression * perturb).flatten()

        # sort the feature changes from the most negative ones
        sorted_features = torch.argsort(change)
        local_budgets = [self.local_perturb for _ in range(num_nodes)]
        num_features = data.x.shape[1]

        # sort = torch.sort(change)[0]
        # logger.info(sort[20:40].sum())
        # logger.info(center)

        candidate_generator = generate_adversarial_candidates(
            sorted_features,
            change,
            local_budgets,
            self.global_perturb,
            num_features,
            center,
            current_features=[[], []],
        )

        adversarial_count = 0
        budget_exceed = False

        for candidate in candidate_generator:
            adversarial_count += 1

            label_changed = self.check_label_change(node, candidate, data)

            if label_changed:
                # found an actual adversarial example
                logger.info(
                    f"Node {data.node_id[node]} has been verified as "
                    "non-robust using an adversairal example"
                )
                return -1

            if adversarial_count > self.budget.num_candidates:
                budget_exceed = True
                break

        if adversarial_count > 0 and self.count_adversarial:
            # count the exact number of adversarial examples
            logging.warning("Counting adversarial examples")
            adversarial_count = 0
            for _ in tqdm(
                count_non_leaf_candidates_generator(
                    sorted_features,
                    change,
                    local_budgets,
                    self.global_perturb,
                    num_features,
                    center,
                )
            ):
                adversarial_count += 1

            if self.adverssarial_callback is not None:
                self.adverssarial_callback(
                    node, target_class, adversarial_count
                )

            logger.warning(
                f"Node {data.node_id[node]} has {adversarial_count} "
                f"candidates for class {target_class}"
            )

        if budget_exceed:
            return 0
        else:
            return 1

    @torch.no_grad()
    def check_label_change(
        self, node: int, perturb_features: torch.LongTensor, data: Data
    ) -> bool:
        """
        Check the actual effect of the perturbation
        Args:
            node: the node to be verified
            perturb_features: the features to be perturbed
            data: the data of the graph
        Return:
            True if the label is changed, False otherwise
        """
        x = data.x.clone().detach()
        x[perturb_features[0], perturb_features[1]] = (
            1 - x[perturb_features[0], perturb_features[1]]
        )

        logits = self.gcn(x, data.edge_index, data.edge_attr)
        node_label = logits[node].argmax().item()

        return node_label != data.labels[node].item()


def identify_adversairal_ub(
    sorted_features: torch.Tensor,
    feature_values: torch.Tensor,
    local_budgets: List[int],
    global_budget: int,
    num_features: int,
    value,
) -> List[Tuple[int, int]]:
    """
    Identify the certified adversarial example using the upper bound

    Args:
        sorted_features: the sorted node features from most negative to most
            positive impact to the value
        feature_values: the values of the features
        local_budgets: the local budgets for each node
        global_budget: the global budget
        value: the current value
    """
    features = []

    for change_id in sorted_features:
        if global_budget == 0:
            break

        node = int(change_id / num_features)
        feature = change_id % num_features

        if local_budgets[node] == 0:
            continue

        global_budget -= 1
        local_budgets[node] -= 1

        features.append((node, feature))
        value = value + feature_values[change_id]

    logger.info(f"final value {value}")
    if value <= 0:
        return features
    else:
        return []


def count_adversarial_candidates(
    sorted_features: torch.Tensor,
    feature_values: torch.Tensor,
    local_budgets: List[int],
    global_budget: int,
    num_features: float,
    value: float,
    current_id: int = 0,
) -> int:
    """
    Count the all combinations making the value to be smaller than zero

    Args:
        sorted_features: the sorted node features from most negative to most
            positive impact to the value
        feature_values: the values of the features
        local_budgets: the local budgets for each node
        global_budget: the global budget
        num_features: the number of features for each node
        value: the current value
        current_id: the current feature change id
    Return:
        the number of combinations
    """
    if global_budget == 0:
        return 0

    n = len(sorted_features)
    count = 0
    for i in range(current_id, n):
        # print(f"current id {i}")
        change_id = sorted_features[i]
        node_id = (change_id / num_features).floor().int().item()

        if local_budgets[node_id] == 0:
            continue

        local_budgets[node_id] -= 1

        new_value = value + feature_values[change_id].item()

        if new_value <= 0:
            # add count for the current combination
            count += 1

        count_for_feature = count_adversarial_candidates(
            sorted_features,
            feature_values,
            local_budgets,
            global_budget - 1,
            num_features,
            new_value,
            current_id=i + 1,
        )
        count += count_for_feature
        local_budgets[node_id] += 1

        if count_for_feature == 0:
            # since the feature is ordered, no need to search further
            break

    return count


def count_non_leaf_candidates_generator(
    sorted_features: torch.Tensor,
    feature_values: torch.Tensor,
    local_budgets: List[int],
    global_budget: int,
    num_features: float,
    value: float,
    current_id: int = 0,
    current_level: int = 0,
) -> Iterator[int]:
    """
    Count the all combinations making the value to be smaller than zero

    Args:
        sorted_features: the sorted node features from most negative to most
            positive impact to the value
        feature_values: the values of the features
        local_budgets: the local budgets for each node
        global_budget: the global budget
        num_features: the number of features for each node
        value: the current value
        current_id: the current feature change id
    Return:
        the number of combinations
    """
    if global_budget == 0:
        return

    n = len(sorted_features)
    for i in range(current_id, n):
        # print(f"current id {i}")
        change_id = sorted_features[i]
        node_id = (change_id / num_features).floor().int().item()

        if local_budgets[node_id] == 0:
            continue

        local_budgets[node_id] -= 1

        new_value = value + feature_values[change_id].item()

        if new_value <= 0:
            # add count for the current combination
            yield 1
            break

        count_for_feature = 0
        for _ in count_non_leaf_candidates_generator(
            sorted_features,
            feature_values,
            local_budgets,
            global_budget - 1,
            num_features,
            new_value,
            current_id=i + 1,
            current_level=current_level + 1,
        ):
            yield 1
            count_for_feature += 1

        local_budgets[node_id] += 1

        if count_for_feature == 0 or current_level > 3:
            # since the feature is ordered, no need to search further
            break
    # return count


def generate_adversarial_candidates(
    sorted_features: torch.Tensor,
    feature_values: torch.Tensor,
    local_budgets: List[int],
    global_budget: int,
    num_features: float,
    value: float,
    current_features: List[List[int]],
    current_id: int = 0,
) -> Iterator[List[List[int]]]:
    """
    Generate all combinations making the value to be smaller than zero

    Args:
        sorted_features: the sorted node features from most negative to most
            positive impact to the value
        feature_values: the values of the features
        local_budgets: the local budgets for each node
        global_budget: the global budget
        current_features: the current features changed
        num_features: the number of features for each node
        current_id: the current feature change id
        value: the current value

    """
    if value <= 0:
        yield copy.deepcopy(current_features)

    if global_budget == 0:
        return

    n = len(sorted_features)
    for i in range(current_id, n):
        change_id = sorted_features[i]
        node_id = (change_id / num_features).floor().int().item()
        feature_id = (change_id % num_features).int().item()

        if local_budgets[node_id] == 0:
            continue

        local_budgets[node_id] -= 1

        # append the current feature change
        current_features[0].append(node_id)
        current_features[1].append(feature_id)

        new_value = value + feature_values[change_id].item()
        count = 0
        for result in generate_adversarial_candidates(
            sorted_features,
            feature_values,
            local_budgets,
            global_budget - 1,
            num_features,
            new_value,
            current_features=current_features,
            current_id=i + 1,
        ):
            yield result
            count += 1

        local_budgets[node_id] += 1
        current_features[0].pop()
        current_features[1].pop()

        # Since the features are sorted from most negative to most positive
        # impact, if no change can make the value to be smaller than zero,
        # then we can stop the search
        if count == 0:
            return
