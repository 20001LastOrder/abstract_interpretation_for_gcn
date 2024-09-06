"""
This file contains the implementation of enumerating all possibilities of edge
perturbations for a given graph on a node and then checking the robustness of
the graph against these perturbations.
"""

import math
from collections import defaultdict
from collections.abc import Iterator
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.sampler.utils import to_csc
from tqdm import tqdm

from abstract_interpretation.neighbor_sampler import (
    NeighborSampler,
    NodeSamplerInput,
)
from utils import datasets, models
from utils.datasets import cal_node_degree, gcn_normalize

dataset = "citeseer"
robust_gcn = False
device = "cuda"
batch_based = True
local_budget = 3
global_budget = 5
sort_candidate_edges = True


def edge_set_to_tensor(
    edge_set: Set[Tuple[int, int]], device: str = "cpu"
) -> torch.LongTensor:
    """
    Convert a set of edges to a tensor.
    """
    result = [[], []]
    for source, target in edge_set:
        result[0].append(source)
        result[1].append(target)
        if source != target:
            result[0].append(target)
            result[1].append(source)
    result = torch.tensor(result, device=device)
    return result.long()


def recalculate_degree(
    removed_edges: Set[Tuple[int, int]],
    node_degree: torch.Tensor,
    device: str = "cpu",
) -> Union[torch.Tensor, int]:
    """
    Recalculate the degree of the nodes after removing the given edges.
    """
    node_degree = node_degree.tolist()
    for source, target in removed_edges:
        node_degree[source] -= 1
        if target != source:
            node_degree[target] -= 1

        if node_degree[source] <= 1 or node_degree[target] <= 1:
            # the graph is disconnected (1 is for the self loop)
            return -1

    return torch.tensor(node_degree, device=device)


def filter_self_loops(
    edge_set: Iterable[Tuple[int, int]]
) -> Set[Tuple[int, int]]:
    """
    Filter out self loops from the given edge set.
    """
    result = []
    for source, target in edge_set:
        if source != target:
            result.append((source, target))
    return set(result)


def generate_possible_perturbation(
    edge_set: Iterable[Tuple[int, int]],
    local_budget: int,
    global_budget: int,
) -> Iterable[Iterable[Tuple[int, int]]]:
    """
    Generate all possible perturbations for the given edge set, respecting
    the local and global budget
    """
    edges = list(edge_set)
    comb = combinations(edges, global_budget)
    possible_perturbations = []
    for c in comb:
        node_change = defaultdict(lambda: 0)
        valid = True
        for edge in c:
            node_change[edge[0]] += 1
            node_change[edge[1]] += 1
            if (
                node_change[edge[0]] > local_budget
                or node_change[edge[1]] > local_budget
            ):
                valid = False
                break
        if valid:
            possible_perturbations.append(c)
    return possible_perturbations


def all_possible_perturbations(
    edges: List[Tuple[int, int]],
    node_change: List[int],
    local_budget: int,
    global_budget: int,
    n: int = 0,
) -> Iterator[List[Tuple[int, int]]]:
    """
    Generate all possible perturbations for the given edge set, respecting
    the local and global budget as an iterator
    """
    for i in range(n, len(edges) - global_budget + 1):
        current_edge = edges[i]
        source, target = current_edge

        # check if the edge is valid
        node_change[source] += 1
        node_change[target] += 1

        if (
            node_change[source] <= local_budget
            and node_change[target] <= local_budget
        ):
            if global_budget == 1:
                yield [current_edge]
            else:
                for perturbation in all_possible_perturbations(
                    edges, node_change, local_budget, global_budget - 1, i + 1
                ):
                    yield [current_edge] + perturbation

        # backtrack
        node_change[source] -= 1
        node_change[target] -= 1


def sort_edges(edges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Sort the edges in the given edge set.
    """
    return list(sorted(edges, key=lambda x: (x[0], x[1])))


class ExactEdgePerturbation:
    def __init__(
        self,
        gcn: Module,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        z: torch.Tensor,
        eval_nodes: torch.Tensor,
        node_degree: torch.Tensor,
        local_budget: int,
        global_budget: int,
        device: str = "cpu",
        sort_edges: bool = True,
    ):
        self.gcn = gcn
        self.X = X
        self.edge_index = edge_index
        self.edge_attrs = edge_attrs
        self.z = z
        self.eval_nodes = eval_nodes
        self.node_degree = node_degree
        self.local_budget = local_budget
        self.global_budget = global_budget
        self.device = device
        self.sort_edges = sort_edges
        self.adversarial_map = defaultdict(lambda: [])

    def get_perturbation_result(
        self,
        gcn: Module,
        batch: Data,
        target_node: int,
        edge_set: Set[Tuple[int, int]],
        edge_to_remove: Set[Tuple[int, int]],
    ) -> int:
        """
        Get the result of perturbing the given edge set.
        return: label of the node after perturbation
        """
        edge_index = batch.edge_index
        node_degree = batch.node_degree
        x = batch.x

        # remove the edges
        edge_index = edge_set_to_tensor(edge_set - edge_to_remove, self.device)
        node_degree = recalculate_degree(
            edge_to_remove, node_degree, self.device
        )

        if isinstance(node_degree, int) and node_degree == -1:
            # the graph is disconnected
            return -1

        # normalize the adjacency matrix
        edge_attr = gcn_normalize(edge_index, deg=node_degree)

        # run the model
        return gcn(x, edge_index, edge_attr)[target_node, :].argmax().item()

    def edge_perturbation(
        self, gcn: Module, batch: Data, target_node: int
    ) -> Union[bool, int]:
        """
        Enumerate all possible edge perturbations for a given node,
        assuming the graph is undirected.
        """
        edge_index = batch.edge_index
        original_label = (
            gcn(batch.x, edge_index, batch.edge_attr)[target_node, :]
            .argmax()
            .item()
        )

        edge_set = construct_edge_set(edge_index)
        valid_edge_set = filter_self_loops(edge_set)
        valid_edge_set = sort_edges(valid_edge_set)

        if not self.sort_edges:
            np.random.shuffle(valid_edge_set)

        all_possibility = math.comb(len(valid_edge_set), self.global_budget)
        if all_possibility > 100000:
            # too many possible perturbations, ignore such case
            return -1

        num_nodes = len(batch.node_id)
        for edges_to_remove in all_possible_perturbations(
            valid_edge_set,
            [0 for i in range(num_nodes)],
            self.local_budget,
            self.global_budget,
        ):
            edges_to_remove = set(edges_to_remove)

            perturbed_label = self.get_perturbation_result(
                gcn, batch, target_node, edge_set, edges_to_remove
            )

            if isinstance(perturbed_label, int) and perturbed_label == -1:
                # the graph is disconnected, ignore such case
                continue

            if perturbed_label != original_label:
                node_idx = batch.node_id[target_node].item()
                self.adversarial_map[node_idx].append(edges_to_remove)
                return False
        return True

    def certify_batch_based(self) -> Dict[int, bool]:
        """
        Certify the robustness of a graph batch based.
        """
        data = Data(
            self.X,
            self.edge_index,
            self.edge_attrs,
            self.z,
            node_id=torch.LongTensor(range(self.X.shape[0])),
            node_degree=self.node_degree,
        )

        # use two hop neighbors for GCNs with two layers
        sampler = NeighborSampler(data, num_neighbors=[-1, -1], directed=False)

        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[-1, -1],
            batch_size=1,
            directed=False,
            neighbor_sampler=sampler,
            input_nodes=self.eval_nodes,
        )

        gcn = self.gcn.to(device)
        result_map = {}
        errors = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            result = self.edge_perturbation(gcn, batch, 0)
            if result == -1:
                # too many possible perturbations, ignore such case
                errors.append(batch.node_id[0].item())
            result_map[batch.node_id[0].item()] = result
        return result_map, errors

    def certify_full_graph(self) -> Dict[int, bool]:
        """
        Certify the robustness of a graph entirely
        """
        original_results = self.gcn(self.X, self.edge_index, self.edge_attrs)
        original_results = original_results.to(self.device)
        gcn = self.gcn.to(self.device)

        data = Data(
            self.X,
            self.edge_index,
            self.edge_attrs,
            self.z,
            node_id=torch.LongTensor(range(self.X.shape[0])),
            node_degree=self.node_degree,
        )
        data = data.to(self.device)

        sampler = NeighborSampler(data, num_neighbors=[-1, -1], directed=False)
        input_data = NodeSamplerInput(input_id=None, node=self.eval_nodes)
        edge_set = construct_edge_set(self.edge_index)

        result = {}

        for i in tqdm(range(len(eval_nodes))):
            target_node = input_data[i]
            local_graph = sampler.sample_from_nodes(target_node)
            subgraph_edges = sampler.edge_permutation[local_graph.edge]

            original_label = (
                original_results[target_node.node, :].argmax().item()
            )
            for edge in subgraph_edges:
                source, target = edge_index[:, edge].tolist()
                if source >= target:
                    # the graph is undirected, and we do not consider
                    # self-loops this order is consistent with how the edge
                    # set is built
                    continue
                edge = {(source, target)}
                new_label = self.get_perturbation_result(
                    gcn,
                    data,
                    target_node.node,
                    edge_set,
                    edge,
                )

                if new_label == -1:
                    # the graph is disconnected
                    continue

                if new_label != original_label:
                    result[target_node.node] = False
                    break

            if target_node.node not in result:
                # the node is robust
                result[target_node.node] = True

        return result


def construct_edge_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    """
    Construct a set of edges from the edge_index tensor,
    ignoring the direction of the edges.
    """
    result = set()
    for i in range(edge_index.shape[1]):
        source, target = edge_index[:, i].tolist()
        if source <= target:
            result.add((source, target))
    return result


if __name__ == "__main__":
    (A, X, z, N, D, edge_index) = datasets.get_dataset(
        "./robust-gcn-structure/datasets/", dataset
    )
    gcn = models.get_model(
        "robust-gcn-structure/pretrained_weights",
        dataset,
        robust_gcn,
        normalize=False,
    )

    # This gives exactly the 500 nodes that we sampled for our experiments.
    np.random.seed(481516)
    eval_nodes = np.random.choice(np.arange(0, A.shape[0]), 500, replace=False)
    eval_nodes = torch.LongTensor(eval_nodes)

    # use gcn norm to add self_loops and normalize the adjacency matrix
    # the gcn_normalization method does not add self_loops
    edge_index, edge_attrs = gcn_norm(edge_index)
    node_degree = cal_node_degree(edge_index, num_nodes=N)

    original_results = gcn(X, edge_index, edge_attrs)

    data = Data(
        X,
        edge_index,
        edge_attrs,
        z,
        node_id=torch.LongTensor(range(X.shape[0])),
        node_degree=node_degree,
    )

    sampler = NeighborSampler(data, num_neighbors=[-1, -1], directed=False)

    loader = NeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[-1, -1],
        batch_size=1,
        directed=False,
        neighbor_sampler=sampler,
    )

    certifier = ExactEdgePerturbation(
        gcn,
        X,
        edge_index,
        edge_attrs,
        z,
        eval_nodes,
        node_degree,
        local_budget,
        global_budget,
        device,
        sort_edges=sort_candidate_edges,
    )

    if batch_based:
        results, errors = certifier.certify_batch_based()
    else:
        results = certifier.certify_full_graph()

    certifications = [results[node] for node in results.keys()]
    # print(results)
    print("Certification rate: {}".format(np.mean(certifications)))
    # print([node for node in results.keys() if not results[node]])

    for node, adversarials in certifier.adversarial_map.items():
        print("Node", node)
        print("Adversarials", adversarials)
        print("Node degree", node_degree[node].int().item())

    print("Number of errors: {}".format(len(errors)))
