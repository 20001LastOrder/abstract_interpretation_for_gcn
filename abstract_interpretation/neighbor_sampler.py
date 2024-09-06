from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import (
    BaseSampler,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.utils import to_csc


class NeighborSampler(BaseSampler):
    def __init__(self, data: Data, num_neighbors: List[int], directed: bool):
        self.data = data
        self.num_neighbors = num_neighbors
        self.directed = directed

        self.colptr, self.row, self.perm = to_csc(
            self.data,
            device="cpu",
            share_memory=False,
            is_sorted=False,
            src_node_time=None,
        )

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> SamplerOutput:
        """
        Sample neighbors from the given nodes.
        """
        # get the given node id
        seed = inputs.node

        out = torch.ops.torch_sparse.neighbor_sample(
            self.colptr,
            self.row,
            seed,
            self.num_neighbors,
            False,  # replace
            self.directed,  # directed
        )

        node, row, col, edge = out

        out = SamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=edge,
            batch=None,
            num_sampled_nodes=None,
            num_sampled_edges=None,
        )

        out.metadata = (inputs.input_id, inputs.time)

        return out

    @property
    def edge_permutation(self) -> torch.Tensor:
        return self.perm


def get_neighbor_loader(
    data: Data,
    num_neighbors: List[int],
    batch_size: int,
    directed: bool = False,
    input_nodes: torch.Tensor = None,
) -> NeighborLoader:
    sampler = NeighborSampler(data, num_neighbors, directed)
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        directed=directed,
        neighbor_sampler=sampler,
        input_nodes=input_nodes,
    )
    return loader
