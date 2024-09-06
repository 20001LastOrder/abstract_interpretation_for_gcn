import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import degree

from utils import datasets

pubmed_ids = [
    11020,
    11021,
    11022,
    11023,
    11024,
    11025,
    11026,
    11027,
    11447,
    11448,
    11449,
    11450,
    11451,
    11452,
    11453,
    11454,
    11894,
    11895,
    11896,
    11897,
    11898,
    11899,
    11900,
    11902,
    18738,
    18739,
    18740,
    18741,
    18742,
    18743,
    18744,
    18745,
]


class NodeClassificationDataset(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_folder = config.data_folder
        self.dataset = config.dataset
        (A, X, z, self.N, D, edge_index) = datasets.get_dataset(
            f"{self.data_folder}/", self.dataset
        )
        self.A = A
        self.node_weights = degree(edge_index[1]).square()
        edge_index, edge_attrs = gcn_norm(edge_index)
        data = Data(
            X,
            edge_index,
            edge_attrs,
            z,
            node_id=torch.LongTensor(range(X.shape[0])),
        )
        transform = RandomNodeSplit(num_val=0, num_test=config.test_ratio)

        self.data = transform(data)

        # the optimization approach will fail on these indexes,
        # so take them out
        if self.dataset == "pubmed":
            mask = torch.ones_like(data.node_id)
            mask[pubmed_ids] = 0
            self.data.train_mask = (self.data.train_mask * mask) > 0
            self.data.test_mask = (self.data.test_mask * mask) > 0
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.batch_size = config.batch_size
        self.steps = config.steps
        self.sampling = config.sampling
        self.loss = config.loss

    def sample_nodes(self, mask):
        idx = torch.nonzero(mask)
        nodes = idx.flatten().cpu().numpy()
        # weights = self.node_weights[nodes].flatten().cpu().numpy()
        # weights = weights / np.sum(weights)
        sampled_nodes = np.random.choice(
            nodes, self.steps * self.batch_size, replace=True
        )
        return torch.LongTensor(sampled_nodes)

    def sample_nodes_unlabelled(self, train_mask, test_mask):
        train_nodes = torch.nonzero(train_mask).flatten().cpu().numpy()
        test_nodes = torch.nonzero(test_mask).flatten().cpu().numpy()

        sampled_nodes = []
        for _ in range(self.steps // 3):
            sampled_train1 = np.random.choice(
                train_nodes, self.batch_size, replace=False
            )
            sampled_train2 = np.random.choice(
                train_nodes, self.batch_size, replace=False
            )
            sampled_test = np.random.choice(
                test_nodes, self.batch_size, replace=False
            )

            batch = np.concatenate(
                [sampled_train1, sampled_train2, sampled_test]
            )
            np.random.shuffle(batch)
            sampled_nodes.extend(batch)

        return torch.LongTensor(sampled_nodes)

    def get_size(self):
        return self.N

    def train_dataloader(self):
        nodes = self.data.train_mask

        if self.sampling:
            if "U" in self.loss:
                # use unlabelled nodes for robust training
                nodes = self.sample_nodes_unlabelled(
                    self.data.train_mask, self.data.test_mask
                )
            else:
                nodes = self.sample_nodes(nodes)
        return self.get_dataloader(nodes)

    def val_dataloader(self):
        return self.get_dataloader(self.data.test_mask)

    def test_dataloader(self):
        # when test use the whole dataset
        return self.get_dataloader(self.data.test_mask + self.data.train_mask)

    def get_dataloader(self, nodes):
        return NeighborLoader(
            self.data,
            num_neighbors=[
                -1,
                -1,
            ],  # use hops equal to the node size neighbors
            batch_size=self.batch_size,
            directed=False,
            input_nodes=nodes,
            num_workers=self.num_workers,
        )
