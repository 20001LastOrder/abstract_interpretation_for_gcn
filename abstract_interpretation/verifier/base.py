from abc import ABC, abstractmethod

from torch_geometric.data import Data

from abstract_interpretation.node_deeppoly import NodeDeeppolyEle


class BaseVerifier(ABC):
    @abstractmethod
    def verify_batch(self, batch: Data):
        pass

    @abstractmethod
    def verify_node(self, node: int, ele: NodeDeeppolyEle, data: Data):
        pass
