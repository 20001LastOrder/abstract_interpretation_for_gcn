from abc import abstractmethod


class AbstractElement:
    @abstractmethod
    def ub(self):
        pass

    @abstractmethod
    def lb(self):
        pass


class AbstractDomain:
    @abstractmethod
    def to_abstract(self, concrete_domain) -> AbstractElement:
        pass

    @abstractmethod
    def transform(self, ele: AbstractElement, layers):
        pass

    @abstractmethod
    def transform_single_layer(self, ele: AbstractElement, layer):
        pass

    @abstractmethod
    def to_concrete(self, ele: AbstractElement):
        pass

