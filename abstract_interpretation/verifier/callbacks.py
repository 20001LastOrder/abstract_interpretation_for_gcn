from typing import Tuple


class AdversarialCountCallback:
    def __init__(self) -> None:
        self.adversarial_count = []

    def __call__(self, node_id, label, count):
        self.adversarial_count.append((node_id, label, count))

    @property
    def max_adversarial_count(self) -> Tuple[int, int, int]:
        return max(self.adversarial_count, key=lambda x: x[2])
