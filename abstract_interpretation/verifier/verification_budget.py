class VerificationBudget:
    def __init__(self, num_candidates: int):
        """
        Args:
            num_candidates (int): number to check for each difference of the
                node
        """
        self.num_candidates = num_candidates


class VerificationPerturbBudget:
    def __init__(self, feature_addition: int, feature_removal: int) -> None:
        self.feature_addition = feature_addition
        self.feature_removal = feature_removal
