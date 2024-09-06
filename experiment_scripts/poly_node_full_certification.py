from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from abstract_interpretation.neighbor_sampler import get_neighbor_loader
from abstract_interpretation.verifier.poly_node_full_verifier import \
    PolyNodeFullVerifier
from abstract_interpretation.verifier.verification_budget import \
    VerificationPerturbBudget
from utils.datasets import prepare_model_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = "citeseer"
robust_gcn = False


def full_verification():
    gcn, data = prepare_model_dataset(dataset_name, robust_gcn=robust_gcn)

    loader = get_neighbor_loader(data, [-1, -1], 8, directed=False)
    gcn = gcn.to(device)
    budget = VerificationPerturbBudget(0, 200)
    cuda_data = data.to(device)

    num_nodes = data.x.shape[0]
    global_perturb_max = budget.feature_addition + budget.feature_removal

    result = np.zeros(
        (num_nodes, budget.feature_addition + 1, budget.feature_removal + 1),
        dtype=bool,
    )
    for gloabl_budget in range(global_perturb_max + 1):
        verifier = PolyNodeFullVerifier(
            gcn,
            cuda_data,
            budget,
            gloabl_budget,
        )

        budget_result = {}
        for batch in tqdm(loader):
            batch = batch.to(device)
            budget_result.update(verifier.verify_batch(batch))

        comb_result = defaultdict(lambda: 0)
        for node in budget_result:
            for comb in budget_result[node]:
                result[node, comb[0], comb[1]] = budget_result[node][comb]

                # Add the node robust result
                comb_result[comb] += budget_result[node][comb]

        # Check if there is a combination causing all nodes to be non-robust
        break_loop = False
        for comb in comb_result:
            robustness = comb_result[comb] / num_nodes
            print(f"Combination: {comb}, robustness: {robustness}")
            if robustness == 0:
                print(f"Combination {comb} is non-robust, finish!")
                break_loop = True

        if break_loop:
            break

    # save the result
    np.save(f"{dataset_name}_poly_attr_del.npy", result)


if __name__ == "__main__":
    full_verification()
    print("Done!")
