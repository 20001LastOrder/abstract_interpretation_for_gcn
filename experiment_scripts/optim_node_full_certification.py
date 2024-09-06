import numpy as np
import torch
from tqdm import tqdm

from abstract_interpretation.utils import load_npz
from robust_gcn.robust_gcn.robust_gcn import RobustGCNModel, certify
from utils import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = "pubmed"
robust_gcn = False
mode = None
global_max = 200


def full_verification():
    X, gcn = prepare_opt_gcn(dataset_name, robust_gcn)
    num_nodes = X.shape[0]

    result = np.zeros(
        (num_nodes, 1, global_max + 1),
        dtype=bool,
    )

    for node in range(num_nodes):
        result[node, 0, 0] = True

    for global_budget in tqdm(range(1, global_max + 1)):
        robustness, _ = certify(
            gcn,
            X,
            global_budget,
            optimize_omega=False,
            Q=global_budget,
            batch_size=8,
            certify_nonrobustness=False,
            progress=True,
        )

        for i, node_robust in enumerate(robustness):
            if mode == "add":
                result[i, global_budget, 0] = node_robust
            else:
                result[i, 0, global_budget] = node_robust

        budget_result = robustness.mean()
        print(f"Global budget: {global_budget}, robustness: {budget_result}")

        if budget_result == 0:
            print(f"Global budget {global_budget} is non-robust, finish!")
            break

    # save the result
    if mode is not None:
        np.save(f"./{dataset_name}_optim_attr_{mode}.npy", result)
    else:
        np.save(f"./{dataset_name}_optim_attr.npy", result)


def prepare_opt_gcn(dataset_name, robust=False):
    A, X_sparse, _ = load_npz(
        f"./robust-gcn-structure/datasets/{dataset_name}.npz"
    )
    A = A + A.T
    A[A > 1] = 1
    A.setdiag(0)
    X_sparse = (X_sparse > 0).astype("float32")

    _, D = X_sparse.shape

    (W1, W2), (b1, b2) = models.get_weights(
        "robust-gcn-structure/pretrained_weights", dataset_name, robust
    )
    gcn_model_normal = RobustGCNModel(A, [D, W1.shape[1], W2.shape[1]])
    gcn_model_normal.layers[0].weights = torch.nn.Parameter(
        W1.detach().clone()
    )
    gcn_model_normal.layers[0].bias = torch.nn.Parameter(b1.detach().clone())
    gcn_model_normal.layers[1].weights = torch.nn.Parameter(
        W2.detach().clone()
    )
    gcn_model_normal.layers[1].bias = torch.nn.Parameter(b2.detach().clone())
    gcn_model_normal = gcn_model_normal.cuda()

    return X_sparse, gcn_model_normal


if __name__ == "__main__":
    full_verification()
    print("Done!")
