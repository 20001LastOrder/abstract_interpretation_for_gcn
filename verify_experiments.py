from abstract_interpretation.verify import verify_interval, verify_deeppoly
from robust_gcn.robust_gcn.robust_gcn import RobustGCNModel, certify, sparse_tensor
import jsonargparse
from utils import datasets, models
from tqdm import tqdm
from abstract_interpretation.utils import load_npz
import torch
import pandas as pd
import time
import numpy as np
import pandas as pd

perturbations = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

"""
To measure the performance of ours-max, change the parameter value of 'approx' in the function 'verify_deeppoly' in verify.py to 'max'
To measure the performance of optim-orign, comment line 123 of robust_gcn/robust_gcn/robust_gcn.py and uncomment line 124
"""

def main(args):
    (A, X, z, N, D, edge_index) = datasets.get_dataset(args.data_dir, args.dataset)
    gcn = models.get_model(args.model_dir, args.dataset, args.robust, normalize=False)
    gcn = gcn.cuda()
    X = X.cuda()
    edge_index = edge_index.cuda()
    print('local perturbation: ', args.l)
    # first run inverval verification
    interval_results = []
    interval_runtimes = []
    print('running interval verification')
    for p in tqdm(perturbations):
        start = time.time()
        robustness = verify_interval(gcn, X, edge_index, l=args.l, q=p, approx='topk')
        end = time.time()
        interval_runtimes.append(end - start)
        interval_results.append(robustness.mean())
    print(interval_runtimes)
    print(interval_results)
    
    # then run deeppoly verification
    deeppoly_robustness = []
    deeppoly_non_robustness = []
    deeppoly_runtimes = []
    print('running deeppoly verification')
    for p in tqdm(perturbations):
        start = time.time()
        robustness, non_robustness = verify_deeppoly(gcn, X, edge_index, l=args.l, q=p, device='cuda', 
                                                     approx='max', verbose=True, batch_size=8)
        end = time.time()
        deeppoly_runtimes.append(end - start)
        deeppoly_robustness.append(robustness.mean())
        deeppoly_non_robustness.append(1 - non_robustness.mean()) 
    print(deeppoly_runtimes)
    print(deeppoly_robustness)
    print(deeppoly_non_robustness)

    # finally, run optimization-based verification
    opt_robustness = []
    opt_non_robustness = []
    opt_runtimes = []
    X_sparse, gcn_model = prepare_opt_gcn(args, A, D)
    print('running optimization-based verification')
    for p in tqdm(perturbations):
        start = time.time()
        robustness, non_robustness = certify(gcn_model, X_sparse, args.l, optimize_omega=False, Q=p,
                                            batch_size=8, certify_nonrobustness=True, progress=True)
        end = time.time()
        opt_robustness.append(robustness.mean())
        opt_non_robustness.append(1 - non_robustness.mean())
        opt_runtimes.append(end - start)
    print(opt_runtimes)
    print(opt_robustness)
    print(opt_non_robustness)

    results = {
        'perturbations': perturbations,
        'interval_lowerbound': interval_results,
        'interval_runtime': interval_runtimes,
        'deeppoly_max_lowerbound': deeppoly_robustness,
        'deeppoly_max_upperbound': deeppoly_non_robustness,
        'deeppoly_max_runtime': deeppoly_runtimes,
        'opt_origin_lowerbound': opt_robustness,
        'opt_origin_upperbound': opt_non_robustness,
        'opt_origin_runtime': opt_runtimes
    }
    df = pd.DataFrame(results)
    df.to_csv(f'./results/{args.dataset}_robust_{args.robust}.csv', index=False)


def runtime_analysis(args):
    (A, X, z, N, D, edge_index) = datasets.get_dataset(args.data_dir, args.dataset)
    gcn = models.get_model(args.model_dir, args.dataset, args.robust, normalize=False)
    gcn = gcn.cuda()
    X = X.cuda()
    edge_index = edge_index.cuda()
    print('running analysis for', args.method)
    runtimes = []
    for _ in range(args.runs):
        runtime = {}
        for p in tqdm(perturbations):
            if args.method == 'interval':
                start = time.time()
                _ = verify_interval(gcn, X, edge_index, l=args.l, q=p, approx='topk')
                end = time.time()
            if args.method == 'poly':
                start = time.time()
                _ = verify_deeppoly(gcn, X, edge_index, l=args.l, q=p, device='cuda', 
                                    approx='topk', verbose=False, batch_size=8)
                end = time.time()
            if args.method == 'poly_max':
                start = time.time()
                _ = verify_deeppoly(gcn, X, edge_index, l=args.l, q=p, device='cuda', 
                                    approx='max', verbose=False, batch_size=8)
                end = time.time()
            if args.method == 'optim_origin':
                X_sparse, gcn_model = prepare_opt_gcn(args, A, D)
                start = time.time()
                _ = certify(gcn_model, X_sparse, args.l, optimize_omega=False, Q=p,
                            batch_size=8, certify_nonrobustness=True, progress=False)
                end = time.time()
            runtime[p] = end - start
        print(runtime)
        runtimes.append(runtime)

        # save results, do this for each iteration to avoid losing data
        df = pd.DataFrame(runtimes)
        df.to_csv(f'./results/runtime_{args.dataset}_{args.method}.csv', index=False)


def prepare_opt_gcn(args, A, D):
    _, X_sparse, _ = load_npz(f'{args.data_dir}/{args.dataset}.npz')
    X_sparse = (X_sparse>0).astype("float32")

    (W1, W2), (b1, b2) = models.get_weights(args.model_dir, args.dataset, args.robust)
    gcn_model_normal = RobustGCNModel(A, [D, W1.shape[1], W2.shape[1]])
    gcn_model_normal.layers[0].weights = torch.nn.Parameter(W1.detach().clone())
    gcn_model_normal.layers[0].bias = torch.nn.Parameter(b1.detach().clone())
    gcn_model_normal.layers[1].weights = torch.nn.Parameter(W2.detach().clone())
    gcn_model_normal.layers[1].bias = torch.nn.Parameter(b2.detach().clone())
    gcn_model_normal = gcn_model_normal.cuda()

    return X_sparse, gcn_model_normal


if __name__ == '__main__':
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', help='dataset name')
    parser.add_argument('--robust', default=False, action='store_true', help='weather to use robust trained model')
    parser.add_argument('--data_dir', type=str, default='./robust-gcn-structure/datasets/', help='directory of the datasets')
    parser.add_argument('--model_dir', type=str, default='./robust-gcn-structure/pretrained_weights', help='directory of the models')
    parser.add_argument('--method', type=str, default='interval', help='verification method')
    parser.add_argument('--runs', type=int, default=10, help='number of runs')
    parser.add_argument('--l', type=int, default=1, help='perturbation l')
    parser.add_argument('--type', type=str, default='lower')

    config = parser.parse_args()

    if config.dataset == 'citeseer':
        config.l = 37
    elif config.dataset == 'cora_ml':
        config.l = 29
    elif config.dataset == 'pubmed':
        config.l= 5

    if config.type == 'lower':
        main(config)
    elif config.type == 'runtime':
        runtime_analysis(config)
