from abstract_interpretation.robust_training.dataset import NodeClassificationDataset
from abstract_interpretation.robust_training.model import GCNModel

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from jsonargparse import ArgumentParser
import torch
from abstract_interpretation.ConcreteGraphDomains import BinaryNodeFeatureDomain
from utils.configs import get_config_file_parser, get_traing_config_parser
from train import main
import numpy as np
import pandas as pd

methods = ['poly', 'optim']
seeds = [61253, 3407, 20230215, 123, 223429]

datasets = ['citeseer', 'pubmed', 'cora_ml']

if __name__ == "__main__":
    # This setting makes the training faster for latest version of pytorch
    torch.set_float32_matmul_precision('medium')

    file_parser = get_config_file_parser()
    train_parser = get_traing_config_parser()
    
    config_path = file_parser.parse_args().config

    cfg = train_parser.parse_path(config_path)
    for dataset in datasets:
        for method in methods:
            print(f"Training {dataset} with {method} method with {cfg.loss} loss")
            results = []
            for seed in seeds:
                cfg.dataset = dataset
                cfg.method = method
                cfg.seed = seed
                test_result = main(cfg)
                results.append(test_result[-1])
                
                # save for every run to avoid data loss
                df = pd.DataFrame(results)
                df.to_csv(f'./results/training_{dataset}_{method}_{cfg.loss}.csv', index=False)
