from abstract_interpretation.robust_training.dataset import NodeClassificationDataset
from abstract_interpretation.robust_training.model import GCNModel

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from jsonargparse import ArgumentParser
import torch
from abstract_interpretation.ConcreteGraphDomains import BinaryNodeFeatureDomain
from utils.configs import get_config_file_parser, get_traing_config_parser


def main(args):
    if args.seed is not None:
        pl.seed_everything(args.seed)
    
    dataset = NodeClassificationDataset(args)
    model = GCNModel(args)
    concrete_domain = BinaryNodeFeatureDomain(dataset.data.edge_index.cuda(), args.p,
                                             dataset.data.x.cuda(), l= args.l)
    
    if args.method == 'poly':
        model.setup_verifier(concrete_domain)
    else:
        model.configure_optimization_verifier(dataset.A, concrete_domain)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=args.run_dir, filename=args.model_name)
    # wandb_logger = pl_loggers.WandbLogger(project=args.project_name)

    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        # logger=wandb_logger,
        gpus=-1,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        check_val_every_n_epoch=args.check_val_every_n_epoch if not args.sampling else 10,
        max_epochs=args.max_epochs if not args.sampling else 1, # when sampling, we only train for 1 epoch with many steps
        callbacks=[checkpoint_callback],
        precision=args.precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.fit(model=model, datamodule=dataset)
    return trainer.test(ckpt_path='last',dataloaders = dataset.test_dataloader())
    
    # model = model.load_from_checkpoint(args.resume_from_checkpoint, config=args)
    # model.setup_verifier(concrete_domain)
    # trainer.test(model=model, dataloaders = dataset.test_dataloader())


if __name__ == "__main__":
    # This setting makes the training faster for latest version of pytorch
    torch.set_float32_matmul_precision('medium')

    file_parser = get_config_file_parser()
    train_parser = get_traing_config_parser()
    
    config_path = file_parser.parse_args().config

    cfg = train_parser.parse_path(config_path)
    main(cfg)
