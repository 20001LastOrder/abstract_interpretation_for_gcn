from jsonargparse import ArgumentParser
import numpy as np

def get_traing_config_parser():
    """
    Get the training configuration parser
    """
    parser = ArgumentParser()
    # training configuration
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dev', default=False, type=bool)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='checkpoint')
    parser.add_argument('--run_dir', type=str, default='runs')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--p', type=int)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--loss', type=str, default='cross_entropy', 
                        help='the loss function to use', choices=['ce', 'hinge', 'bce', 'adv'])
    parser.add_argument('--steps', type=int, default=1000, help='number of steps to train for when use sampling dataset')

    # dataset configuration
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sampling', type=bool, default=False)

    # model configuration
    parser.add_argument('--model_folder', type=str)
    parser.add_argument('--robust', type=bool, default=False)
    parser.add_argument('--load_weights', type=bool, default=False)
    parser.add_argument('--robust_train', type=bool, default=True)
    parser.add_argument('--margin', type=float, default=np.log(90/10), help='margin for hinge loss')
    parser.add_argument('--margin_u', type=float, default=np.log(60/40), help='margin for hinge loss of unlabelled nodes')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--method', type=str, choices=['poly', 'optim'], default='poly')

    parser.add_argument('--project_name', type=str)


    return parser


def get_config_file_parser():
    """
    Get the parser for the path of the configuration file
    """
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file', default='config.yaml')
    return parser