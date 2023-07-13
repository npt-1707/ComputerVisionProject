from fixmatch import FixMatch
from utils import set_logger, set_seed
import argparse
import logging
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FixMatch Training')
    parser.add_argument("--dataset",
                        type=str,
                        default="cifar10",
                        help="dataset name")
    parser.add_argument("--arch",
                        type=str,
                        default="wide_resnet28_2",
                        help="model architecture")
    parser.add_argument("--pretrained",
                        type=str,
                        default="defined",
                        help="model trained")
    parser.add_argument("--num_labels",
                        type=int,
                        default=250,
                        help="number of labeled data")
    parser.add_argument("--fold", type=int, default=0, help="fold of stl10")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="batch size")
    parser.add_argument("--uratio",
                        type=int,
                        default=7,
                        help="ratio between labeled and unlabeled data")
    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument("--epochs",
                        type=int,
                        default=1024,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.03, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--wd",
                        type=float,
                        default=0.0001,
                        help="weight decay")
    parser.add_argument("--ema_decay",
                        type=float,
                        default=0.999,
                        help="ema decay")
    parser.add_argument("--T", type=float, default=0.5, help="temperature")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.95,
                        help="threshold")
    parser.add_argument("--wu",
                        type=int,
                        default=1,
                        help="coefficient of unlabeled loss")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument("--nesterov",
                        action="store_true",
                        help="use nesterov momentum")
    parser.add_argument("--warmup", type=int, default=5, help="warmup steps")
    parser.add_argument("--root",
                        type=str,
                        default="../data",
                        help="root data directory")
    parser.add_argument("--save", type=str, default="save", help="save path")
    parser.add_argument("--debug", type=bool, default=False, help="debug mode")
    # parser.add_argument('--total_steps',
    #                     default=2**20,
    #                     type=int,
    #                     help='number of total steps to run')
    parser.add_argument('--eval_steps',
                        default=1024,
                        type=int,
                        help='number of eval steps to run')
    args = parser.parse_args()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    args.total_steps = args.eval_steps * args.epochs
    if args.dataset == 'stl10':
        args.num_labels = 1000

    set_seed(args.seed)
    set_logger(f'{args.save}/{args.dataset}_{args.num_labels}_training.log')

    logging.info(args)
    fixmatch = FixMatch(args)
    fixmatch.test()
    logging.shutdown()