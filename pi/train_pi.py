import argparse, logging
from utils.utils import set_logger, set_seed
from pi import Pi

parser = argparse.ArgumentParser(description='Pi model Training')
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
parser.add_argument("--num_labels", type=int, default=4000, help="number of labeled data")
parser.add_argument("--lr", type=float, default=0.03, help="learning rate")
parser.add_argument("--ramp_up_length", type=int, default=80, help="ramp up length")
parser.add_argument("--ramp_down_length", type=int, default=50, help="ramp down length")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--root", type=str, default="../data", help="data path")
parser.add_argument("--save", type=str, default="save", help="save path")
parser.add_argument('--seed', type=int, default=-1, help='seed for random behaviors, no seed if negtive')
args = parser.parse_args()

set_seed(args.seed)
set_logger(f'{args.save}/{args.dataset}_{args.num_labels}_training.log')
pi_model = Pi(args)
pi_model.train()
logging.shutdown()