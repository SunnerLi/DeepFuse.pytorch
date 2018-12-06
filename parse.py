from utils import INFO
import argparse
import torch

def __presentParameters(args_dict):
    """
        Print the parameters setting line by line
        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    INFO("========== Parameters ==========")
    for key in sorted(args_dict.keys()):
        INFO("{:>15} : {}".format(key, args_dict[key]))
    INFO("===============================")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type = str, required = True)
    parser.add_argument('--folder2', type = str, required = True)
    parser.add_argument('--batch_size', type = int, default = 32)
    opts = parser.parse_args()
    opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return opts