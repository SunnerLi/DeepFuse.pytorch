from utils import INFO
import argparse
import torch
import os

def presentParameters(args_dict):
    """
        Print the parameters setting line by line
        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    INFO("========== Parameters ==========")
    for key in sorted(args_dict.keys()):
        INFO("{:>15} : {}".format(key, args_dict[key]))
    INFO("===============================")

class TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--folder1', type = str, required = True)
        parser.add_argument('--folder2', type = str, required = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--resume', type = str, default = "1.pth")
        parser.add_argument('--det', type = str, default = "train_result")
        parser.add_argument('--epoch', type = int, default = 100)
        self.opts = parser.parse_args()
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def parse(self):
        # Print the parameter first
        presentParameters(vars(self.opts))

        # Create the folder
        det_name = self.opts.det
        image_folder_name = os.path.join(det_name, "image")
        model_folder_name = os.path.join(det_name, "model")
        if not os.path.exists(self.opts.det):
            os.mkdir(self.opts.det)
        if not os.path.exists(image_folder_name):
            os.mkdir(image_folder_name)
        if not os.path.exists(model_folder_name):
            os.mkdir(model_folder_name)    
        return self.opts

###############################################################################################################

class TestOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--image1', type = str, required = True)
        parser.add_argument('--image2', type = str, required = True)
        parser.add_argument('--model', type = str, default = "model.pth")
        parser.add_argument('--res', type = str, default = 'result.png')
        self.opts = parser.parse_args()
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def parse(self):
        presentParameters(vars(self.opts))
        return self.opts