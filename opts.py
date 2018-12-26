from lib.utils import INFO
import argparse
import torch
import os

"""
    This script defines the procedure to parse the parameters

    Author: SunnerLi
"""

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
    """
                                                    Argument Explaination
        ======================================================================================================================
                Symbol          Type            Default                         Explaination
        ----------------------------------------------------------------------------------------------------------------------
            --folder            Str         /images/Bracketed_images        The folder path of bracketed image
            --crop_size         Int         256                             -
            --batch_size        Int         8                               -
            --resume            Str         1.pth                           The path of pre-trained model
            --det               Str         train_result                    The path of folder you want to store the result in
            --epoch             Int         15000                           -
            --record_epoch      Int         100                             The period you want to store the result
        ----------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--folder'          , type = str, default = "/home/sunner/Music/HDREyeDataset/images/Bracketed_images")
        parser.add_argument('--crop_size'       , type = int, default = 256)
        parser.add_argument('--batch_size'      , type = int, default = 8)
        parser.add_argument('--resume'          , type = str, default = "1.pth")
        parser.add_argument('--det'             , type = str, default = "train_result")
        parser.add_argument('--epoch'           , type = int, default = 15000)
        parser.add_argument('--record_epoch'    , type = int, default = 100)
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

###########################################################################################################################################

class TestOptions():
    """
                                                    Argument Explaination
        ======================================================================================================================
                Symbol          Type            Default                         Explaination
        ----------------------------------------------------------------------------------------------------------------------
            --image1            Str         X                               The path of under-exposure image
            --image2            Str         X                               The path of over-exposure image
            --model             Str         model.pth                       The path of pre-trained model
            --res               Str         result.png                      The path to store the fusing image
            --H                 Int         400                             The height of the result image
            --W                 Int         600                             The width of the result image
        ----------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--image1'  , type = str, required = True)
        parser.add_argument('--image2'  , type = str, required = True)
        parser.add_argument('--model'   , type = str, default = "model.pth")
        parser.add_argument('--res'     , type = str, default = 'result.png')
        parser.add_argument('--H'       , type = int, default = 400)
        parser.add_argument('--W'       , type = int, default = 600)
        self.opts = parser.parse_args()
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def parse(self):
        presentParameters(vars(self.opts))
        return self.opts