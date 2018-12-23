from dataset import BracketedDataset
from utils import INFO, fusePostProcess
from loss  import MEF_SSIM_Loss
from opts  import TrainOptions
from model import DeepFuse

import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from torch.optim import Adam
from tqdm import tqdm

import numpy as np
import torch
import cv2
import os

"""
    This script defines the training procedure of DeepFuse

    Author: SunnerLi
"""

def train(opts):
    # Create the loader
    loader = sunnerData.DataLoader(
        dataset = BracketedDataset(
            root = opts.folder,
            crop_size = opts.crop_size,
            transform = transforms.Compose([
                sunnertransforms.ToTensor(),
                sunnertransforms.ToFloat(),
                sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
                sunnertransforms.Normalize(),
            ])
        ), batch_size = opts.batch_size, shuffle = True, num_workers = 8
    )

    # Create the model
    model = DeepFuse(device = opts.device)
    criterion = MEF_SSIM_Loss().to(opts.device)
    optimizer = Adam(model.parameters(), lr = 0.0001)

    # Load pre-train model
    if os.path.exists(opts.resume):
        state = torch.load(opts.resume)
        Loss_list = state['loss']
        model.load_state_dict(state['model'])
    else:
        Loss_list = []

    # Train
    bar = tqdm(range(opts.epoch))
    for ep in bar:
        loss_list = []
        for (patch1, patch2) in loader:
            # Extract the luminance and move to computation device
            patch1, patch2 = patch1.to(opts.device), patch2.to(opts.device)
            patch1_lum = patch1[:, 0:1]
            patch2_lum = patch2[:, 0:1]

            # Forward and compute loss
            model.setInput(patch1_lum, patch2_lum)
            y_f  = model.forward()
            loss, y_hat = criterion(y_1 = patch1_lum, y_2 = patch2_lum, y_f = y_f)
            loss_list.append(loss.item())
            bar.set_description("Epoch: %d   Loss: %.6f" % (ep, loss_list[-1]))

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss_list.append(np.mean(loss_list))

        # Save the training image
        if ep % 100 == 0:
            img = fusePostProcess(y_f, y_hat, patch1, patch2, single=False)
            cv2.imwrite(os.path.join(opts.det, 'image', str(ep) + ".png"), img[0, :, :, :])

        # Save the training model
        if ep % (opts.epoch // 5) == 0:
            model_name = str(ep) + ".pth"
        else:
            model_name = "latest.pth"
        state = {
            'model': model.state_dict(),
            'loss' : Loss_list
        }
        torch.save(state, os.path.join(opts.det, 'model', model_name))

    # Plot the loss curve
    plt.clf()
    plt.plot(Loss_list, '-')
    plt.title("loss curve")
    plt.savefig(os.path.join(opts.det, 'image', "curve.png"))

if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)