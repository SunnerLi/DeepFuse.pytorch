from utils import INFO, fusePostProcess
from loss  import MEF_SSIM_Loss
from model import DeepFuse
from opts  import TrainOptions

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

def train(opts):
    # Create the loader
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root = [[opts.folder1], [opts.folder2]],
        transform = transforms.Compose([
            sunnertransforms.Resize((256, 256)),
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize(),
        ])), batch_size = opts.batch_size, shuffle = False, num_workers = 2
    )

    # Create the model
    model = DeepFuse(device = opts.device)
    criterion = MEF_SSIM_Loss().to(opts.device)
    optimizer = Adam(model.parameters(), lr = 0.0004)

    # Load pre-train model
    if os.path.exists(opts.resume):
        state = torch.load(opts.resume)
        Loss_list = state['loss']
        model.load_state_dict(state['model'])
    else:
        Loss_list = []

    # Train
    for ep in range(opts.epoch):
        loss_list = []
        bar = tqdm(loader)
        for (patch1, patch2) in bar:
            # Extract the luminance and move to computation device
            patch1, patch2 = patch1.to(opts.device), patch2.to(opts.device)
            patch1_lum = patch1[:, 0:1]
            patch2_lum = patch2[:, 0:1]

            # Forward and compute loss
            model.setInput(patch1_lum, patch2_lum)
            y_f  = model.forward()
            loss = criterion(y_1 = patch1_lum, y_2 = patch2_lum, y_f = y_f)
            loss_list.append(loss.item())
            bar.set_description("Epoch: %d   Loss: %.6f" % (ep, loss_list[-1]))

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss_list.append(np.mean(loss_list))

        # Save the training image
        img = fusePostProcess(y_f, patch1, patch2, single=False)
        cv2.imwrite(os.path.join(opts.det, 'image', str(ep) + ".png"), img[0, :, :, ::-1])

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
    plt.plot(Loss_list, '-o')
    plt.title("loss curve")
    plt.savefig(os.path.join(opts.det, 'image', "curve.png"))

if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)