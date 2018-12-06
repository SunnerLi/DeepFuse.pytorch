from utils import INFO, fusePostProcess
from loss  import MEF_SSIM_Loss
from model import DeepFuse
from parse import parse

import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from torch.optim import Adam
from tqdm import tqdm

import numpy as np
import cv2

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

    # Train
    for ep in range(1000):
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
            bar.set_description("Epoch: %d   Loss: %.6f" % (ep, loss.item()))

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # patch2 = (patch2 + 1) * 127.5
            # img = patch2.transpose(1, 2).transpose(2, 3).cpu().numpy()[0]
            # img = (img).astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
            # cv2.imshow('show', img[:, :, ::-1])
            # cv2.waitKey()
            # exit()

            img = fusePostProcess(y_f, patch1, patch2)
            cv2.imshow('show', img[:, :, ::-1])
            cv2.waitKey(10)
            # exit()

if __name__ == '__main__':
    opts = parse()
    train(opts)