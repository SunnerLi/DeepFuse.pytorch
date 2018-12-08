from utils import INFO, fusePostProcess
from opts  import TestOptions
from model import DeepFuse

import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from skimage import io as io
import torch
import cv2
import os

def inference(opts):
    # Load the image
    ops = transforms.Compose([
        sunnertransforms.Resize((256, 256)),
        sunnertransforms.ToTensor(),
        sunnertransforms.ToFloat(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize(),
    ])
    img1 = io.imread(opts.image1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
    img1 = torch.unsqueeze(ops(img1), 0)
    img2 = io.imread(opts.image2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    img2 = torch.unsqueeze(ops(img2), 0)

    # Load the pre-trained model
    model = DeepFuse()
    state = torch.load(opts.model)
    model.load_state_dict(state['model'])
    model.to(opts.device)
    model.eval()

    # Fuse!
    with torch.no_grad():
        # Forward
        img1, img2 = img1.to(opts.device), img2.to(opts.device)
        img1_lum = img1[:, 0:1]
        img2_lum = img2[:, 0:1]
        model.setInput(img1_lum, img2_lum)
        y_f  = model.forward()

        # Save the image
        img = fusePostProcess(y_f, img1, img2, single=True)
        cv2.imwrite(opts.res, img[0, :, :, ::-1])

if __name__ == '__main__':
    opts = TestOptions().parse()
    inference(opts)