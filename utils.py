import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

import numpy as np
import torch
import cv2

"""
    This script defines the fundamental function which will be used in other script

    Author: SunnerLi
"""

L1_NORM = lambda b: torch.sum(torch.abs(b))

def INFO(string):
    print("[ DeepFuse ] %s" % (string))

def weightedFusion(cr1, cr2, cb1, cb2):
    """
        Perform the weighted fusing for Cb and Cr channel (paper equation 6)

        Arg:    cr1     (torch.Tensor)  - The Cr slice of 1st image
                cr2     (torch.Tensor)  - The Cr slice of 2nd image
                cb1     (torch.Tensor)  - The Cb slice of 1st image
                cb2     (torch.Tensor)  - The Cb slice of 2nd image
        Ret:    The fused Cr slice and Cb slice
    """
    # Fuse Cr channel
    cr_up = (cr1 * L1_NORM(cr1 - 127.5) + cr2 * L1_NORM(cr2 - 127.5))
    cr_down = L1_NORM(cr1 - 127.5) + L1_NORM(cr2 - 127.5)
    cr_fuse = cr_up / cr_down

    # Fuse Cb channel
    cb_up = (cb1 * L1_NORM(cb1 - 127.5) + cb2 * L1_NORM(cb2 - 127.5))
    cb_down = L1_NORM(cb1 - 127.5) + L1_NORM(cb2 - 127.5)
    cb_fuse = cb_up / cb_down

    return cr_fuse, cb_fuse

def fusePostProcess(y_f, y_hat, img1, img2, single = True):
    """
        Perform the post fusion process toward the both image with generated luminance slice

        Arg:    y_f     (torch.Tensor)  - The generated luminance slice
                img1    (torch.Tensor)  - The 1st image tensor (in YCrCb format)
                img2    (torch.Tensor)  - The 2nd image tensor (in YCrCb format)
                single  (Bool)          - If return the fusion result only or not
        Ret:    The fusion output image
    """
    with torch.no_grad():    
        # Recover value space [-1, 1] -> [0, 255]
        y_f   = (y_f   + 1) * 127.5
        y_hat = (y_hat + 1) * 127.5
        img1  = (img1  + 1) * 127.5
        img2  = (img2  + 1) * 127.5

        # weight fusion for Cb and Cr
        cr_fuse, cb_fuse = weightedFusion(
            cr1 = img1[:, 1:2],
            cr2 = img2[:, 1:2],
            cb1 = img1[:, 2:3],
            cb2 = img2[:, 2:3]
        )

        # YCbCr -> BGR
        fuse_out = torch.zeros_like(img1)
        fuse_out[:, 0:1] = y_f
        fuse_out[:, 1:2] = cr_fuse
        fuse_out[:, 2:3] = cb_fuse
        fuse_out = fuse_out.transpose(1, 2).transpose(2, 3).cpu().numpy()
        fuse_out = fuse_out.astype(np.uint8)
        for i, m in enumerate(fuse_out):
            fuse_out[i] = cv2.cvtColor(m, cv2.COLOR_YCrCb2BGR)

        # Combine the output
        if not single:
            out1 = img1.transpose(1, 2).transpose(2, 3).cpu().numpy().astype(np.uint8)
            for i, m in enumerate(out1):
                out1[i] = cv2.cvtColor(m, cv2.COLOR_YCrCb2BGR)
            out2 = img2.transpose(1, 2).transpose(2, 3).cpu().numpy().astype(np.uint8)
            for i, m in enumerate(out2):
                out2[i] = cv2.cvtColor(m, cv2.COLOR_YCrCb2BGR)
            out3 = torch.zeros_like(img1)
            out3[:, 0:1] = y_hat
            out3[:, 1:2] = cr_fuse
            out3[:, 2:3] = cb_fuse
            out3 = out3.transpose(1, 2).transpose(2, 3).cpu().numpy()
            out3 = out3.astype(np.uint8)
            for i, m in enumerate(out3):
                out3[i] = cv2.cvtColor(m, cv2.COLOR_YCrCb2BGR)
            out  = np.concatenate((out1, out2, fuse_out, out3), 2)
        else:
            out = fuse_out
        return out