import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

import numpy as np
import torch
import cv2

L1_NORM = lambda b: torch.mean(torch.abs(b))

def INFO(string):
    print("[ DeepFuse ] %s" % (string))

def weightedFusion(cr1, cr2, cb1, cb2):
    # Fuse Cr channel
    cr_up = (cr1 * L1_NORM(cr1 - 128) + cr2 * L1_NORM(cr2 - 128))
    cr_down = L1_NORM(cr1 - 128) + L1_NORM(cr2 - 128)
    cr_fuse = cr_up / cr_down

    # Fuse Cb channel
    cb_up = (cb1 * L1_NORM(cb1 - 128) + cb2 * L1_NORM(cb2 - 128))
    cb_down = L1_NORM(cb1 - 128) + L1_NORM(cb2 - 128)
    cb_fuse = cb_up / cb_down

    return cr_fuse, cb_fuse

def fusePostProcess(y_f, img1, img2, single = True):
    with torch.no_grad():    
        # Recover value space [-1, 1] -> [0, 255]
        y_f  = (y_f + 1) * 127.5
        img1 = (img1 + 1) * 127.5
        img2 = (img2 + 1) * 127.5

        # weight fusion for Cb and Cr
        cr_fuse, cb_fuse = weightedFusion(
            cr1 = img1[:, 1:2],
            cr2 = img2[:, 1:2],
            cb1 = img1[:, 2:3],
            cb2 = img2[:, 2:3]
        )

        # YCbCr -> BGR
        fuse_out = torch.zeros_like(img1)
        fuse_out[:, 0] = y_f
        fuse_out[:, 1] = cr_fuse
        fuse_out[:, 2] = cb_fuse
        fuse_out = fuse_out.transpose(1, 2).transpose(2, 3).cpu().numpy()
        fuse_out = fuse_out.astype(np.uint8)
        for i, m in enumerate(fuse_out):
            fuse_out[i] = cv2.cvtColor(m, cv2.COLOR_YCrCb2RGB)

        # Combine the output
        if not single:
            img1 = img1.transpose(1, 2).transpose(2, 3).cpu().numpy().astype(np.uint8)
            for i, m in enumerate(img1):
                img1[i] = cv2.cvtColor(m, cv2.COLOR_YCrCb2RGB)
            img2 = img2.transpose(1, 2).transpose(2, 3).cpu().numpy().astype(np.uint8)
            for i, m in enumerate(img2):
                img2[i] = cv2.cvtColor(m, cv2.COLOR_YCrCb2RGB)
            out  = np.concatenate((img1, img2, fuse_out), 2)
        else:
            out = fuse_out
        return out