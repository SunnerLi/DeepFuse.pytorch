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

def fusePostProcess(y_f, img1, img2):
    with torch.no_grad():    
        # Recover value space [-1, 1] -> [0, 255]
        y_f  = (y_f + 1) * 127.5
        # img1 = torch.cat([y_f, img1[:, 1:]], 1)
        img1 = (img1 + 1) * 127.5
        # img2 = torch.cat([y_f, img2[:, 1:]], 1)
        img2 = (img2 + 1) * 127.5

        # weight fusion for Cb and Cr
        cr_fuse, cb_fuse = weightedFusion(
            cr1 = img1[:, 1:2],
            cr2 = img2[:, 1:2],
            cb1 = img1[:, 2:3],
            cb2 = img2[:, 2:3]
        )

        # img2[:, 0] = y_f
        # img2[:, 1] = cr_fuse
        # img2[:, 2] = cb_fuse
        # img = img2.transpose(1, 2).transpose(2, 3).cpu().numpy()[0]
        # img = (img).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        # cv2.imshow('show', img[:, :, ::-1])
        # cv2.waitKey()
        # exit()

        # YCbCr -> BGR
        # img = torch.cat([y_f, cr_fuse, cb_fuse], 1)
        fuse_out = torch.zeros_like(img1)
        fuse_out[:, 0] = y_f
        fuse_out[:, 1] = cr_fuse
        fuse_out[:, 2] = cb_fuse
        # img[:, 0] = img1[:, 0]
        # img[:, 1] = img1[:, 1]
        # img[:, 2] = img1[:, 2]
        fuse_out = fuse_out.transpose(1, 2).transpose(2, 3).cpu().numpy()[0]
        fuse_out = fuse_out.astype(np.uint8)
        fuse_out = cv2.cvtColor(fuse_out, cv2.COLOR_YCrCb2BGR)

        img1 = img1.transpose(1, 2).transpose(2, 3).cpu().numpy()[0].astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_YCrCb2BGR)
        img2 = img2.transpose(1, 2).transpose(2, 3).cpu().numpy()[0].astype(np.uint8)
        img2 = cv2.cvtColor(img2, cv2.COLOR_YCrCb2BGR)

        out  = np.concatenate((img1, img2, fuse_out), 1)
        return out