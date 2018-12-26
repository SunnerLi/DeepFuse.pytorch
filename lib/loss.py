from math import exp
import torch.nn.functional as F
import torch.nn as nn
import torch

"""
    This script defines the MEF-SSIM loss function which is mentioned in the DeepFuse paper
    The code is heavily borrowed from: https://github.com/Po-Hsun-Su/pytorch-ssim

    Author: SunnerLi
"""

L2_NORM = lambda b: torch.sqrt(torch.sum((b + 1e-8) ** 2))

class MEF_SSIM_Loss(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average = True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def w_fn(self, y):
        """
            Return the weighting function that MEF-SSIM defines
            We use the power engery function as the paper describe: https://ece.uwaterloo.ca/~k29ma/papers/15_TIP_MEF.pdf

            Arg:    y   (torch.Tensor)  - The structure tensor
            Ret:    The weight of the given structure
        """
        out = torch.sqrt(torch.sum(y ** 2))
        return out

    def forward(self, y_1, y_2, y_f):
        """
            Compute the MEF-SSIM for the given image pair and output image
            The y_1 and y_2 can exchange

            Arg:    y_1     (torch.Tensor)  - The LDR image
                    y_2     (torch.Tensor)  - Another LDR image in the same stack
                    y_f     (torch.Tensor)  - The fused HDR image
            Ret:    The loss value
        """
        miu_y = (y_1 + y_2) / 2

        # Get the c_hat
        c_1 = L2_NORM(y_1 - miu_y)
        c_2 = L2_NORM(y_2 - miu_y)
        c_hat = torch.max(torch.stack([c_1, c_2]))

        # Get the s_hat
        s_1 = (y_1 - miu_y) / L2_NORM(y_1 - miu_y)
        s_2 = (y_2 - miu_y) / L2_NORM(y_2 - miu_y)
        s_bar = (self.w_fn(y_1) * s_1 + self.w_fn(y_2) * s_2) / (self.w_fn(y_1) + self.w_fn(y_2))
        s_hat = s_bar / L2_NORM(s_bar)

        # =============================================================================================
        # < Get the y_hat >
        #
        # Rather to output y_hat, we shift it with the mean of the over-exposure image and mean image
        # The result will much better than the original formula
        # =============================================================================================
        y_hat = c_hat * s_hat
        y_hat += (y_2 + miu_y) / 2

        # Check if need to create the gaussian window 
        (_, channel, _, _) = y_hat.size()
        if channel == self.channel and self.window.data.type() == y_hat.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(y_f.get_device())
            window = window.type_as(y_hat)
            self.window = window
            self.channel = channel

        # Compute SSIM between y_hat and y_f
        score = self._ssim(y_hat, y_f, window, self.window_size, channel, self.size_average)        
        return 1 - score, y_hat