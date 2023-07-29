# encoding: utf-8
"""
@author:  ZeroOneGame

More details are coming soon.

"""

import torch, cv2
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# TIP2011: FSIM: A Feature Similarity Index for Image Quality Assessment
sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
sobel_y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

prewitt_x = torch.tensor([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
prewitt_y = torch.tensor([[1,  1,  1],
                          [0,  0,  0],
                          [-1, -1, -1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

scharr_x = torch.tensor([[3, 0, -3],
                         [10, 0, -10],
                         [3, 0, -3]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
scharr_y = torch.tensor([[3, 10, 3],
                        [0, 0, 0],
                        [-3, -10, -3]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

avgpool = torch.tensor([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

curvature = torch.tensor([[-1/16, 5/16, -1/16],
                               [5/16,  -1,   5/16],
                               [-1/16, 5/16, -1/16],], dtype=torch.float, requires_grad=False).view(1,1,3,3)


class CurvatureLoss(nn.Module):
    def __init__(self, in_chans, reduction="mean"):
        super(CurvatureLoss, self).__init__()
        self.in_chans = in_chans
        self.reduction = reduction
        self.param_curvature = nn.Parameter(curvature.repeat(1, self.in_chans, 1, 1), requires_grad=False)

    def forward(self, pr, gt):
        gt = gt.detach().clone()
        cur_pr = F.conv2d(pr, self.param_curvature, stride=1, padding=1)
        cur_gt = F.conv2d(gt, self.param_curvature, stride=1, padding=1)

        loss_cur = torch.abs((cur_pr * gt) / (cur_gt + 1e-5))

        if self.reduction == "mean":
            loss_cur = torch.mean(loss_cur.sum((0,1)))
        elif self.reduction == "sum":
            loss_cur = torch.sum(loss_cur)
        else:
            pass
        return loss_cur


class SobelLoss2(nn.Module):
    def __init__(self, in_chans, reduction="mean", sim_measure_type="l1"):
        """
        reduction: mean, sum, none
        sim_measure: cos, kl, l2, l1
        """
        super(SobelLoss2, self).__init__()
        self.in_chans = in_chans
        self.reduction = reduction
        self.sim_measure_type = sim_measure_type

        if sim_measure_type == "l1":
            self.sim_measure = nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError(f"Note implemented {sim_measure_type}")

        self.param_conv_x = nn.Parameter(sobel_x.repeat(1, self.in_chans, 1, 1), requires_grad=False)
        self.param_conv_y = nn.Parameter(sobel_y.repeat(1, self.in_chans, 1, 1), requires_grad=False)

    def forward(self, pr, gt):
        gt = gt.detach().clone()
        pr_grad_x = F.conv2d(pr, self.param_conv_x, stride=1, padding=1)
        pr_grad_y = F.conv2d(pr, self.param_conv_y, stride=1, padding=1)

        gt_grad_x = F.conv2d(gt, self.param_conv_x, stride=1, padding=1)
        gt_grad_y = F.conv2d(gt, self.param_conv_y, stride=1, padding=1)

        if self.sim_measure_type == "l1":
            loss = self.sim_measure(pr_grad_x, gt_grad_x) + self.sim_measure(pr_grad_y, gt_grad_y)
        elif self.sim_measure_type == "l2":
            loss = self.sim_measure(pr_grad_x, gt_grad_x) + self.sim_measure(pr_grad_y, gt_grad_y)
        else:
            loss = 0
        return loss


class ScharrLoss2(nn.Module):
    def __init__(self, in_chans, reduction="mean", sim_measure_type="l1"):
        """
        reduction: mean, sum, none
        sim_measure: cos, kl, l2, l1
        """
        super(ScharrLoss2, self).__init__()
        self.in_chans = in_chans
        self.reduction = reduction
        self.sim_measure_type = sim_measure_type

        if sim_measure_type == "l1":
            self.sim_measure = nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError(f"Note implemented {sim_measure_type}")

        self.param_conv_x = nn.Parameter(scharr_x.repeat(1, self.in_chans, 1, 1), requires_grad=False)
        self.param_conv_y = nn.Parameter(scharr_y.repeat(1, self.in_chans, 1, 1), requires_grad=False)

    def forward(self, pr, gt):
        gt = gt.detach().clone()
        pr_grad_x = F.conv2d(pr, self.param_conv_x, stride=1, padding=1)
        pr_grad_y = F.conv2d(pr, self.param_conv_y, stride=1, padding=1)

        gt_grad_x = F.conv2d(gt, self.param_conv_x, stride=1, padding=1)
        gt_grad_y = F.conv2d(gt, self.param_conv_y, stride=1, padding=1)

        if self.sim_measure_type == "l1":
            loss = self.sim_measure(pr_grad_x, gt_grad_x) + self.sim_measure(pr_grad_y, gt_grad_y)
        elif self.sim_measure_type == "l2":
            loss = self.sim_measure(pr_grad_x, gt_grad_x) + self.sim_measure(pr_grad_y, gt_grad_y)
        else:
            loss = 0
        return loss


class LaplaceLoss(nn.Module):
    def __init__(self, in_chans, reduction="mean", sim_measure_type="l1"):
        """
        reduction: mean, sum, none
        sim_measure: cos, kl, l2, l1
        """
        super(LaplaceLoss, self).__init__()
        self.in_chans = in_chans
        self.reduction = reduction
        self.sim_measure_type = sim_measure_type

    def forward(self, pr, gt):
        gt = gt.detach().clone()

        pr_grad_x = F.conv2d(pr, self.param_conv, stride=1, padding=1)
        gt_grad_x = F.conv2d(gt, self.param_conv, stride=1, padding=1)

        if self.sim_measure_type == "cos":
            B,C,H,W = pr_grad_x.shape
            loss = self.sim_measure(input1=pr_grad_x.view(B,-1),
                                    input2=gt_grad_x.view(B,-1),
                                    target=torch.ones(size=(B,), device="cuda:0"))
        elif self.sim_measure_type == "l2":
            loss = self.sim_measure(pr_grad_x, gt_grad_x)
        elif self.sim_measure_type == "l1":
            loss = self.sim_measure(pr_grad_x, gt_grad_x)
        else:
            loss = 0
        return loss



def conv_operator(filename, kernel, in_channels=1):
    if in_channels == 1:
        img = np.expand_dims(cv2.imread(filename, 0), 2)    # gray
    elif in_channels == 3:
        img = cv2.imread(filename, 1)                        # bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        exit()

    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    y = F.conv2d(x, kernel.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
    y = y.squeeze(0).numpy().transpose(1, 2, 0)

    return img, y

def grad_fuse(filename, kernel1, kernel2, in_channels=1):
    if in_channels == 1:
        img = np.expand_dims(cv2.imread(filename, 0), 2)    # gray
    elif in_channels == 3:
        img = cv2.imread(filename, 1)                        # bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        exit()

    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()

    sobel_detect = SobelDetect(in_chans=in_channels,grad_fuse=True)
    y = sobel_detect(x)

    y = y.squeeze(0).numpy().transpose(1, 2, 0)

    return img, y

def cur_fuse(filename, in_channels=1):
    if in_channels == 1:
        img = np.expand_dims(cv2.imread(filename, 0), 2)    # gray
    elif in_channels == 3:
        img = cv2.imread(filename, 1)                        # bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        exit()

    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    cur = CurvatureDetect(in_chans=in_channels)
    y = cur(x)
    y = y.squeeze(0).numpy().transpose(1, 2, 0)

    return img, y

def plt_show(windowsname, img, channels=1):
    plt.figure(windowsname)
    if channels ==1:
        plt.imshow(img, cmap='gray')
    elif channels == 3:
        plt.imshow(img, )
    else:
        exit()

    plt.title(windowsname)
    plt.axis('on')
    plt.show()


if __name__=="__main__":
    img_name = r'xxx.png'
