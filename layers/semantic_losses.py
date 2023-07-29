from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import torch.nn as nn


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DSLoss(nn.Module):
    def __init__(self):
        super(DSLoss, self).__init__()
        
        self.DS_type = DS_type
        self.mode = mode
        self.pool_type = pool_type
        self.classes=classes
        self.ds_depth = ds_depth
        self.from_logits=from_logits
        self.ac_func = nn.Sigmoid()
        self.dcloss_weight = dcloss_weight
        self.avg_thres = avg_thres
        
        if pool_type == "maxpool":
            self.label_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError(f"Not implemented pool type{pool_type}")
        self.dcloss = SMP.losses.DiceLoss(mode=mode, classes=classes, from_logits=from_logits)
        self.bceloss = torch.nn.BCEWithLogitsLoss()

    def forward(self, ds_out:List, label:Tensor):

        if self.DS_type == "MLDS":
            bs, n_c, _, _ = ds_out[0].shape
            max_depth = len(ds_out)
            assert len(ds_out) == self.ds_depth, \
                f"Length of ds_out({len(ds_out)}) is not equal to ds_depth ({self.ds_depth})."

            if self.pool_type == "maxpool":
                hard_label = label
                bs_l, h, w = hard_label.shape
                assert bs == bs_l, f"ds_out's({bs}) bs not equal to label's({bs_l})"
                MultiBinary_label = F.one_hot(hard_label, num_classes=self.classes
                                     ).permute(0, 3, 1, 2).float()  # 变为 BS * N_C * H * W, {0,1}
                hir_label = self._get_hierarchical_label(label=MultiBinary_label, max_depth=max_depth)

            else:
                raise NotImplementedError(f"Not implemented pool type{self.pool_type}")


            dc_loss, ce_loss = 0., 0.
            for d in range(max_depth):
                dc_loss += self.dcloss(y_pred=ds_out[d], y_true=hir_label[d])
                ce_loss += self.bceloss(ds_out[d], hir_label[d])

        else:
            raise NotImplementedError(f"Not implemented {self.DS_type}")

        total_loss = self.dcloss_weight * dc_loss + (1-self.dcloss_weight) * ce_loss
        return total_loss


class Semantic_Loss(nn.Module):
    def __init__(self):
        super(Semantic_Loss, self).__init__()
        self.DS_type = DS_type
        self.mode = mode
        self.pool_type = pool_type
        self.classes=classes
        self.ds_depth = ds_depth
        self.from_logits=from_logits
        self.ac_func = nn.Sigmoid()
        self.dcloss_weight = dcloss_weight
        self.avg_thres = avg_thres
        if pool_type == "maxpool":
            self.label_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "avgpool":
            self.label_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError(f"Not implemented pool type{pool_type}")
        self.dcloss = SMP.losses.DiceLoss(mode=mode, classes=classes, from_logits=from_logits)
        self.bceloss = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def _get_hierarchical_label(self, label, max_depth=4):
        hir_label = [label] 
        temp_label = label
        temp_possibility = label
        for d in range(max_depth):
            if self.pool_type == "maxpool":
                temp_label = self.label_pool(temp_label)
            elif self.pool_type == "avgpool":
                temp_possibility = self.label_pool(temp_possibility)
                temp_label = (temp_possibility > self.avg_thres).float()
            else:
                raise NotImplementedError(f"Not implemented pool type{self.pool_type}")
            hir_label.append(temp_label)
        return hir_label

    def forward(self, ds_out: List, label: Tensor, deep_list=None):
        if deep_list is None:
            deep_list = [2, 4]
        if self.DS_type == "MLDS":
            bs, n_c, _, _ = ds_out[0].shape
            max_depth = max(deep_list)

            if self.pool_type == "maxpool":
                hard_label = label
                bs_l, h, w = hard_label.shape
                assert bs == bs_l, f"ds_out's({bs}) bs not equal to label's({bs_l})"
                MultiBinary_label = F.one_hot(hard_label, num_classes=self.classes
                                     ).permute(0, 3, 1, 2).float()  # 变为 BS * N_C * H * W, {0,1}
                hir_label = self._get_hierarchical_label(label=MultiBinary_label, max_depth=max_depth)

            elif self.pool_type == "avgpool":
                bs_l, n_c, h, w = label.shape
                assert bs == bs_l, f"ds_out's({bs}) bs not equal to label's({bs_l})"
                hir_label = self._get_hierarchical_label(label=label, max_depth=max_depth)

            else:
                raise NotImplementedError(f"Not implemented pool type{self.pool_type}")

            dc_loss, ce_loss = 0., 0.
            for idx, d in enumerate(deep_list):
                dc_loss += self.dcloss(y_pred=ds_out[idx], y_true=hir_label[d])
                ce_loss += self.bceloss(ds_out[idx], hir_label[d])

        else:
            raise NotImplementedError(f"Not implemented {self.DS_type}")

        total_loss = self.dcloss_weight * dc_loss + (1-self.dcloss_weight) * ce_loss
        return total_loss



class Multi_Label_Semantic_Loss(nn.Module):
    def __init__(self, DS_type:str="MLDS", mode:str="multilabel", pool_type:str="maxpool",
                 classes:int=4, ds_depth:int=3, from_logits:bool=True,
                 dcloss_weight:float=0.5, en_weight:float=0.5, avg_thres:float=0.25):

        super(Multi_Label_Semantic_Loss, self).__init__()
        self.DS_type = DS_type
        self.mode = mode
        self.pool_type = pool_type
        self.classes=classes
        self.ds_depth = ds_depth
        self.from_logits=from_logits
        self.ac_func = nn.Sigmoid()
        self.dcloss_weight = dcloss_weight
        self.en_weight = en_weight
        self.avg_thres = avg_thres
        if pool_type == "maxpool":
            self.label_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            raise NotImplementedError(f"Not implemented pool type{pool_type}")
        elif pool_type == "avgpool" or pool_type == "avgpool2" or pool_type == "adj_avgpool":
            self.label_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError(f"Not implemented pool type{pool_type}")
        self.dcloss = SMP.losses.DiceLoss(mode=mode, classes=classes, from_logits=from_logits)
        self.bceloss = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def _get_hierarchical_label(self, label, max_depth=4):
        hir_label = []
        temp_label = label[0]
        temp_possibility = label[0]
        for d in range(max_depth):
            if self.pool_type == "maxpool":
                temp_label = self.label_pool(temp_label)

            elif self.pool_type == "avgpool":
                temp_possibility = self.label_pool(temp_possibility)
                temp_label = (temp_possibility + self.en_weight * label[d+1]) / 2
                temp_label = (temp_label > self.avg_thres).float()

            elif self.pool_type == "avgpool2":
                temp_possibility = self.label_pool(temp_possibility)
                temp_label = (1 - self.en_weight) * temp_possibility + self.en_weight * label[d+1]
                temp_label = (temp_label > self.avg_thres).float()

            elif self.pool_type == "adj_avgpool":
                temp_possibility = (1 - self.en_weight) * self.label_pool(label[d]) + self.en_weight * label[d+1]
                temp_label = (temp_possibility > self.avg_thres).float()

            else:
                raise NotImplementedError(f"Not implemented pool type{self.pool_type}")
            hir_label.append(temp_label)
        return hir_label

    def forward(self, ds_out:List[Tensor], label:List[Tensor]):
        if self.DS_type == "MLDS":
            bs, n_c, _, _ = ds_out[0].shape
            max_depth = len(ds_out)
            assert len(ds_out) == self.ds_depth == (len(label) - 1), \
                f"Length of ds_out({len(ds_out)}) is not equal to ds_depth ({self.ds_depth})."

            if self.pool_type == "maxpool":
                hard_label = label
                bs_l, h, w = hard_label.shape
                assert bs == bs_l, f"ds_out's({bs}) bs not equal to label's({bs_l})"
                MultiBinary_label = F.one_hot(hard_label, num_classes=self.classes).permute(0, 3, 1, 2).float()
                hir_label = self._get_hierarchical_label(label=MultiBinary_label, max_depth=max_depth)

            elif self.pool_type == "avgpool" or self.pool_type == "avgpool2" or self.pool_type == "adj_avgpool":
                bs_l, n_c, h, w = label[0].shape
                assert bs == bs_l, f"ds_out's({bs}) bs not equal to label's({bs_l})"
                hir_label = self._get_hierarchical_label(label=label, max_depth=max_depth)

            else:
                raise NotImplementedError(f"Not implemented pool type{self.pool_type}")


            dc_loss, ce_loss = 0., 0.
            for d in range(max_depth):
                dc_loss += self.dcloss(y_pred=ds_out[d], y_true=hir_label[d])
                ce_loss += self.bceloss(ds_out[d], hir_label[d])

        else:
            raise NotImplementedError(f"Not implemented {self.DS_type}")

        total_loss = self.dcloss_weight * dc_loss + (1-self.dcloss_weight) * ce_loss
        return total_loss



def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


if __name__=="__main__":

    pass
