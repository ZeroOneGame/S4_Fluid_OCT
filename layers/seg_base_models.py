# encoding: utf-8
"""
@author:  ZeroOneGame

More details are coming soon.

"""


from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from baseline.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)



class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)



class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]



class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output



class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg



class MLDS_OP_Head(nn.Module):
    def __init__(self, in_chns, n_class):
        super(MLDS_OP_Head, self).__init__()
        self.in_chns=in_chns
        self.n_class=n_class

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_chns, out_channels=in_chns, kernel_size=(1,1)),
            nn.BatchNorm2d(in_chns),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_chns, out_channels=n_class, kernel_size=(1,1)),
        )

    def forward(self, feat):
        return self.head(feat)



class Decoder_MLDS(nn.Module):
    def __init__(self, params):
        super(Decoder_MLDS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.aux_ml_head = self.params['aux_ml_head']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        if self.aux_ml_head:
            self.aux_head = MLDS_OP_Head(in_chns=self.ft_chns[0],n_class=self.n_class)
        else:
            self.aux_head = None
        # self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
        #                               kernel_size=3, padding=1)

        self.out_head_dp3 = MLDS_OP_Head(in_chns=self.ft_chns[3], n_class=self.n_class)
        self.out_head_dp2 = MLDS_OP_Head(in_chns=self.ft_chns[2], n_class=self.n_class)
        self.out_head_dp1 = MLDS_OP_Head(in_chns=self.ft_chns[1], n_class=self.n_class)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)

        if self.aux_ml_head:
            d0_out_aux = self.aux_head(x)
            return dp0_out_seg, d0_out_aux, dp1_out_seg, dp2_out_seg, dp3_out_seg
        else:
            return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg



class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output



class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg



class UNet_MLDS(nn.Module):
    def __init__(self, in_chns:int, class_num:int, aux_ml_head:bool=False, use_ddpp2:bool=False):
        super(UNet_MLDS, self).__init__()
        self.aux_ml_head = aux_ml_head
        self.encoder = Encoder(params)
        self.use_ddpp2 = use_ddpp2

        if use_ddpp2:
            self.ddpp2 = DDPP2(x14_chan=params['feature_chns'][-1],
            target_chan=params['feature_chns'][-1], dd_rates=params['dd_rates'])
        else:
            self.ddpp2 = nn.Identity()

        self.decoder = Decoder_MLDS(params)

    def forward(self, x):
        # shape = x.shape[2:]
        feature = self.encoder(x)
        feature[-1] = self.ddpp2(feature[-1])

        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg\
                = self.decoder(feature)
            return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg



class UNet_ASPP(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_ASPP, self).__init__()

        self.encoder = Encoder(params)
        self.aspp = ASPP(in_channels=params['feature_chns'][-1], out_channels=params['feature_chns'][-1],
                         atrous_rates=params['atrous_rates'], separable=True)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        feature[-1] = self.aspp(feature[-1])
        output = self.decoder(feature)
        return output



class UNet_DDPP(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DDPP, self).__init__()

        self.encoder = Encoder(params)
        self.ddpp = DilatedDispensedPyramidPooling(
            x14_chan=params['feature_chns'][-1], x28_chan=params['feature_chns'][-2],
            target_chan=params['feature_chns'][-1], dd_rates=params['dd_rates'])
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        feature[-1] = self.ddpp(feature[-1], feature[-2])
        output = self.decoder(feature)
        return output



class UNet_DDPP2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DDPP2, self).__init__()

        self.encoder = Encoder(params)
        self.ddpp = DDPP2(x14_chan=params['feature_chns'][-1],
            target_chan=params['feature_chns'][-1], dd_rates=params['dd_rates'])
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        feature[-1] = self.ddpp(feature[-1])
        output = self.decoder(feature)
        return output


class UNet_ASPP_DDPP2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_ASPP_DDPP2, self).__init__()

    def forward(self, x, aspp_out=True, ddpp_out=True):
        feature = self.encoder(x)
        feature_14_aspp = self.aspp(feature[-1])
        feature_14_ddpp = self.ddpp(feature[-1])

        feat_aspp = feature[:-1]
        feat_aspp.append(feature_14_aspp)
        feat_ddpp = feature[:-1]
        feat_ddpp.append(feature_14_ddpp)

        if aspp_out:
            aspp_output = self.aspp_decoder(feat_aspp)
        else:
            aspp_output = 0.

        if ddpp_out:
            ddpp_output = self.ddpp_decoder(feat_ddpp)
        else:
            ddpp_output = 0.

        return aspp_output, ddpp_output


class SwinUnet(nn.Module):
    """
    # coding=utf-8
    # This file borrowed from Swin-UNet: https://github.com/HuCaoFighting/Swin-Unet
    """
    def __init__(self, config, img_size=224, num_classes=4, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 


if __name__ == "__main__":
    pass
