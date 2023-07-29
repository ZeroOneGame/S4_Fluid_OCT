# encoding: utf-8
"""
@author:  ZeroOneGame

More details are coming soon.

"""

import os
from typing import Optional, Union, List

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import random


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

class Unet_MLDS_Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            num_classes=4,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        mlds_heads = [
            MLDS_OP_Head(in_chns=de_out, n_class=num_classes) for de_out in out_channels
        ]
        self.mlds_heads = nn.ModuleList(mlds_heads)


    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        mlds_outs = []

        for i, (decoder_block, mlds_head) in enumerate(zip(self.blocks, self.mlds_heads)):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            mlds_outs.append(mlds_head(x))

        return x, mlds_outs


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):

        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class EffiUnet(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "tu-tf_efficientnetv2_b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class ASPP3(nn.Module):
    def __init__(self, in14_chan:int, out14_chan:int, in7_chan:int, out7_chan:int,
                 atrous14_rates: List[int] = (6, 12, 18), atrous7_rates: List[int] = (3, 6, 9)):
        super(ASPP3, self).__init__()

    def forward(self, x14, x7):

        x14_aspp = self.x14_path(x14)
        x7_aspp = self.x7_path(x7)

        x14_out = torch.cat([x14, x14_aspp], dim=1)
        x7_out = torch.cat([x7, x7_aspp], dim=1)

        x14_out = self.x14_adap(x14_out)
        x7_out = self.x7_adap(x7_out)

        return x14_out, x7_out


class Adap_Inter(nn.Module):
    def __init__(self, target_size=(8,8)):
        super(Adap_Inter, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size)


class DDPT(nn.Module):
    def __init__(self, in14_chan:int=112, out14_chan:int=112,
                 in7_chan:int=192, out7_chan:int=192,
                 dd14_rates:List[int]=(2, 4, 8), dd7_rates:List[int]=(2, 2, 4)):
        super(DDPT, self).__init__()

        self.mid14_chan = out14_chan // 4 # =64
        self.mid7_chan = out7_chan // 2 #

        self.x14_pad_paths = nn.ModuleList([
            self._get_x14_pad_DDDP_Model(in_chan=in14_chan, out_chan=self.mid14_chan, patch_size=dd14_rates[0], proj_factor=4),
            self._get_x14_pad_DDDP_Model(in_chan=in14_chan, out_chan=self.mid14_chan, patch_size=dd14_rates[1], proj_factor=4),
            self._get_x14_pad_DDDP_Model(in_chan=in14_chan, out_chan=self.mid14_chan, patch_size=dd14_rates[2], proj_factor=4),
        ])
        self.x14_con_adap = nn.Conv2d(in_channels=(in14_chan + len(self.x14_pad_paths) * self.mid14_chan),
                                      out_channels= out14_chan, kernel_size=(3, 3), padding=1)

        self.x7_adap = Adap_Inter(target_size=(8, 8))
        self.x7_pad_paths = nn.ModuleList([
            self._get_x8_pad_DDDP_Model(in_chan=in7_chan, out_chan=self.mid7_chan, patch_size=dd7_rates[0], proj_factor=2),
            self._get_x8_pad_DDDP_Model(in_chan=in7_chan, out_chan=self.mid7_chan, patch_size=dd7_rates[1], proj_factor=2),
            self._get_x8_pad_DDDP_Model(in_chan=in7_chan, out_chan=self.mid7_chan, patch_size=dd7_rates[2], proj_factor=2),
        ])
        self.x7_con_adap = nn.Conv2d(in_channels=(in7_chan + len(self.x7_pad_paths) * self.mid7_chan),
                                     out_channels=out7_chan, kernel_size=(3, 3), padding=1)


    def forward(self, x14, x7):
        """
        :param x14: (B,C,14,14)
        :param x7: (B,C,7,7)
        :return:
        """

        x16_x14s = []
        for x14_pad_p in self.x14_pad_paths:
            x14_ = x14_pad_p(x14)
            x16_x14s.append(x14_)
        x14_out = torch.cat([x14, *x16_x14s], dim=1)
        x14_out = self.x14_con_adap(x14_out)

        x8_x7s = []
        x7_x8 = self.x7_adap(x7)
        for x7_pad_p in self.x7_pad_paths:
            x8_x7s.append(x7_pad_p(x7_x8))
        x7_out = torch.cat([x7, *x8_x7s], dim=1)
        x7_out = self.x7_con_adap(x7_out)

        return x14_out, x7_out


class EffiUnet_ASPP3_DDPT(torch.nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_chns: A number of input channels for the model, default is 3 (RGB images)
        class_num: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: Unet
    """

    def __init__(
        self,
        encoder_name: str = "tu-tf_efficientnetv2_b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_chns: int = 3,
        class_num: int = 1,
        activation: Optional[Union[str, callable]] = None,
        rand_route: bool = False,
        rand_route_thres: float = 0.5,
        pp_module1:str = "ASPP3",
        pp_module2: str = "DDPT",
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_chns,
            depth=encoder_depth,
            weights=encoder_weights,
        ) # [224@3, 112@16, 56@32, 28@48, 14@112, 7@192]
        self.encode_out_chan = self.encoder.out_channels

        self.rand_route = rand_route
        self.rand_route_thres = rand_route_thres

        self.name = "u-{}".format(encoder_name)
        self.initialize()



    def initialize(self):
        init.initialize_decoder(self.pp_module1)
        init.initialize_decoder(self.pp_decoder1)
        init.initialize_decoder(self.pp_module2)
        init.initialize_decoder(self.pp_decoder2)

        init.initialize_head(self.pp_seg_head1)
        init.initialize_head(self.pp_seg_head2)


    def forward(self, x, aspp_out=True, ddpp_out=True):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        aspp3_feats = features[:-2]
        DDPT_feats = features[:-2]

        if self.training and self.rand_route and random() < self.rand_route_thres:
            aspp3_feats, DDPT_feats = DDPT_feats, aspp3_feats
        else:
            pass

        aspp3_feats.extend(self.pp_module1(x14=features[-2], x7=features[-1]))
        DDPT_feats.extend(self.pp_module2(x14=features[-2], x7=features[-1]))

        aspp3_decoder_output = self.pp_decoder1(*aspp3_feats)
        DDPT_decoder_output = self.pp_decoder2(*DDPT_feats)

        pp_masks1 = self.pp_seg_head1(aspp3_decoder_output)
        pp_masks2 = self.pp_seg_head2(DDPT_decoder_output)

        return pp_masks1, pp_masks2


class EffiUnet_AS_DD_PP3(torch.nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_chns: A number of input channels for the model, default is 3 (RGB images)
        class_num: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: Unet
    """

    def __init__(
        self,
        encoder_name: str = "tu-tf_efficientnetv2_b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_chns: int = 3,
        class_num: int = 1,
        activation: Optional[Union[str, callable]] = None,
        rand_route: bool = False,
        rand_route_thres: float = 0.5,
        pp_module:str = "ASPP3"
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_chns,
            depth=encoder_depth,
            weights=encoder_weights,
        ) # [224@3, 112@16, 56@32, 28@48, 14@112, 7@192]
        self.encode_out_chan = self.encoder.out_channels


        self.decoder = UnetDecoder(
            encoder_channels=self.encode_out_chan,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=class_num,
            activation=activation,
            kernel_size=3,
        )

        self.rand_route = rand_route
        self.rand_route_thres = rand_route_thres

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.pp3)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.seg_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        aspp3_feats = features[:-2]

        aspp3_feats.extend(self.pp3(x14=features[-2], x7=features[-1]))
        aspp3_decoder_output = self.decoder(*aspp3_feats)
        aspp3_masks = self.seg_head(aspp3_decoder_output)

        return aspp3_masks


class EffiUnet_ASPP3_DDPT_MLDS(torch.nn.Module):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_chns: A number of input channels for the model, default is 3 (RGB images)
        class_num: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: Unet
    """

    def __init__(
        self,
        encoder_name: str = "tu-tf_efficientnetv2_b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_chns: int = 3,
        class_num: int = 1,
        activation: Optional[Union[str, callable]] = None,
        rand_route:bool = False,
        rand_route_thres:float=0.5
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_chns,
            depth=encoder_depth,
            weights=encoder_weights,
        ) # [224@3, 112@16, 56@32, 28@48, 14@112, 7@192]
        self.encode_out_chan = self.encoder.out_channels

        self.rand_route = rand_route
        self.rand_route_thres = rand_route_thres

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.aspp3)
        init.initialize_decoder(self.aspp3_decoder)
        init.initialize_decoder(self.DDPT)
        init.initialize_decoder(self.DDPT_decoder)

        init.initialize_head(self.aspp3_seg_head)
        init.initialize_head(self.DDPT_seg_head)


    def forward(self, x, aspp_out=True, ddpp_out=True):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        aspp3_feats = features[:-2]
        DDPT_feats = features[:-2]

        aspp3_feats.extend(self.aspp3(x14=features[-2], x7=features[-1]))
        DDPT_feats.extend(self.DDPT(x14=features[-2], x7=features[-1]))

        aspp3_decoder_output, aspp3_mlds_outs = self.aspp3_decoder(*aspp3_feats)
        DDPT_decoder_output, DDPT_mlds_outs = self.DDPT_decoder(*DDPT_feats)

        aspp3_masks = self.aspp3_seg_head(aspp3_decoder_output)
        DDPT_masks = self.DDPT_seg_head(DDPT_decoder_output)

        return aspp3_masks, DDPT_masks,\
               aspp3_mlds_out7, aspp3_mlds_out14, aspp3_mlds_out28, aspp3_mlds_out56, aspp3_mlds_out112, aspp3_mlds_out224, \
               DDPT_mlds_out7, DDPT_mlds_out14, DDPT_mlds_out28, DDPT_mlds_out56, DDPT_mlds_out112, DDPT_mlds_out224



if __name__ == "__main__":
  pass
