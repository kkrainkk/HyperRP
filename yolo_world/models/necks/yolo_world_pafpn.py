# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig
from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
from .Custom6 import SingleScaleTransformerWithHypergraph
from .Custom8 import FeatureFusionWithTextUpdateModule
#from .Custom import CustomFPN
from .HyperG import HypergraphLearningModule
from .Back import HyperFPN
from .txt_attention import TextGuidedAttention
from typing import List, Tuple
from torch import Tensor
import torch.nn.init as init
from .trans_shape import FeatureShapeAdapter
from .Seqtospace import VitSpatialConverter


@MODELS.register_module()
class YOLOWorldPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.reshape = FeatureShapeAdapter()
        self.Seqtospace = VitSpatialConverter()
        # 定义自定义 FPN 模块，保持通道和空间尺寸不变
        self.fpn = HyperFPN(in_channels_list=[256, 512, 512])

        #self.fpn = CustomFPN(in_channels_list=[256, 512, 512])
        # 创建自定义模块实例，输入和输出通道数为 512
        #self.custom_module2 = SingleScaleTransformerWithHypergraph(in_channels=512)
        self.fuse = FeatureFusionWithTextUpdateModule([256,512,512], text_channels=512, embed_dim=256)
        self.Hyper = HypergraphLearningModule(256)


    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx - 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx - 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx - 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)


    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx + 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx + 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx + 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    '''
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)

    '''

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function."""

        # print(img_feats[0].shape)
        # print(img_feats[1].shape)
        # print(img_feats[2].shape)
        assert len(img_feats) == len(self.in_channels)

        # Step 1: 使用 reduce_layers 对图像特征进行降维
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # Step 2: 融合特征图并进行超图构建
        new_feature, txt_feats = self.fuse(reduce_outs, txt_feats)  # 筛选后的特征图

        # Step 3: 构建超图特征图
        Hyper_feature = self.Hyper(new_feature, txt_feats)  # 构建超图的特征图

        # Step 5: 将更新后的特征图送入自定义 FPN 进行融合
        fpn_features = self.fpn(reduce_outs, Hyper_feature)

        # Step 6: 使用 out_layers 对 FPN 输出的特征图进行处理
        results = []
        for idx, feat in enumerate(fpn_features):  # 直接遍历 list
            feat = self.out_layers[idx](feat)
            results.append(feat)

        return tuple(results)





@MODELS.register_module()
class YOLOWorldDualPAFPN(YOLOWorldPAFPN):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 text_enhancder: ConfigType = dict(
                     type='ImagePoolingAttentionModule',
                     embed_channels=256,
                     num_heads=8,
                     pool_size=3),
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         guide_channels=guide_channels,
                         embed_channels=embed_channels,
                         num_heads=num_heads,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         block_cfg=block_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)


        text_enhancder.update(
            dict(
                image_channels=[int(x * widen_factor) for x in out_channels],
                text_channels=guide_channels,
                num_feats=len(out_channels),
            ))
        print(text_enhancder)
        self.text_enhancer = MODELS.build(text_enhancder)

        # 创建自定义模块实例，输入和输出通道数为 512
        #self.custom_module2 = CustomMultiScaleModule(in_channels=512, out_channels=512, H=20, W=20)
        #self.custom_module1 = CustomMultiScaleModule(in_channels=512, out_channels=512, H=40, W=40)
        #self.custom_module0 = CustomMultiScaleModule(in_channels=256, out_channels=256, H=80, W=80)
        # 使用全局关系
        #self.custom_module2 = CustomMultiScaleModule(in_channels=512, out_channels=512,
                                           # use_local_graph=False)

        # 使用局部关系
       # self.custom_module1 = CustomMultiScaleModule(in_channels=512, out_channels=512,
                                                      #use_local_graph=True, local_ks=5)

        #self.custom_module0 = CustomMultiScaleModule(in_channels=256, out_channels=256,
                                                     # use_local_graph=True, local_ks=5)
        # 定义自定义 FPN 模块，保持通道和空间尺寸不变
        #self.fpn = CustomFPN(in_channels_list=[256, 512, 512])


    '''
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function."""
        assert len(img_feats) == len(self.in_channels)
        # reduce layers

        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        txt_feats = self.text_enhancer(txt_feats, inner_outs)
        print("text feature shape",txt_feats.shape)
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

            # 打印每个特征图的形状
        for idx, feat in enumerate(outs):
            print(f"Feature map {idx} shape: {feat.shape}")

        #map2_out = self.custom_module2(inner_outs[2])  # 直接传入单一特征图
        #map1_out = self.custom_module1(inner_outs[1])
        #map0_out = self.custom_module0(inner_outs[0])

        # 组合其他两个特征图与 map2_out
        outs = [inner_outs[0], inner_outs[1], inner_outs[2]]

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
    '''


    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function."""


        assert len(img_feats) == len(self.in_channels)
        # reduce layers

        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        txt_feats = self.text_enhancer(txt_feats, reduce_outs)  # 直接使用 reduce_outs 进行文本增强
        #print(txt_feats.shape)


        reduce_outs[2] = self.custom_module2(reduce_outs[2])
        #print(reduce_outs[2].shape)
        reduce_outs[1] = self.custom_module1(reduce_outs[1])
        #print(reduce_outs[1].shape)
        reduce_outs[0] = self.custom_module0(reduce_outs[0])
        #print(reduce_outs[0].shape)

        # 将更新后的特征图送入自定义 FPN 进行融合

        fpn_features = self.fpn(reduce_outs)
        #print("FPN Features:")
        #for idx, (key, feat) in enumerate(fpn_features.items()):
         #   print(f"FPN Feature map {idx} ({key}) shape: {feat.shape}")

        # out_layers
        results = []
        # 按键访问字典中的特征图
        for idx, key in enumerate(fpn_features.keys()):
            results.append(self.out_layers[idx](fpn_features[key]))

        return tuple(results)




