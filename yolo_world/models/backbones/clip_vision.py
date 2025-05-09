from typing import Sequence, Tuple
from mmengine.registry import MODELS
import torch
from torch import nn
import clip  # 确保导入 OpenAI 的官方 CLIP 库
from mmengine.model import BaseModule
from mmdet.utils import OptMultiConfig
from typing import Sequence, Tuple
from mmengine.registry import MODELS
import torch
from torch import nn
from mmengine.model import BaseModule
from mmdet.utils import OptMultiConfig


@MODELS.register_module()
class CLIPResNetBackbone(BaseModule):
    """CLIP中的预训练ResNet主干网络，适配ModifiedResNet结构，输出多尺度特征"""

    def __init__(self,
                 model_name: str = "RN50",
                 out_indices: Sequence[int] = (1, 2, 3, 4),
                 frozen_modules: Sequence[str] = ('layer1', 'layer2', 'layer3'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        # 加载CLIP模型并提取视觉部分（ModifiedResNet）
        self.clip_model, _ = clip.load(model_name, device="cpu")
        self.resnet = self.clip_model.visual

        self.out_indices = out_indices
        self.frozen_modules = frozen_modules
        self._freeze_modules()


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # 仅在第一次前向传播时打印权重信息
        if not hasattr(self, '_checked_weights'):
            # 检查第一个卷积层的权重
            conv1_weights = self.resnet.conv1.weight.data
            print("[DEBUG] Conv1 Weight Stats:")
            print("  Mean:", conv1_weights.mean().item())
            print("  Std :", conv1_weights.std().item())
            print("  Min :", conv1_weights.min().item())
            print("  Max :", conv1_weights.max().item())
            self._checked_weights = True  # 标记已检查

        # 调整输入尺寸至224x224
        if x.shape[-2:] != torch.Size([224, 224]):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )

        features = []

        # 处理stem部分（CLIP的ResNet包含三次卷积、BN和ReLU，最后是avgpool）
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu1(x)
        x = self.resnet.conv2(x)
        x = self.resnet.bn2(x)
        x = self.resnet.relu2(x)
        x = self.resnet.conv3(x)
        x = self.resnet.bn3(x)
        x = self.resnet.relu3(x)
        x = self.resnet.avgpool(x)  # 使用avgpool而非maxpool

        # 提取各层特征
        x = self.resnet.layer1(x)
        if 1 in self.out_indices:
            features.append(x)
        x = self.resnet.layer2(x)
        if 2 in self.out_indices:
            features.append(x)
        x = self.resnet.layer3(x)
        if 3 in self.out_indices:
            features.append(x)
        x = self.resnet.layer4(x)
        if 4 in self.out_indices:
            features.append(x)

        return tuple(features)

    def _freeze_modules(self):
        for name, param in self.resnet.named_parameters():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    param.requires_grad = False

    def get_preprocessor(self):
        """返回CLIP的预处理参数"""
        return {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
            'size': 224
        }



