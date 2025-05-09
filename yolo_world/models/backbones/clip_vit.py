from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmdet.utils import OptMultiConfig
import torch
import clip
from typing import Sequence, Tuple


@MODELS.register_module()
class CLIPViTBackbone(BaseModule):
    """CLIP中的预训练ViT主干网络，适配ViT-B/32结构，输出多尺度特征"""

    def __init__(self,
                 model_name: str = "ViT-B/32",
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_modules: Sequence[str] = ('transformer.resblocks.0', 'transformer.resblocks.1'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        # 加载CLIP模型并提取视觉部分（ViT）
        self.clip_model, _ = clip.load(model_name, device="cpu")
        self.vit = self.clip_model.visual
        self.out_indices = out_indices
        self.frozen_modules = frozen_modules
        self._freeze_modules()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # 调整输入尺寸至224x224
        if x.shape[-2:] != torch.Size([224, 224]):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )

        features = []

        # ViT前向过程
        x = self.vit.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, width, grid**2]
        x = x.permute(0, 2, 1)  # [*, grid**2, width]
        x = torch.cat([self.vit.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
                      dim=1)  # [*, grid**2+1, width]
        x = x + self.vit.positional_embedding.to(x.dtype)
        x = self.vit.ln_pre(x)

        # 提取中间层特征
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
        for i, blk in enumerate(self.vit.transformer.resblocks):
            x = blk(x)
            if i in self.out_indices:
                features.append(x.permute(1, 0, 2))  # 恢复为 [batch_size, seq_len, dim]

        return tuple(features)

    def _freeze_modules(self):
        for name, param in self.vit.named_parameters():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    param.requires_grad = False

    def get_preprocessor(self):
        """返回CLIP的预处理参数（与ResNet相同）"""
        return {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
            'size': 224
        } 