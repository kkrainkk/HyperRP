import torch
import torch.nn as nn


class VitSpatialConverter(nn.Module):
    def __init__(self):
        super().__init__()

        # 分支1: (16, 50, 768) → (16, 256, 80, 80)
        self.proj1 = nn.Linear(768, 256)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(4, 2), stride=(4, 2)),  # 5x10 → 20x20
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4),  # 20x20 → 80x80
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # 分支2: (16, 50, 768) → (16, 512, 40, 40)
        self.proj2 = nn.Linear(768, 512)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 4), stride=(8, 4)),  # 5x10 → 40x40
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # 分支3: (16, 50, 768) → (16, 512, 20, 20)
        self.proj3 = nn.Linear(768, 512)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 2), stride=(4, 2)),  # 5x10 → 20x20
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, inputs: list) -> list:
        outputs = []
        for i, x in enumerate(inputs):
            B, N, C = x.shape

            # 通道投影 & 空间重塑
            if i == 0:  # 分支1: 256通道 → 80x80
                x_proj = self.proj1(x)  # (B,50,256)
                x_2d = x_proj.view(B, 5, 10, 256).permute(0, 3, 1, 2)  # (B,256,5,10)
                x_out = self.deconv1(x_2d)  # (B,256,80,80)

            elif i == 1:  # 分支2: 512通道 → 40x40
                x_proj = self.proj2(x)  # (B,50,512)
                x_2d = x_proj.view(B, 5, 10, 512).permute(0, 3, 1, 2)  # (B,512,5,10)
                x_out = self.deconv2(x_2d)  # (B,512,40,40)

            elif i == 2:  # 分支3: 512通道 → 20x20
                x_proj = self.proj3(x)  # (B,50,512)
                x_2d = x_proj.view(B, 5, 10, 512).permute(0, 3, 1, 2)  # (B,512,5,10)
                x_out = self.deconv3(x_2d)  # (B,512,20,20)

            outputs.append(x_out)

        return outputs  # 返回列表形式的输出