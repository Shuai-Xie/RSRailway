import torch.nn as nn
import numpy as np
import torch
from models.model_parts import CombinationModule
from models import resnet
import torch.nn.functional as F


class CTRSEG(nn.Module):
    """使用 ctrbox 结构，只修改 head 的分割网络"""

    def __init__(self, num_classes, pretrained):
        super(CTRSEG, self).__init__()

        self.base_network = resnet.resnet101(pretrained=pretrained)  # [C0,...,C5], [1, 1/32]
        # self.base_network = resnet.resnet50(pretrained=pretrained)
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)

        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.base_network(x)  # encoder feature
        c4_combine = self.dec_c4(x[-1], x[-2])  # 1/16
        c3_combine = self.dec_c3(c4_combine, x[-3])  # 1/8
        c2_combine = self.dec_c2(c3_combine, x[-4])  # 1/4

        # seg head
        x = self.seg_head(c2_combine)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x

    def get_train_params(self):
        # 只将 seg_head 加入模型训练
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p


if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512)
    model = CTRSEG(num_classes=6, pretrained=False)
    res = model(x)
    print(res.shape)
