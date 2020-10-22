import torch.nn as nn
import numpy as np
import torch
from models.model_parts import CombinationModule
from models import resnet


class CTRBOX(nn.Module):
    """center rotated box"""

    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_channels):
        """
        :param heads: 4 heads
        :param pretrained: True
        :param down_ratio: 4
        :param final_kernel: final kernel size = 1
        :param head_channels: head conv chans
        """
        super(CTRBOX, self).__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))

        self.base_network = resnet.resnet101(pretrained=pretrained)  # [C0,...,C5], [1, 1/32]
        # self.base_network = resnet.resnet50(pretrained=pretrained)
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]  # 15/2/10/1
            if head == 'wh':  # final conv_kernel = 3 for wh regression
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_channels, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_channels),  # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_channels, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_channels, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_channels),  # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_channels, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)  # todo: tricks?
            else:
                self.fill_fc_weights(fc)

            # nn.Module 方法
            self.__setattr__(head, fc)  # name can be 'parameter/buffer/module'

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)  # encoder feature
        c4_combine = self.dec_c4(x[-1], x[-2])  # 1/16
        c3_combine = self.dec_c3(c4_combine, x[-3])  # 1/8
        c2_combine = self.dec_c2(c3_combine, x[-4])  # 1/4

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:  # ~[0,1]
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict


if __name__ == '__main__':
    heads = {
        'hm': 15,  # heatmap
        'reg': 2,  # offset
        'wh': 10,  # box param
        'cls_theta': 1,  # α
    }
    down_ratio = 4
    model = CTRBOX(heads=heads,
                   pretrained=True,
                   down_ratio=down_ratio,
                   final_kernel=1,
                   head_channels=256)

    x = torch.rand(1, 3, 608, 608)
    model(x)
