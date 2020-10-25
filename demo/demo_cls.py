"""
地面网格场景分类，46类
    恶劣天气 云层检测
    地面车站定位
"""
import os
import sys

sys.path.insert(0, '/nfs/xs/Codes/BBAVectors-Oriented-Object-Detection')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import math
import torchvision.transforms as tr
import os
import pandas as pd

from PIL import Image
from datasets.config.railway import cls_label_names
from torchvision.models.resnet import resnet50


def load_model():
    model = resnet50(pretrained=False, num_classes=46)
    ckpt = torch.load('runs/railway/cls_resnet50_bs128_Oct25_141839/model_best.pth')
    model.load_state_dict(ckpt['state_dict'])
    print('load model, epoch: {}, acc: {}'.format(ckpt['epoch'], ckpt['acc']))
    return model.cuda().eval()


test_trans = tr.Compose([
    tr.ToTensor(),
    tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# numpy 元素映射函数，可得到 str
translate = np.vectorize(lambda t: cls_label_names[t])


@torch.no_grad()
def sliding_predict(model, img, crop_size=224, overlap=0):
    img = test_trans(img)
    img = img.unsqueeze(0).cuda()
    _, _, H, W = img.shape

    stride = int(math.ceil(crop_size * (1 - overlap)))  # overlap -> stride
    tile_rows = int(math.ceil((H - crop_size) / stride) + 1)
    tile_cols = int(math.ceil((W - crop_size) / stride) + 1)
    num_tiles = tile_rows * tile_cols
    print("Need %i x %i = %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, num_tiles, stride))

    tile_imgs = []

    for row in range(tile_rows):
        for col in range(tile_cols):
            # bottom-right / left-top 保证右下有效，反推左上
            x2, y2 = min(col * stride + crop_size, W), min(row * stride + crop_size, H)
            x1, y1 = max(int(x2 - crop_size), 0), max(int(y2 - crop_size), 0)
            tile_imgs.append(img[:, :, y1:y2, x1:x2])  # 1,3,h,w

    bs = 8
    nb, remain = num_tiles // bs, num_tiles % bs
    if remain > 0:
        nb += 1

    preds = np.zeros(0, dtype=int)
    for i in range(nb):
        batch_imgs = tile_imgs[i * bs:min((i + 1) * bs, num_tiles)]
        batch_imgs = torch.cat(batch_imgs, dim=0)

        output = model(batch_imgs)
        pred = output.argmax(dim=1)
        preds = np.hstack((preds, pred.cpu().numpy()))

    preds = preds.reshape((tile_rows, tile_cols))
    preds = translate(preds)

    df = pd.DataFrame(preds)
    print(df)
    return df


def demo_dir():
    data_dir = 'data/geo_hazard/7_恶劣天气'
    # data_dir = 'data/geo_hazard/8_车站定位'
    for img in os.listdir(data_dir):
        print(img)
        image = Image.open(os.path.join(data_dir, img))
        sliding_predict(model, image)


if __name__ == '__main__':
    model = load_model()
    demo_dir()
    pass
