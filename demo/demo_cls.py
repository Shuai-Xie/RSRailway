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
import cv2
import subprocess
import matplotlib.pyplot as plt

from PIL import Image
from datasets.config.railway import cls_label_names
from torchvision.models.resnet import resnet50
from utils.misc import recover_color_img
from utils.gaode import get_img_info
from constants import project_dir


def load_model():
    model = resnet50(pretrained=False, num_classes=46)
    # ckpt = torch.load('runs/railway/cls_resnet50_bs128_Oct25_141839/model_best.pth')
    ckpt = torch.load('runs/railway/cls_resnet50_bs128_Oct25_204533/model_best.pth')
    model.load_state_dict(ckpt['state_dict'])
    print('load model, epoch: {}, acc: {}'.format(ckpt['epoch'], ckpt['acc']))
    return model.cuda().eval()


test_trans = tr.Compose([
    tr.ToTensor(),
    tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# numpy 元素映射函数，可得到 str
translate = np.vectorize(lambda t: cls_label_names[t])


def get_resize_size(x, crop_size):
    """
    返回 crop_size 整数倍长度
    """
    quo, remain = x // crop_size, x % crop_size
    if remain >= x * 2 / 3:
        return crop_size * (quo + 1)
    else:
        return crop_size * quo


@torch.no_grad()
def sliding_predict(model, img, crop_size=224, overlap=0, return_station_patches=False):
    """
    :param model:
    :param img: PIL Image
    :param crop_size:
    :param overlap:
    :param return_station_patches:
    :return:
    """
    W, H = img.size
    img = img.resize((get_resize_size(W, crop_size), get_resize_size(H, crop_size)))
    img = test_trans(img)
    img = img.unsqueeze(0).cuda()
    _, _, H, W = img.shape
    print(img.shape)

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

    # 解析网格场景分类结果
    preds = preds.reshape((tile_rows, tile_cols))
    preds = translate(preds)
    df = pd.DataFrame(preds)

    if return_station_patches:
        preds = preds.flatten()
        station_idxs = np.where(preds == 'railway_station')[0]
        patches = []
        for idx in station_idxs:
            patches.append(recover_color_img(tile_imgs[idx]))
        return df, patches
    else:
        return df


CH2EN_citys = {
    '北京市': 'beijing',
    '广州市': 'guangzhou',
    '合肥市': 'hefei',
    '济南市': 'jinan',
    '青岛市': 'qingdao',
    '石家庄市': 'shijiazhuang',
    '太原市': 'taiyuan',
    '西安市': 'xian'
}

# opencv-contrib-python
# https://blog.csdn.net/Eddy_zheng/article/details/78916009
sift = cv2.SIFT_create(nfeatures=400)  # surf is patented, can't use
stations = ['北站', '东站', '西站', '南站']


def station_recognition(img, location, vis=False):
    # city info and name
    city = get_img_info(location)['city']
    if isinstance(city, list):  # 直辖市会返回空 list []
        city = get_img_info(location)['province']
    city_en = CH2EN_citys[city]

    # img to gray, for matching
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    match = {
        'kps': 0,
        'name': '',
        'img': None
    }

    for i in range(1, 5):  # 1-4 NEWS
        img_path = f'data/stations/{city_en}/{i}.png'
        if not os.path.exists(img_path):
            continue
        # 匹配的 基准 img
        base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = base_img.shape
        base_img = cv2.resize(base_img, dsize=(400, int(400 * h / w)))

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img, None)
        kp2, des2 = sift.detectAndCompute(base_img, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
        match_pts = len(good)
        if match_pts > match['kps']:
            match = {
                'kps': match_pts,
                'name': stations[i - 1],
                'img': cv2.drawMatchesKnn(img, kp1, base_img, kp2, good, None, flags=2)
            }

    if vis and match['img'] is not None:
        plt.imshow(match['img'])
        plt.show()

    return city.replace('市', '') + match['name']


def station_recognition_cpp(img, location):
    """
    install opencv
        https://blog.csdn.net/heyijia0327/article/details/54575245
    .bashrc 指定 opencv 2.4.9 路径
        export PKG_CONFIG_PATH=/nfs/xs/local/opencv249/lib//pkgconfig
        export LD_LIBRARY_PATH=/nfs/xs/local/opencv249/lib/
    查看指定后 opencv 版本
        pkg-config --modversion opencv
    fatal error: opencv2/nonfree/nonfree.hpp, 依赖 opencv 2.4.9
        https://blog.csdn.net/wuzuyu365/article/details/52329732
    """
    # city info and name
    city = get_img_info(location)['city']
    if isinstance(city, list):  # 直辖市会返回空 list []
        city = get_img_info(location)['province']
    city = CH2EN_citys[city]

    # save for `surfnew` reading
    cv2.imwrite(f'{project_dir}/Match/{city}.png', img[:, :, ::-1])  # to BGR

    # exe city_dir test_img
    cmd = f'./Match/surfnew {city} {city}.png'

    try:
        out_bytes = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,
                                     executable='/bin/bash').stdout.read()
        out_text = out_bytes.decode('utf-8').replace('\n', '')
        info = {
            'city': city.replace('市', ''),
            'station': out_text,  # result image name
        }
        print(info)
        return info
    finally:
        return 'error'


def demo_weather():
    data_dir = 'data/geo_hazard/7_恶劣天气'
    for img_name in os.listdir(data_dir):
        print(img_name)
        image = Image.open(os.path.join(data_dir, img_name))
        info = sliding_predict(model, image)
        print(info)


def demo_station():
    data_dir = 'data/geo_hazard/8_车站定位'
    for img_name in os.listdir(data_dir):
        print(img_name)
        image = Image.open(os.path.join(data_dir, img_name))
        info, station_imgs = sliding_predict(model, image, return_station_patches=True)

        stations = []
        locations = [
            [113.27, 23.13],  # 广州
            [116.41, 39.91],  # 北京
        ]
        for idx, img in enumerate(station_imgs):  # RGB 通道图片
            stations.append(station_recognition(img, locations[idx], vis=True))
            # plt.imshow(img)
            # plt.show()
        print(stations)


if __name__ == '__main__':
    model = load_model()
    # demo_weather()
    demo_station()
