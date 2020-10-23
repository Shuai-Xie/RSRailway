import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.ctrbox_net import CTRBOX
from models.ctrseg_net import CTRSEG

from utils.func_utils import decode_prediction, non_maximum_suppression, preprocess, draw_results
from utils.decoder import DecDecoder
from utils.misc import *

import datasets.rs_segment.transforms as tr

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# dota
dec_classes = 17
seg_classes = 7
input_w, input_h = 960, 540
down_ratio = 4

category = [
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter',
    'train',
    'rail',
]

label_names = ['bg', 'rail', 'plant', 'buildings', 'road', 'land', 'water', 'train']
label_colors = [
    (0, 0, 0),
    (0, 0, 255),  # 铁轨
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),  # road 公路   粉
    (255, 255, 0),  # land 黄土地  黄
    (0, 255, 255),  # water
    (128, 128, 128),  # train
]


def load_dec_model():
    # create model
    heads = {
        'hm': dec_classes,
        'wh': 10,
        'reg': 2,  # offset
        'cls_theta': 1,  # orientation cls
    }

    model = CTRBOX(heads,
                   pretrained=False, down_ratio=down_ratio,
                   final_kernel=1, head_channels=256)
    # load param
    resume = 'runs/railway/dec_res101_epoch100_data1501_Oct22_143548/model_best.pth'
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=True)
    print('loaded dec model from {}, epoch {}'.format(resume, checkpoint['epoch']))

    return model.eval().cuda()


def load_seg_model():
    model = CTRSEG(num_classes=seg_classes, pretrained=False)
    resume = 'runs/railway/seg_cls7_res101_data100_Oct22_144156/model_best.pth'
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    model.load_state_dict(state_dict_, strict=True)
    print('loaded seg model from {}, epoch {}'.format(resume, checkpoint['epoch']))

    return model.eval().cuda()


@torch.no_grad()
def detect(image):
    pr_decs = dec_model(image)

    # heatmap point nms + topK + conf_thresh + HBB/RBB 解析
    predictions = decoder.ctdet_decode(pr_decs)  # np -> 1,num_obj,12 = 2+8+1+1
    # 解析 predictions 得到 dict 类型结果
    cat_pts, cat_scores = decode_prediction(predictions, category, input_w, input_h, ori_w, ori_h, down_ratio)

    results = {cat: [] for cat in category}

    # multi-label nms 逐类 nms
    for cat in category:
        pts, scores = cat_pts[cat], cat_scores[cat]
        pts = np.asarray(pts, np.float32)
        scores = np.asarray(scores, np.float32)

        if pts.shape[0]:  # 存在 obj
            nms_results = non_maximum_suppression(pts, scores)
            results[cat].extend(nms_results)

    return results


@torch.no_grad()
def segment(image):
    res = seg_model(image).argmax(dim=1)
    res = res.cpu().numpy().squeeze()
    return res + 1  # to color


if __name__ == '__main__':
    # img_dir = 'data/dota'
    img_dir = 'data/geo_hazard'
    # img_dir = '/datasets/rs_segment/railway/train/images'
    # img_dir = '/datasets/rs_detect/railway/train/images'

    dec_model = load_dec_model()
    decoder = DecDecoder(K=500, conf_thresh=0.18, num_classes=dec_classes)

    seg_model = load_seg_model()
    test_trans = tr.get_test_transfrom()

    remap_fn = tr.remap(bg_idx=0)

    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img in os.listdir(img_dir):
        if img == '@eaDir':
            continue

        print(img)

        # preprocess
        ori_image = cv2.imread(os.path.join(img_dir, img))
        ori_h, ori_w, _ = ori_image.shape
        image = preprocess(ori_image, input_w, input_h).cuda()

        # # detect
        results = detect(image)
        dec_img = draw_results(results, ori_image)

        # segment
        seg_img = segment(image)
        plt_img_target(dec_img[:, :, ::-1], seg_img, label_colors,
                       title=img,
                       # save_path=f'{save_dir}/{img}'
                       )
        # plt_img_target(ori_image[:, :, ::-1], seg_img, label_colors)
