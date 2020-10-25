"""
地物检测，17类目标检测
"""
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import cv2
import matplotlib.pyplot as plt
from models.ctrbox_net import CTRBOX
from tqdm import tqdm

from datasets.config.railway import dec_label_names

from utils.func_utils import *
from utils.decoder import DecDecoder
from utils.misc import *
from pprint import pprint

# dota
dec_classes = 17
input_w, input_h = 960, 540
category = dec_label_names


def load_dec_model():
    # create model
    heads = {
        'hm': dec_classes,
        'wh': 10,
        'reg': 2,  # offset
        'cls_theta': 1,  # orientation cls
    }

    model = CTRBOX(heads,
                   pretrained=False, down_ratio=4,
                   final_kernel=1, head_channels=256)
    # load param
    resume = 'runs/railway/dec_res101_epoch100_data1501_Oct22_143548/model_best.pth'
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=True)
    print('loaded dec model from {}, epoch {}'.format(resume, checkpoint['epoch']))

    return model.eval().cuda()


@torch.no_grad()
def detect(model, image, decoder, input_w, input_h, ori_w, ori_h):
    pr_decs = model(image)

    # heatmap point nms + topK + conf_thresh + HBB/RBB 解析
    predictions = decoder.ctdet_decode(pr_decs)  # np -> 1,num_obj,12 = 2+8+1+1
    # 解析 predictions 得到 dict 类型结果
    cat_pts, cat_scores = decode_prediction(predictions, category, input_w, input_h, ori_w, ori_h, down_ratio=4)

    results = {cat: None for cat in category}

    # multi-label nms 逐类 nms
    for cat in category:
        pts, scores = cat_pts[cat], cat_scores[cat]
        pts = np.asarray(pts, np.float32)
        scores = np.asarray(scores, np.float32)

        if pts.shape[0]:  # 存在 obj
            results[cat] = non_maximum_suppression(pts, scores)  # n,9

    # 剩下的框统一 nms
    dets = np.zeros((0, 9))
    cats = []
    for cat, result in results.items():
        if result is None:
            continue
        dets = np.vstack((dets, result))
        cats += [cat] * result.shape[0]

    keep_index = py_cpu_nms_poly_fast(dets=dets, thresh=0.05)  # 0.1

    results = {cat: [] for cat in category}
    for idx in keep_index:
        # 对应类别添加对应 dec
        results[cats[idx]].append(dets[idx])

    return results


def demo_dir():
    img_dir = 'data/railway/img'
    # img_dir = 'data/geo_hazard/6_汽车误入'

    model = load_dec_model()
    decoder = DecDecoder(K=500, conf_thresh=0.18, num_classes=dec_classes)

    for img in tqdm(os.listdir(img_dir)):
        if img == '@eaDir' or img.endswith('seg.png') or img.endswith('dec.png'):  # 跳过 dec/seg 结果
            continue
        print(img)

        # preprocess
        ori_image = cv2.imread(os.path.join(img_dir, img))
        ori_h, ori_w, _ = ori_image.shape
        image = preprocess(ori_image, input_w, input_h).cuda()

        # detect
        results = detect(model, image, decoder,
                         input_w, input_h, ori_w, ori_h)

        # vis_plt
        plt_results(results, ori_image, vis=False, save_path=f'data/railway/dec_plt/{img}')

        # vis_cv
        dec_img = draw_results(results, ori_image)
        cv2.imwrite(f'data/railway/dec_cv/{img}', dec_img)


if __name__ == '__main__':
    demo_dir()
    pass
