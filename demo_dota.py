import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import ctrbox_net

from utils.func_utils import decode_prediction, non_maximum_suppression, preprocess, draw_results
from utils.decoder import DecDecoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# dota
num_classes = 15
input_h, input_w = 608, 608
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
    'helicopter'
]


def load_model():
    # create model
    heads = {
        'hm': num_classes,
        'wh': 10,
        'reg': 2,  # offset
        'cls_theta': 1,  # orientation cls
    }
    model = ctrbox_net.CTRBOX(heads,
                              pretrained=False, down_ratio=down_ratio,
                              final_kernel=1, head_channels=256)
    # load param
    resume = 'runs/dota/model_50.pth'  # dota demo
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=True)
    print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))

    return model.eval().cuda()


@torch.no_grad()
def infer(image):
    pr_decs = model(image)
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


if __name__ == '__main__':
    decoder = DecDecoder(K=500, conf_thresh=0.18, num_classes=num_classes)
    model = load_model()

    img_dir = 'data/dota'
    for img in os.listdir(img_dir):
        print(img)
        ori_img = cv2.imread(os.path.join(img_dir, img))
        ori_h, ori_w, _ = ori_img.shape

        img = preprocess(ori_img, input_w, input_h).cuda()

        results = infer(img)

        img = draw_results(results, ori_img)

        plt.figure(figsize=(10, 10))
        plt.imshow(img[:, :, ::-1])
        plt.show()
