"""
地物检测，17类目标检测
"""
import torch
import cv2
import matplotlib.pyplot as plt
from models.ctrbox_net import CTRBOX
from tqdm import tqdm

from utils.func_utils import decode_prediction, non_maximum_suppression, preprocess, draw_results
from utils.decoder import DecDecoder
from utils.misc import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# dota
dec_classes = 17
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


if __name__ == '__main__':
    # img_dir = 'data/dota'
    img_dir = 'data/geo_hazard/6_汽车误入'

    dec_model = load_dec_model()
    decoder = DecDecoder(K=500, conf_thresh=0.1, num_classes=dec_classes)

    for img in tqdm(os.listdir(img_dir)):
        if img == '@eaDir' or img.endswith('seg.png') or img.endswith('dec.png'):  # 跳过 dec/seg 结果
            continue

        print(img)

        # preprocess
        ori_image = cv2.imread(os.path.join(img_dir, img))
        ori_h, ori_w, _ = ori_image.shape
        image = preprocess(ori_image, input_w, input_h).cuda()

        # # detect
        results = detect(image)
        dec_img = draw_results(results, ori_image)

        plt.imshow(dec_img[:, :, ::-1])
        plt.show()

        # cv2.imwrite(f'results/{img}', dec_img)
