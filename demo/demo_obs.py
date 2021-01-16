"""
地貌分割，7类语义分割
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import torch
from models.hrnet.hrnet import HRNet
from datasets.rs_segment.config import train_rail_config
import torchvision.transforms as tr
from demo.base import predict_sliding

from utils.func_utils import preprocess
from utils.misc import *
from utils.metric import SegmentationMetric
from utils.vis import plt_img_target_pred, plt_img_target
import random
from PIL import Image

seg_classes = 3
input_w, input_h = 960, 540
label_colors = [
    (0, 0, 0),  # bg=0; 作为 0 类
    (0, 0, 255),  # rail=1
    (0, 255, 0),  # train=2
    # (255, 0, 0),  # obstacle=3
]

trans_img = tr.Compose([
    tr.ToTensor(),
    tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_seg_model():
    model = HRNet(
        # cfg_path='models/hrnet/cfg/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
        # cfg_path='models/hrnet/cfg/seg_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        cfg_path='models/hrnet/cfg/seg_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        # cfg_path='models/hrnet/cfg/seg_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        num_classes=seg_classes,
        use_pretrain=False
    )
    # resume = 'runs/railway/seg_fg_scale_cls3_epoch30_hrnet_Nov13_194822/model_best.pth'
    # resume = 'runs/railway/seg_weighted_cls3_epoch30_hrnet_Nov13_163340/model_best.pth'
    resume = 'runs/railway/seg_hrnet18_fg_scale_cls3_epoch40_hrnet_Nov13_234125/model_best.pth'
    # resume = 'runs/railway/seg_hrnet30_fg_scale_cls3_epoch40_hrnet_Nov13_234309/model_best.pth'
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    model.load_state_dict(state_dict_)
    print('loaded seg model from {}, epoch {}'.format(resume, checkpoint['epoch']))

    return model.eval().cuda()


@torch.no_grad()
def segment(model, image):
    pred = model(image).argmax(dim=1)
    pred = pred.cpu().numpy().squeeze().astype(np.uint8)
    return pred


def demo_valid():
    img_dir = '/datasets/rs_detect/railway/train/images'
    msk_dir = '/datasets/rs_detect/railway/train/mask'

    model = load_seg_model()

    evaluator = SegmentationMetric(num_class=seg_classes)
    img_list = os.listdir(img_dir)
    random.shuffle(img_list)

    for img in img_list:
        if img == '@eaDir':
            continue
        print(img)
        ori_image = cv2.imread(os.path.join(img_dir, img), cv2.IMREAD_UNCHANGED)
        image = preprocess(ori_image, input_w, input_h).cuda()

        # segment
        pred = segment(model, image)
        target = cv2.imread(os.path.join(msk_dir, img), cv2.IMREAD_UNCHANGED)

        evaluator.reset()
        evaluator.add_batch(target, pred)

        acc = evaluator.Pixel_Accuracy()
        miou = evaluator.Mean_Intersection_over_Union()

        plt_img_target_pred(ori_image, target, pred,
                            label_colors,
                            figsize=(15, 4),
                            title='{} acc: {}, mIoU: {}'.format(img[:-4], acc, miou))


def demo_dir():
    img_dir = 'data/obstacle/img'
    msk_dir = 'data/obstacle/msk'
    res_dir = 'data/obstacle/res'

    model = load_seg_model()
    img_list = os.listdir(img_dir)

    for img in img_list:
        print(img)
        ori_image = Image.open(os.path.join(img_dir, img))

        image = trans_img(ori_image).unsqueeze(0).cuda()

        # segment
        pred = segment(model, image)
        # pred = predict_sliding(model, image, seg_classes, crop_size=500)
        # pred = pred.argmax(dim=1).cpu().numpy().squeeze()

        # plt
        plt_img_target(ori_image, pred, label_colors,
                       figsize=(12, 4),
                       save_path=os.path.join(res_dir, img[:-4] + '.png')
                       )

        # msk
        cv2.imwrite(os.path.join(msk_dir, img[:-4] + '.png'), pred)


def demo_img():
    # img_path = 'data/obstacle/img/P0756_hzd.png'
    img_path = 'data/obstacle/img/5_large.png'

    ori_image = Image.open(img_path)
    image = trans_img(ori_image).unsqueeze(0).cuda()

    model = load_seg_model()
    pred = segment(model, image)
    # plt
    plt_img_target(ori_image, pred, label_colors,
                   figsize=(12, 4),
                   save_path='data/obstacle/res/5_large.png'
                   )
    # msk
    cv2.imwrite('data/obstacle/msk/5_large.png', pred)


if __name__ == '__main__':
    demo_dir()
    # demo_img()
    pass
