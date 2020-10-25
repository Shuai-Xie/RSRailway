"""
地貌分割，7类语义分割
"""
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import torch
from models.ctrseg_net import CTRSEG

from datasets.config.railway import seg_label_colors
from utils.func_utils import preprocess
from utils.misc import *

seg_classes = 7
input_w, input_h = 960, 540


def load_seg_model():
    model = CTRSEG(num_classes=seg_classes, pretrained=False)
    predume = 'runs/railway/seg_cls7_res101_data100_Oct22_144156/model_best.pth'
    checkpoint = torch.load(predume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    model.load_state_dict(state_dict_, strict=True)
    print('loaded seg model from {}, epoch {}'.format(predume, checkpoint['epoch']))

    return model.eval().cuda()


@torch.no_grad()
def segment(model, image):
    pred = model(image).argmax(dim=1)
    pred = pred.cpu().numpy().squeeze().astype(np.uint8)
    return pred + 1  # to cls idx


def demo_dir():
    img_dir = 'data/geo_hazard/5_异物侵线'
    save_dir = 'data/geo_hazard/5_异物侵线'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = load_seg_model()

    for img in os.listdir(img_dir):
        if img == '@eaDir':
            continue

        print(img)

        # preprocess
        ori_image = cv2.imread(os.path.join(img_dir, img))
        image = preprocess(ori_image, input_w, input_h).cuda()

        # segment
        pred = segment(model, image)
        seg_img = color_code_target(pred, label_colors=seg_label_colors)
        plt_img_target(ori_image[:, :, ::-1], seg_img,
                       title=img,
                       # save_path=f'{save_dir}/{img}'
                       )
        cv2.imwrite(f'{save_dir}/{img[:-4]}_seg.png', seg_img[:, :, ::-1])


def demo_img():
    img_dir = 'data/geo_hazard/5_异物侵线'

    model = load_seg_model()

    for img in os.listdir(img_dir):
        if img.startswith('3') or img.startswith('4'):
            print(img)
            ori_image = cv2.imread(os.path.join(img_dir, img))
            image = preprocess(ori_image, input_w, input_h).cuda()

            # segment
            pred = segment(model, image)
            cv2.imwrite(os.path.join(img_dir, img.replace('.png', '_class.png')), pred)
            cv2.imwrite(os.path.join(img_dir, img.replace('.png', '_seg.png')), color_code_target(pred, label_colors)[:, :, ::-1])


if __name__ == '__main__':
    # demo_dir()
    demo_img()
    pass
