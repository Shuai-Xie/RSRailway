import os
import cv2
import shutil
from utils.misc import plt_img_target, mkdir, write_list_to_txt, read_txt_as_list
import random

root = '/datasets/rs_segment/railway'
label_names = ['bg', 'rail', 'plant', 'buildings', 'road', 'land', 'water']
label_colors = [
    (0, 0, 0),
    (0, 0, 255),  # 铁轨
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),  # road 公路   粉
    (255, 255, 0),  # land 黄土地  黄
    (0, 255, 255),  # water
]


def vis_data():
    img_dir = os.path.join(root, 'train', 'images')
    cls_dir = os.path.join(root, 'train', 'mask')
    img = cv2.imread(os.path.join(img_dir, '149.png'))[:, :, ::-1]
    target = cv2.imread(os.path.join(cls_dir, '149.png'), cv2.IMREAD_UNCHANGED)
    plt_img_target(img, target, label_colors)


def split_seg_data_dir():
    root = '/datasets/rs_segment/railway'
    img_dir = os.path.join(root, 'images')
    cls_dir = os.path.join(root, 'mask')
    vis_dir = os.path.join(root, 'vis')
    mkdir(img_dir)
    mkdir(cls_dir)
    mkdir(vis_dir)

    for img in os.listdir(root):
        if img.endswith('_class.png'):
            shutil.move(src=os.path.join(root, img), dst=os.path.join(cls_dir, img.replace('_class.png', '.png')))
        elif img.endswith('_visuable.png'):
            shutil.move(src=os.path.join(root, img), dst=os.path.join(vis_dir, img.replace('_visuable.png', '.png')))
        elif img.endswith('.png'):  # ori img, 过滤掉前2种情况
            shutil.move(src=os.path.join(root, img), dst=os.path.join(img_dir, img))


def cvt_4k_to_1k():
    data_dir = '/datasets/rs_segment/railway/train_4k'
    img_dir = os.path.join(data_dir, 'images')
    cls_dir = os.path.join(data_dir, 'mask')

    cvt_dir = '/datasets/rs_segment/railway/train'
    cvt_img_dir = os.path.join(cvt_dir, 'images')
    cvt_cls_dir = os.path.join(cvt_dir, 'mask')
    mkdir(cvt_img_dir)
    mkdir(cvt_cls_dir)

    dsize = (960, 540)

    for img_name in os.listdir(img_dir):
        if img_name == '@eaDir':
            continue
        img = cv2.imread(os.path.join(img_dir, img_name))
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)

        target = cv2.imread(os.path.join(cls_dir, img_name), cv2.IMREAD_UNCHANGED)
        target = cv2.resize(target, dsize, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(cvt_img_dir, img_name), img)
        cv2.imwrite(os.path.join(cvt_cls_dir, img_name), target)


def cvt_4k_to_1k_dir():
    img_dir = 'data/geo_hazard/5_异物侵线'
    dsize = (960, 540)

    for img_name in os.listdir(img_dir):
        if img_name.endswith('hzd.png'):
            img = cv2.imread(os.path.join(img_dir, img_name))
            img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(img_dir, img_name), img)


cvt_4k_to_1k_dir()


def save_train_paths():
    img_paths, msk_paths = [], []

    def add_paths(data_dir, shuffle=False):
        img_dir = os.path.join(data_dir, 'images')
        msk_dir = os.path.join(data_dir, 'mask')

        img_ids = [p for p in os.listdir(img_dir) if p != '@eaDir']
        if shuffle:
            random.shuffle(img_ids)
            img_ids = img_ids[:39]

        for img in img_ids:
            img_paths.append(os.path.join(img_dir, img))
            msk_paths.append(os.path.join(msk_dir, img))

    # ori 6 class
    add_paths(data_dir='/datasets/rs_segment/railway/train')
    print('num:', len(img_paths))  # 61
    # todo: rail/train 加入样本太多了
    add_paths(data_dir='/datasets/rs_detect/railway/train', shuffle=True)
    print('num:', len(img_paths))  # 1062

    write_list_to_txt(img_paths, '/datasets/rs_segment/railway/train/train_img_paths.txt')
    write_list_to_txt(msk_paths, '/datasets/rs_segment/railway/train/train_target_paths.txt')
