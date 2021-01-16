import os
from tqdm import tqdm
import cv2
import random
from pprint import pprint
from utils.misc import *
import matplotlib.pyplot as plt


def save_data_path():
    """save image paths to *.txt"""
    data_dir = f'/datasets/rs_detect/DOTA/train_split'
    img_ids = [p[:-4] for p in os.listdir(f'{data_dir}/images') if p != '@eaDir']  # .png
    print('data num:', len(img_ids))
    write_list_to_txt(img_ids, txt_path=os.path.join(data_dir, f'train.txt'))


def check_ann_stats():
    # 查看数据集中 每张图像的 ann 数量分布
    data_dir = '/datasets/rs_detect/DOTA/val_split/labelTxt'
    # data_dir = '/datasets/rs_detect/railway/train/labelTxt'
    ann_stats = {}

    tbar = tqdm(os.listdir(data_dir))
    for txt in tbar:
        if txt == '@eaDir':
            continue
        anns = read_txt_as_list(os.path.join(data_dir, txt))
        num = len(anns)
        ann_stats[num] = ann_stats.get(num, 0) + 1

    pprint(ann_stats)


def compact_dota_data():
    """
    除去 labelTxt objs < 10 的数据
    train   del: 10917  save: 3431  total: 14348
    valid   del: 3920   save: 951   total: 4871
    """
    data_dir = '/datasets/rs_detect/DOTA/val_split/labelTxt'

    del_txt = []
    save_txt = []

    tbar = tqdm(os.listdir(data_dir))
    for txt in tbar:
        if txt == '@eaDir':
            continue
        anns = read_txt_as_list(os.path.join(data_dir, txt))
        if len(anns) < 10:
            del_txt.append(txt[:-4])  # 只存 id
        else:
            save_txt.append(txt[:-4])
    print('del:', len(del_txt))  #
    print('save:', len(save_txt))  #
    write_list_to_txt(del_txt, '/datasets/rs_detect/DOTA/val_split/del.txt')
    write_list_to_txt(save_txt, '/datasets/rs_detect/DOTA/val_split/train.txt')


def merge_dota_railway_path():
    dota_dir = '/datasets/rs_detect/DOTA/train_split'  # split 后图像
    railway_dir = '/datasets/rs_detect/railway/train/images'

    train_paths = []

    dota_img_ids = read_txt_as_list(f'{dota_dir}/train.txt')
    for img in dota_img_ids:
        train_paths.append(os.path.join(dota_dir, 'images', f'{img}.png'))
    print('data num:', len(train_paths))  # 3431

    random.shuffle(train_paths)
    train_paths = train_paths[:500]  # 取少量其他种类

    for img in os.listdir(railway_dir):
        if img == '@eaDir':
            continue
        train_paths.append(os.path.join(railway_dir, img))
    print('data num:', len(train_paths))  # 3431 + 1001 = 4432

    random.shuffle(train_paths)
    write_list_to_txt(train_paths, '/datasets/rs_detect/railway/train/train_path.txt')


def check_data():
    """有图片多余标注了 station 类"""
    data_dir = '/datasets/rs_detect/railway/train/labelTxt'
    img_ids = read_txt_as_list('/datasets/rs_detect/railway/train/train.txt')

    issue_imgs = set()

    category = {'rail', 'train'}

    for img in img_ids:
        anns = read_txt_as_list(os.path.join(data_dir, f'{img}.txt'))
        for ann in anns:
            obj = ann.split(' ')
            if obj[-2] not in category:
                issue_imgs.add(img)
                break
    print(issue_imgs)  # {'P0634', 'P0767', 'P0510'}


def clean_dota_split_data():
    # 移除 split 后 标注较少的子图
    data_dir = '/datasets/rs_detect/DOTA/val_split'
    del_imgs = read_txt_as_list(os.path.join(data_dir, 'del.txt'))

    print('before delete')
    print('imgs:', len(os.listdir(os.path.join(data_dir, 'images'))))
    print('txts:', len(os.listdir(os.path.join(data_dir, 'labelTxt'))))

    for img in tqdm(del_imgs):
        img_path = os.path.join(data_dir, 'images', f'{img}.png')
        txt_path = os.path.join(data_dir, 'labelTxt', f'{img}.txt')

        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)

    print('after delete')
    print('imgs:', len(os.listdir(os.path.join(data_dir, 'images'))))
    print('txts:', len(os.listdir(os.path.join(data_dir, 'labelTxt'))))


def mv_out_val_data():
    val_img_ids = read_txt_as_list('val_img_ids.txt')

    root = '/datasets/rs_detect/DOTA'

    train_img_dir = f'{root}/train_split/images'
    train_lbl_dir = f'{root}/train_split/labelTxt'

    val_dir = f'{root}/val_split'
    val_img_dir = f'{root}/val_split/images'
    val_lbl_dir = f'{root}/val_split/labelTxt'
    mkdir(val_dir)
    mkdir(val_img_dir)
    mkdir(val_lbl_dir)

    val_img_paths = [p for p in os.listdir(train_img_dir) if p[:5] in val_img_ids]
    for p in tqdm(val_img_paths):
        shutil.move(src=f'{train_img_dir}/{p}', dst=f'{val_img_dir}/{p}')
        shutil.move(src=f'{train_lbl_dir}/{p[:-4]}.txt', dst=f'{val_lbl_dir}/{p[:-4]}.txt')


def resize_imgs():
    root = 'data/railway'

    dsize = (960, 540)  # 1920*1080 / 2

    for img_name in os.listdir(root):
        img_path = os.path.join(root, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        print(img_name, img.shape)
        cv2.imwrite(img_path, img)


def demo_fillPoly():
    img = np.zeros((1080, 1920, 3), np.uint8)
    area1 = np.array([[250, 200], [300, 100], [750, 800], [100, 1000]])
    area2 = np.array([[1000, 200], [1500, 200], [1500, 400], [1000, 400]])

    # cv2.fillPoly(img, [area1, area2], (255, 255, 255))
    cv2.fillPoly(img, [area1], (255, 255, 255))  # 必须传入 list; 区域填充标签

    plt.imshow(img)
    plt.show()


def cvt_dec_to_seg_data():
    """将 detect 数据集 框出的 box 转化成 seg 数据集
    """
    root = '/datasets/rs_detect/railway/train'

    img_dir = os.path.join(root, 'images')
    lbl_dir = os.path.join(root, 'labelTxt')
    msk_dir = os.path.join(root, 'mask')

    img_ids = [p for p in os.listdir(img_dir) if p != '@eaDir']

    stats = {
        'rail': {
            'anns': 0,
            'imgs': 0,
        },
        'train': {
            'anns': 0,
            'imgs': 0,
        }
    }

    for img_id in tqdm(img_ids):
        img = cv2.imread(os.path.join(img_dir, img_id))
        h, w, _ = img.shape
        mask = np.zeros((h, w, 3), dtype=np.uint8)  # 为了 fillPoly 使用
        anns = read_txt_as_list(os.path.join(lbl_dir, img_id.replace('.png', '.txt')))

        rail_pts, train_pts = [], []
        for ann in anns:
            obj = ann.split(' ')
            cat = obj[-2]
            pts = np.array(list(map(int, obj[:8]))).reshape((4, 2))
            if cat == 'rail':
                rail_pts.append(pts)
            elif cat == 'train':
                train_pts.append(pts)

        # 先画 rail 再画 train; 防止 train 被 rail 覆盖 ?
        if len(rail_pts) > 0:
            cv2.fillPoly(mask, rail_pts, color=(1, 0, 0))
            stats['rail']['imgs'] += 1
            stats['rail']['anns'] += len(rail_pts)
        if len(train_pts) > 0:
            cv2.fillPoly(mask, train_pts, color=(2, 0, 0))
            stats['train']['imgs'] += 1
            stats['train']['anns'] += len(train_pts)

        mask = mask[:, :, 0]  # 取出第1维即可
        cv2.imwrite(os.path.join(msk_dir, img_id), mask)

    pprint(stats)


def cvt_4k_to_1k():
    """
    原始 railway 4k 图 转成 (960,540)
    """
    root = '/datasets/rs_detect/railway'

    # 4k dir
    raw_dir = os.path.join(root, 'train_4k')
    raw_img_dir = f'{raw_dir}/images'
    raw_lbl_dir = f'{raw_dir}/labelTxt'

    # 1k dir
    cvt_dir = os.path.join(root, 'train')
    cvt_img_dir = f'{cvt_dir}/images'
    cvt_lbl_dir = f'{cvt_dir}/labelTxt'
    mkdir(cvt_dir)
    mkdir(cvt_img_dir)
    mkdir(cvt_lbl_dir)

    img_ids = [p for p in os.listdir(raw_img_dir) if p != '@eaDir']
    # print(len(img_ids))

    target_h, target_w = 540, 960

    for img_name in tqdm(img_ids):
        img = cv2.imread(os.path.join(raw_img_dir, img_name))
        ori_h, ori_w, c = img.shape
        x_ratio = target_w / ori_w
        y_ratio = target_h / ori_h

        img = cv2.resize(img, dsize=(target_w, target_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(cvt_img_dir, img_name), img)

        # resize ann box
        anns = read_txt_as_list(os.path.join(raw_lbl_dir, img_name.replace('.png', '.txt')))
        cvt_anns = []

        for ann in anns:
            obj = ann.split(' ')
            x1 = int(float(obj[0]) * x_ratio)
            x2 = int(float(obj[2]) * x_ratio)
            x3 = int(float(obj[4]) * x_ratio)
            x4 = int(float(obj[6]) * x_ratio)

            y1 = int(float(obj[1]) * y_ratio)
            y2 = int(float(obj[3]) * y_ratio)
            y3 = int(float(obj[5]) * y_ratio)
            y4 = int(float(obj[7]) * y_ratio)

            cvt_ann = f'{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {obj[-2]} {obj[-1]}'
            cvt_anns.append(cvt_ann)

        write_list_to_txt(cvt_anns, os.path.join(cvt_lbl_dir, img_name.replace('.png', '.txt')))


if __name__ == '__main__':
    cvt_dec_to_seg_data()
