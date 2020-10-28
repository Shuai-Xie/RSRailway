import os, shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import torch


def dist_ab(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def lineABC(line_pts):
    x1, y1 = line_pts[0]
    x2, y2 = line_pts[1]

    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def box_major_axis(box):
    """
    获取检测 box 长轴，作为 rail/train 行驶方向
    """
    tl, tr, br, bl = box

    # 4边中点
    tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
    rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
    bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
    ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

    if dist_ab(tt, bb) > dist_ab(rr, ll):
        return lineABC([tt, bb])
    else:
        return lineABC([rr, ll])


def pt2line_distance(pts, line):
    """
    点到直线距离 line: 直线 A,B,C 公式
    """
    x0, y0 = pts
    A, B, C = line
    d = abs(A * x0 + B * y0 + C) / (A ** 2 + B ** 2) ** 0.5
    return d


# io: txt <-> list
def write_list_to_txt(a_list, txt_path):
    with open(txt_path, 'w') as f:
        for p in a_list:
            f.write(p + '\n')


def read_txt_as_list(f):
    with open(f, 'r') as f:
        return [p.replace('\n', '') for p in f.readlines()]


def mkdir(path, rm_exist=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if rm_exist:  # 先删再创建
            shutil.rmtree(path)
            os.makedirs(path)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time


def color_code_target(target, label_colors):
    return np.array(label_colors)[target.astype('int')]


def plt_img_target(img, target, label_colors=None, title=None, save_path=None):
    f, axs = plt.subplots(nrows=1, ncols=2, dpi=100)
    f.set_size_inches((16, 6))
    ax1, ax2 = axs.flat[0], axs.flat[1]

    # ax1.axis('off')
    ax1.imshow(img)

    # ax2.axis('off')
    if label_colors is not None:
        target = color_code_target(target, label_colors)
    ax2.imshow(target)

    if title:
        plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)

    plt.show()


def plt_compare_seg(target, pred, label_colors, title=None, save_path=None):
    f, axs = plt.subplots(nrows=1, ncols=2, dpi=100)
    f.set_size_inches((16, 6))
    ax1, ax2 = axs.flat[0], axs.flat[1]

    ax1.axis('off')
    ax1.imshow(color_code_target(target, label_colors))

    ax2.axis('off')
    ax2.imshow(color_code_target(pred, label_colors))

    if title:
        plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)

    plt.show()


def recover_color_img(img):
    """
    cvt tensor image to RGB [note: not BGR]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()

    img = np.transpose(img, axes=[1, 2, 0])  # h,w,c
    img = img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)  # 直接通道相成?
    img = (img * 255).astype('uint8')
    return img
