import os, shutil
import time
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


