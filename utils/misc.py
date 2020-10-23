import os, shutil
import time
import matplotlib.pyplot as plt
import numpy as np


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
