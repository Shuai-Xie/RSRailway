import numpy as np
import matplotlib.pyplot as plt
import torch


def color_code_target(target, label_colors):
    return np.array(label_colors, dtype='uint8')[target.astype('uint8')]


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


def plt_img_target(img, target, label_colors=None, figsize=(10, 5), title=None, save_path=None):
    f, axs = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=figsize)
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


def plt_img_target_pred(img, target, pred, label_colors=None, figsize=(10, 5), title=None, save_path=None):
    f, axs = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=figsize)
    ax1, ax2, ax3 = axs.flat[0], axs.flat[1], axs.flat[2]

    # ax1.axis('off')
    ax1.imshow(img)

    # ax2.axis('off')
    if label_colors is not None:
        target = color_code_target(target, label_colors)
        pred = color_code_target(pred, label_colors)

    ax2.imshow(target)
    ax3.imshow(pred)

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
