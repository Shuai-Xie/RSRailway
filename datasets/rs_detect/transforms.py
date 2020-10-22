import numpy as np
import cv2


def random_flip(image, gt_pts, crop_center=None):
    """
    :param image: h x w x c
    :param gt_pts: num_obj x 4 x 2
    :param crop_center:
    :return:
    """
    h, w, c = image.shape
    # horizontal
    if np.random.random() < 0.5:
        image = image[:, ::-1, :]
        if gt_pts.shape[0]:
            gt_pts[:, :, 0] = w - 1 - gt_pts[:, :, 0]
        if crop_center is not None:
            crop_center[0] = w - 1 - crop_center[0]
    # vertical
    if np.random.random() < 0.5:
        image = image[::-1, :, :]
        if gt_pts.shape[0]:
            gt_pts[:, :, 1] = h - 1 - gt_pts[:, :, 1]
        if crop_center is not None:
            crop_center[1] = h - 1 - crop_center[1]
    return image, gt_pts, crop_center


def _get_border(size, border):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def random_crop_info(h, w, border_ratio):
    """
    :return:
        random cropsize 原尺寸 * random scale [0.9, 1.0, 1.1]
        random center 设置 border 随机取中心
    """
    # h,w: ori image size
    if np.random.random() < 1.0:
        # random scale
        random_scale = np.random.choice(np.arange(0.8, 1.2, 0.1))
        # random_scale = 1.
        random_crop_w = w * random_scale
        random_crop_h = h * random_scale

        # random center
        w_border = _get_border(size=w, border=w * border_ratio)
        h_border = _get_border(size=h, border=h * border_ratio)
        random_center_x = np.random.randint(low=w_border, high=w - w_border)  # border 之内取 center
        random_center_y = np.random.randint(low=h_border, high=h - h_border)
        return [random_crop_w, random_crop_h], [random_center_x, random_center_y]
    else:
        return None, None


def Rotation_Transform(src_point, degree):
    radian = np.pi * degree / 180
    R_matrix = [[np.cos(radian), -np.sin(radian)],  # 逆时针, sin(θ) 控制
                [np.sin(radian), np.cos(radian)]]
    R_matrix = np.asarray(R_matrix, dtype=np.float32)
    R_pts = np.matmul(R_matrix, src_point)
    return R_pts


def get_3rd_point(a, b):
    """
    :param a: (center_x, center_y) 中心点
    :param b: (center_x, center_y - crop_size // 2) 中心上方边界
    :return: (center_x - crop_size // 2, center_y - crop_size // 2) 左上角
    """
    direct = a - b  # (0, crop_size//2) ?
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def load_affine_matrix(crop_center, crop_size, dst_size, inverse=False, rotation=False):
    """
    :param crop_center: border 范围内随机取的中心点
    :param crop_size: 切图大小, [0.9, 1.0, 1.1] 倍原图
    :param dst_size: model input size
    :param inverse:
    :param rotation: True
    :return:
    """
    dst_center = np.array([dst_size[0] // 2, dst_size[1] // 2], dtype=np.float32)
    max_degree = 10

    if rotation and np.random.rand(1) > 0.5:
        random_degree = np.random.rand(1)[0] * max_degree
    else:
        random_degree = 0.

    # compute affine matrix
    src_1 = crop_center
    src_2 = crop_center + Rotation_Transform([0, -crop_size[0] // 2], degree=random_degree)  # 左边中点
    src_3 = get_3rd_point(src_1, src_2)
    src = np.asarray([src_1, src_2, src_3], np.float32)

    dst_1 = dst_center
    dst_2 = dst_center + [0, -dst_center[0]]
    dst_3 = get_3rd_point(dst_1, dst_2)
    dst = np.asarray([dst_1, dst_2, dst_3], np.float32)
    if inverse:
        M = cv2.getAffineTransform(dst, src)
    else:
        M = cv2.getAffineTransform(src, dst)
    return M


def ex_box_jaccard(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
    inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
    inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
    inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
    x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
    y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
    y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
    mask_w = np.int(np.ceil(x2 - x1))
    mask_h = np.int(np.ceil(y2 - y1))
    mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    a[:, 0] -= x1
    a[:, 1] -= y1
    b[:, 0] -= x1
    b[:, 1] -= y1
    mask_a = cv2.fillPoly(mask_a, pts=np.asarray([a], 'int32'), color=1)
    mask_b = cv2.fillPoly(mask_b, pts=np.asarray([b], 'int32'), color=1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter) / (float(union) + 1e-12)
    # cv2.imshow('img1', np.uint8(mask_a*255))
    # cv2.imshow('img2', np.uint8(mask_b*255))
    # k = cv2.waitKey(0)
    # if k==ord('q'):
    #     cv2.destroyAllWindows()
    #     exit()
    return iou
