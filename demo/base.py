import numpy as np
import pandas as pd
from utils.misc import pt2line_distance
import cv2


# 输入颜色的字典，输出对应的种类
def train_kind(colordict):
    # print(colordict)
    blackvalue = 0
    brownvalue = 0
    whitevalue = 0
    grayvalue = 0
    threshold = 0.5
    if colordict.get('white'):
        whitevalue = colordict.get('white')
    if colordict.get('gray'):
        grayvalue = colordict.get('gray')
    if whitevalue + grayvalue > threshold:
        trainkind = '白色高铁'
        return trainkind
    if colordict.get('green'):
        greenvalue = colordict.get('green')
        if greenvalue > threshold:
            trainkind = '绿色普通客车'
            return trainkind
    if colordict.get('red'):
        redvalue = colordict.get('red')
        if redvalue > threshold:
            trainkind = '红色普通客车'
            return trainkind
    if colordict.get('red') or colordict.get('blue') or colordict.get('green'):
        trainkind = '多色集装箱货车'
        return trainkind
    if colordict.get('black'):
        blackvalue = colordict.get('black')
    if colordict.get('brown'):
        brownvalue = colordict.get('brown')
    if brownvalue + blackvalue + grayvalue > 0.6:
        trainkind = '棕黑色运货火车'
        return trainkind
    trainkind = '多色集装箱货车'
    return trainkind


def getcross(point1, point2, point):
    return (point2[0] - point1[0]) * (point[1] - point1[1]) - (point[0] - point1[0]) * (point2[1] - point1[1])


def point_in_train(points, point):
    point1 = points[0]
    point2 = points[1]
    point3 = points[2]
    point4 = points[3]
    p1 = getcross(point1, point2, point)
    p2 = getcross(point2, point3, point)
    p3 = getcross(point3, point4, point)
    p4 = getcross(point4, point1, point)
    flag = p1 * p3 >= 0 and p2 * p4 >= 0
    return flag


def rgbless(r, g, b, thresholdlike):  # rgb值均小于thresholdlike
    if r < thresholdlike and g < thresholdlike and b < thresholdlike:
        return 1
    else:
        return 0


def rgbmore(r, g, b, thresholdlike):  # rgb值均大于thresholdlike
    if r > thresholdlike and g > thresholdlike and b > thresholdlike:
        return 1
    else:
        return 0


def colorlike(r, g, b, thresholdlike):
    if abs(r - g) < thresholdlike and abs(r - b) < thresholdlike and abs(g - b) < thresholdlike:
        return 1
    else:
        return 0


# 根据RGB选择颜色
def choosecolor(rgb):
    r = rgb[2]
    g = rgb[1]
    b = rgb[0]
    rthreshold = 30
    white_threshold = 150
    thresholdlike = 32
    black_threshold = 60
    brown_threshold = 130
    gray_threshold = 130

    if rgbless(r, g, b, black_threshold):
        color = 'black'
    elif r - g > rthreshold and r - b > rthreshold and rgbmore(r, g, b, 40) \
            or 2 * r - g - b > rthreshold * 3 and rgbmore(r, g, b, 50):
        color = 'red'  # 红
    elif g - r > rthreshold and g - b > rthreshold and rgbmore(r, g, b,
                                                               40) \
            or 2 * g - r - b > rthreshold * 3 and rgbmore(r, g, b, 50):
        color = 'grean'  # 绿
    elif b - r > rthreshold and b - g > rthreshold and rgbmore(r, g, b,
                                                               40) \
            or 2 * b - r - g > rthreshold * 3 and rgbmore(r, g, b, 50):
        color = 'blue'  # 蓝
    elif colorlike(r, g, b, thresholdlike):
        if rgbmore(r, g, b, white_threshold):
            color = 'white'  # 白
        elif rgbless(r, g, b, black_threshold):
            color = 'black'  # 黑
        elif rgbless(r, g, b, brown_threshold):
            color = 'brown'  # 棕色
        else:
            color = 'gray'  # 灰色
    else:
        color = 'unknown'  # unknown
    return color


# 针对一行信息，输出对应的火车种类
def find_train_kind(fourpoints, img):
    # fourpoints = map(int,fourpoints)
    fourpoints = [int(float(x)) for x in fourpoints]
    points = [[fourpoints[0], fourpoints[1]], [fourpoints[2], fourpoints[3]],
              [fourpoints[4], fourpoints[5]], [fourpoints[6], fourpoints[7]]]
    # points = map(eval,points)
    xmin = min(fourpoints[0], fourpoints[2], fourpoints[4], fourpoints[6])
    xmax = max(fourpoints[0], fourpoints[2], fourpoints[4], fourpoints[6])
    ymin = min(fourpoints[1], fourpoints[3], fourpoints[5], fourpoints[7])
    ymax = max(fourpoints[1], fourpoints[3], fourpoints[5], fourpoints[7])
    num = 0
    # rgb = np.array([0,0,0])
    rgb = [0, 0, 0]
    colordict = {}
    for x in range(xmin, xmax, 5):
        for y in range(ymin, ymax, 5):
            p = [x, y]
            flag = point_in_train(points, p)
            if flag:
                num = num + 1
                rgb = rgb + img[y, x]
                color = choosecolor(list(map(int, img[y, x])))
                if color not in colordict:
                    colordict[color] = 0
                colordict[color] = colordict[color] + 1

    # rgb = list(map(str, map(int, (rgb / num))))
    # color = choosecolor(list(map(int,rgb)))
    key_values = list(colordict.keys())
    if 'unknown' in colordict.keys():
        num = num - colordict['unknown']
    for key in key_values:
        if colordict[key] < 0.05 * num or key == 'unknown':
            del colordict[key]
    for key in colordict.keys():
        colordict[key] = round(colordict[key] / num, 3)

    return train_kind(colordict)


def parse_box_info(pred):
    """
    :param pred: (9,) 4点坐标 + score
    """
    score = pred[-1]
    tl = np.asarray([pred[0], pred[1]], np.float32)
    tr = np.asarray([pred[2], pred[3]], np.float32)
    br = np.asarray([pred[4], pred[5]], np.float32)
    bl = np.asarray([pred[6], pred[7]], np.float32)

    box = np.asarray([tl, tr, br, bl], np.float32)
    cen_pts = np.mean(box, axis=0)

    return box, cen_pts, score


def dist_judge(alter_objs, protect_objs):
    """
    计算 alert_obj 到 protect_obj 距离，给出详细状态说明
    :param alter_objs:
    :param protect_objs:
    :return: situation = {} 状态说明
    """

    alert_cens = alter_objs['cen_pts']
    protect_lines = protect_objs['lines']

    w = len(alert_cens)
    h = len(protect_lines)

    distances = []

    for line in protect_lines:  # row
        for cen_pts in alert_cens:  # col
            distances.append(pt2line_distance(cen_pts, line))

    distances = np.array(distances).reshape((h, w))

    df = pd.DataFrame(distances)
    df.columns = alter_objs['objs']
    df.index = protect_objs['objs']

    return False, '异常', df


def iou_judge(target, pred, cls_idx, thre=0.1):
    """
    根据各类 iou 相较正常状态的变化，给出评估
    :param target: 以存储的正常情况下 img_seg 结果作为 target
    :param pred: 测试异常图片的预测的结果
    :param cls_idx:
    :param thre: 增减阈值判断
    :return:
    """
    gt_mask = (target == cls_idx).astype(int)
    pred_mask = (pred == cls_idx).astype(int)

    gt_sum = gt_mask.sum()
    pred_sum = pred_mask.sum()
    cls_percent = pred_sum / pred_mask.size  # 本类全图占比

    # 抑制噪声
    inter = (gt_mask * pred_mask).sum()
    union = gt_sum + pred_sum - inter

    # iou judge
    iou = (inter + 1) / (union + 1)
    if iou > 0.7:
        status_ok = True
    else:
        status_ok = False

    # 消减判断 平滑处理; 即便是不存在的类，也无妨
    less_percent = (gt_sum - inter + 1) / (gt_sum + 1)
    more_percent = (pred_sum - inter + 1) / (gt_sum + 1)

    if gt_sum == 0 and pred_sum == 0:
        less_percent, more_percent = 0, 0

    if cls_idx == 1:  # rail 轨道，异物侵线，造成轨道减少
        if less_percent > 0.05:
            status_ok = False

    if cls_idx == 2:  # plant 植被，减少区域不超过 thre 正常
        status_ok = (less_percent - more_percent) < thre

    if cls_idx == 3 or cls_idx == 4 or cls_idx == 7:  # buildings, road, train 抑制噪声
        if gt_sum < pred_mask.size * 0.02 or pred_sum < pred_mask.size * 0.02:
            status_ok = True

    if cls_idx == 5:  # land 土地，荒漠化增加区域不超过 0.1，正常
        status_ok = (more_percent - less_percent) < thre

    if cls_idx == 6:  # water 水域，变化区域不超
        status_ok = abs(more_percent - less_percent) < thre

    info = '原始区域减少 {:.2f}%, 增加 {:.2f}% 新区域, 现在占全图比例 {:.2f}%'.format(
        less_percent * 100, more_percent * 100, cls_percent * 100)

    return status_ok, info, iou, less_percent, more_percent, cls_percent
