"""
地面态势评估

将正常情况分割结果 作为 GT，只要通过 class_wise iou 判断即可

segment
    1.轨道变形
    2.水漫线路
    3.植被退化
    4.水域干涸
    5.异物侵线；边坡泥石流; 语义分割只得到铁轨对象，判断是否出现空洞异常
detect
    6.汽车误入
    7.经过桥梁，水中船只，不算
"""

import cv2
from demo.demo_seg import segment, preprocess, label_names
from pprint import pprint

input_w, input_h = 960, 540

seg_classes = 7


def iou_judge(target, pred, cls_idx, thre=0.1):
    """根据 iou 状况 判断场景变换"""
    gt_mask = (target == cls_idx).astype(int)
    pred_mask = (pred == cls_idx).astype(int)

    gt_sum = gt_mask.sum()
    pred_sum = pred_mask.sum()

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

    if cls_idx == 1:  # rail 轨道，异物琴弦，造成轨道减少
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

    info = '原始区域减少 {:.2f}%, 增加 {:.2f}% 新区域'.format(less_percent * 100, more_percent * 100)

    return status_ok, info, iou, less_percent, more_percent


# 基于语义分割的态势评估
def seg_situation_assessment(base_path, test_path):
    """
    :param base_path: 正常情况的基础图像分割结果，作为 target
    :param test_path: 测试图片
    :return: 态势评估结果
    """
    target = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)

    img = cv2.imread(test_path)
    img = preprocess(img, input_w, input_h).cuda()
    pred = segment(img)

    situation = {}

    for i in range(1, seg_classes + 1):
        status_ok, info, iou, less_percent, more_percent = iou_judge(target, pred, cls_idx=i)
        cat = label_names[i]

        if not status_ok:
            if i == 1:
                if less_percent > 0.15 and more_percent < 0.1:  # 只减不增
                    assessment = '异物侵线'
                else:  # 有减有增
                    assessment = '轨道变形'
            elif i == 2:
                assessment = '植被退化'
            elif i == 3:
                assessment = '建筑变化'
            elif i == 4:
                assessment = '道路变化'
            elif i == 5:
                assessment = '土地荒漠化'
            elif i == 6:
                if less_percent > more_percent:
                    assessment = '水域干涸'
                else:
                    assessment = '水漫线路'
            else:
                assessment = '列车变化'
        else:
            assessment = '正常'

        situation[cat] = {
            'status': status_ok,
            'assessment': assessment,
            'info': info,
            'iou': iou
        }

    judge = all([s['status'] for s in situation.values()])  # 全正常为 True
    summary = [s['assessment'] for s in situation.values() if not s['status']]

    pprint(situation)
    print(summary)
    print(judge)

    return judge


if __name__ == '__main__':
    demo_pairs = [
        # ('data/geo_hazard/1_轨道变形/1_ori_class.png', 'data/geo_hazard/1_轨道变形/1_hzd.png'),
        # ('data/geo_hazard/2_水漫线路/1_ori_class.png', 'data/geo_hazard/2_水漫线路/1_hzd.png'),
        # ('data/geo_hazard/3_植被退化/1_ori_class.png', 'data/geo_hazard/3_植被退化/1_hzd.png'),
        # ('data/geo_hazard/4_水域干涸/1_ori_class.png', 'data/geo_hazard/4_水域干涸/1_hzd.png'),
        # ('data/geo_hazard/5_异物侵线/1_ori_class.png', 'data/geo_hazard/5_异物侵线/1_hzd.png'),
        # ('data/geo_hazard/5_异物侵线/2_ori_class.png', 'data/geo_hazard/5_异物侵线/2_hzd.png'),
        ('data/geo_hazard/5_异物侵线/3_ori_class.png', 'data/geo_hazard/5_异物侵线/3_hzd.png'),
        ('data/geo_hazard/5_异物侵线/4_ori_class.png', 'data/geo_hazard/5_异物侵线/4_hzd.png'),
    ]

    for base_path, test_path in demo_pairs:
        print(test_path)
        seg_situation_assessment(base_path, test_path)
        print()
