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
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np
import pandas as pd

from datasets.config.railway import *
from demo.demo_seg import segment, load_seg_model
from demo.demo_dec import detect, load_dec_model, DecDecoder
from utils.func_utils import preprocess, plt_results
from utils.misc import plt_compare_seg, box_major_axis, pt2line_distance
from pprint import pprint


class RailwayAlert:
    """态势评估 & 预警"""

    def __init__(self):
        self.input_w = 960
        self.input_h = 540

        self.seg_classes = 7
        self.dec_classes = 17

        self.seg_model = load_seg_model()
        self.dec_model = load_dec_model()

        self.decoder = DecDecoder(K=500, conf_thresh=0.18, num_classes=self.dec_classes)

    @staticmethod
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

    def dec_alert(self, test_path, vis=False):
        """
        基于目标检测的态势评估
            根据场景中活动类别 到 train/rail 距离，提出预警
        :param test_path: 测试图片路径
        :return: 态势评估结果
        """
        ori_image = cv2.imread(test_path)
        ori_h, ori_w, _ = ori_image.shape
        img = preprocess(ori_image, self.input_w, self.input_h).cuda()

        dec_results = detect(self.dec_model, img, self.decoder,
                             self.input_w, self.input_h, ori_w, ori_h)

        if vis:
            plt_results(dec_results, ori_image)

        # 预警类，取物体中心
        alter_cls = ['small-vehicle', 'large-vehicle', 'helicopter', 'plane', 'ship']  # 可活动类，潜在危险
        alter_objs = {'cen_pts': [], 'objs': []}

        for cat in alter_cls:
            result = dec_results[cat]  # n,9
            for idx, pred in enumerate(result):
                box, cen_pts, score = self.parse_box_info(pred)
                alter_objs['cen_pts'].append(cen_pts)
                alter_objs['objs'].append(f'{cat}_{idx + 1}')  # 直接编排同类物体

        # 如果场景中没有预警类出现，直接 status 正常
        if len(alter_objs['objs']) == 0:
            return True, '正常', ''

        # 保护类，取直线方程
        protect_cls = ['train', 'rail']
        protect_objs = {'lines': [], 'objs': []}

        for cat in protect_cls:
            result = dec_results[cat]
            for idx, pred in enumerate(result):
                box, cen_pts, score = self.parse_box_info(pred)
                protect_objs['lines'].append(box_major_axis(box))
                protect_objs['objs'].append(f'{cat}_{idx + 1}')

        # 如果场景中没有 要保护的类出现，直接 status 正常
        if len(protect_objs['lines']) == 0:
            return True, '正常', ''

        # alert-protect 类同时出现时，逐对比较，对场景状况进行判断
        status, assesement, info = self.dist_judge(alter_objs, protect_objs)
        return status, assesement, info

    @staticmethod
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

    def seg_alert(self, base_path, test_path, vis=False):
        """
        基于语义分割的态势评估
            根据测试场景 各类 iou 相较正常状态的变化，给出评估
        :param base_path: 正常情况的基础图像分割结果，作为 target
        :param test_path: 测试图片
        :return: 态势评估结果
        """
        target = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)

        ori_image = cv2.imread(test_path)
        img = preprocess(ori_image, self.input_w, self.input_h).cuda()
        pred = segment(self.seg_model, img)

        if vis:
            plt_compare_seg(target, pred, seg_label_colors)

        situation = {}

        for i in range(1, self.seg_classes + 1):
            status_ok, info, iou, less_percent, more_percent = self.iou_judge(target, pred, cls_idx=i)
            cat = seg_label_names[i]

            if not status_ok:
                if i == 1:
                    if less_percent > 0.05 and more_percent < 0.1:
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

    @staticmethod
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

        info = '原始区域减少 {:.2f}%, 增加 {:.2f}% 新区域'.format(less_percent * 100, more_percent * 100)

        return status_ok, info, iou, less_percent, more_percent


if __name__ == '__main__':
    seg_pairs = [
        ('data/geo_hazard/1_轨道变形/1_ori_class.png', 'data/geo_hazard/1_轨道变形/1_hzd.png'),
        ('data/geo_hazard/2_水漫线路/1_ori_class.png', 'data/geo_hazard/2_水漫线路/1_hzd.png'),
        ('data/geo_hazard/3_植被退化/1_ori_class.png', 'data/geo_hazard/3_植被退化/1_hzd.png'),
        ('data/geo_hazard/4_水域干涸/1_ori_class.png', 'data/geo_hazard/4_水域干涸/1_hzd.png'),
        ('data/geo_hazard/5_异物侵线/1_ori_class.png', 'data/geo_hazard/5_异物侵线/1_hzd.png'),
        ('data/geo_hazard/5_异物侵线/2_ori_class.png', 'data/geo_hazard/5_异物侵线/2_hzd.png'),
        ('data/geo_hazard/5_异物侵线/3_ori_class.png', 'data/geo_hazard/5_异物侵线/3_hzd.png'),
        ('data/geo_hazard/5_异物侵线/4_ori_class.png', 'data/geo_hazard/5_异物侵线/4_hzd.png'),
    ]

    dec_imgs = [
        'data/geo_hazard/6_汽车误入/1.png',
        'data/geo_hazard/6_汽车误入/2.png',
    ]

    ra = RailwayAlert()

    for test_path in dec_imgs:
        print(test_path)
        ra.dec_alert(test_path, vis=True)

    # for base_path, test_path in seg_pairs:
    #     print(test_path)
    #     ra.seg_alert(base_path, test_path)
    #     print()
