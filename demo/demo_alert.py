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
import sys

sys.path.insert(0, '/nfs/xs/Codes/BBAVectors-Oriented-Object-Detection')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np

from datasets.config.railway import *
from demo.demo_seg import segment, load_seg_model
from demo.demo_dec import detect, load_dec_model, DecDecoder
from demo.base import *
from utils.func_utils import preprocess, plt_results
from utils.misc import box_major_axis
from utils.vis import plt_compare_seg


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

    def load_img(self, img_path):
        self.ori_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        self.ori_h, self.ori_w, _ = self.ori_image.shape
        self.img = preprocess(self.ori_image, self.input_w, self.input_h).cuda()

    def dec_alert(self, vis=False):
        """
        基于目标检测的态势评估
            根据场景中活动类别 到 train/rail 距离，提出预警
        :return: 态势评估结果
        """
        # self.load_img(img_path=test_path)
        dec_results = detect(self.dec_model, self.img, self.decoder,
                             self.input_w, self.input_h, self.ori_w, self.ori_h)
        dec_results = {
            cat: dec_results[cat]
            for cat in dec_results if len(dec_results[cat]) > 0
        }
        if vis:
            plt_results(
                dec_results, self.ori_image, vis=False,
                save_path=test_path.replace('.png', '_det.png')
            )

        # 预警类，取物体中心
        alter_cls = ['small-vehicle', 'large-vehicle', 'helicopter', 'plane', 'ship']  # 可活动类，潜在危险
        alter_objs = {'cen_pts': [], 'objs': []}

        for cat in alter_cls:
            if cat not in dec_results:
                continue
            result = dec_results[cat]  # n,9
            for idx, pred in enumerate(result):
                box, cen_pts, score = parse_box_info(pred)
                alter_objs['cen_pts'].append(cen_pts)
                alter_objs['objs'].append(f'{cat}_{idx + 1}')  # 直接编排同类物体

        # 如果场景中没有预警类出现，直接 status 正常
        if len(alter_objs['objs']) == 0:
            return {
                'dec_results': dec_results,
                'status': True,
                'assesement': '正常',
                'info': ''
            }

        # 保护类，取直线方程
        protect_cls = ['train', 'rail']
        protect_objs = {'lines': [], 'objs': []}

        for cat in protect_cls:
            if cat not in dec_results:
                continue
            result = dec_results[cat]
            for idx, pred in enumerate(result):
                box, cen_pts, score = parse_box_info(pred)
                protect_objs['lines'].append(box_major_axis(box))
                protect_objs['objs'].append(f'{cat}_{idx + 1}')

        # 如果场景中没有 要保护的类出现，直接 status 正常
        if len(protect_objs['lines']) == 0:
            return {
                'dec_results': dec_results,
                'status': True,
                'assesement': '正常',
                'info': ''
            }

        # alert-protect 类同时出现时，逐对比较，对场景状况进行判断
        status, assesement, info = dist_judge(alter_objs, protect_objs)

        return {
            'dec_results': dec_results,  # 原始检测结果
            'status': status,  # 态势评估，True 正常，False 不正常
            'assesement': assesement,  # 态势评估 文字描述
            'info': info,  # 详细信息
        }

    def seg_alert(self, base_path, vis=False):
        """
        基于语义分割的态势评估
            根据测试场景 各类 iou 相较正常状态的变化，给出评估
        :param base_path: 正常情况的基础图像分割结果，作为 target
        :return: 态势评估结果
        """
        target = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
        pred = segment(self.seg_model, self.img)

        if vis:
            plt_compare_seg(target, pred, seg_label_colors)

        situation = {}

        for i in range(1, self.seg_classes + 1):
            status_ok, info, iou, less_percent, more_percent, cls_percent = iou_judge(target, pred, cls_idx=i)
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
                'iou': iou,
                'proportion': cls_percent,
            }

        status = all([s['status'] for s in situation.values()])  # 全正常为 True
        assessment = [s['assessment'] for s in situation.values() if not s['status']]
        seg_results = {
            seg_map_zh[cat]: situation[cat]['proportion']  # 有效场景占比
            for cat in situation if situation[cat]['proportion'] > 0.05
        }
        return {
            'seg_results': seg_results,
            'status': status,
            'assessment': assessment,
            'info': situation,  # 各类详细信息
        }

    # @staticmethod
    def semantic_tree(self, seg_results, dec_results):
        """
        :param seg_results: 出现类别，及 占比
            {'rail': 0.17600694444444445, 'plant': 0.5232851080246913, 'land': 0.1841184413580247, 'train': 0.06840277777777778}
        :param dec_results: 各类 box, 前 8 个 为 4点坐标; 最后1个为 score
            {'train': [array([8.72206299e+02, 1.28269379e+02, 8.75129395e+02, 1.61081055e+02,
                              8.57380371e+01, 2.35098236e+02, 8.28148804e+01, 2.02286530e+02, 2.17964023e-01])]}
        :return:
            Tree
        """
        treedict = {}
        First_heading = '轨道交通语义结构树'
        treedict[First_heading] = {}

        print(seg_results)
        print(dec_results)

        for classname in seg_results.keys():
            if classname in ['绿植', '房屋', '黄土地', '水域']:
                if '地貌' not in treedict[First_heading]:
                    treedict[First_heading]['地貌'] = {}
                treedict[First_heading]['地貌'][classname] = 1
            if classname in ['轨道']:
                if '道路' not in treedict[First_heading]:
                    treedict[First_heading]['道路'] = {}
                treedict[First_heading]['道路'][classname] = 1

        for classname in dec_results.keys():
            if classname in ['bridge', 'rail']:
                count = len(dec_results[classname])
                if '道路' not in treedict[First_heading]:
                    treedict[First_heading]['道路'] = {}
                if classname == 'bridge':
                    classname = '桥梁'
                    treedict[First_heading]['道路'][classname] = count
                if classname == 'rail':
                    classname = '轨道'
                    treedict[First_heading]['道路'][classname] = count
            elif classname in ['plane', 'ship', 'helicopter', 'small-vehicle', 'large-vehicle', 'train']:
                if '交通工具' not in treedict[First_heading]:
                    treedict[First_heading]['交通工具'] = {}
                if classname in ['plane', 'ship', 'helicopter']:
                    count = len(dec_results[classname])
                    if classname == 'plane':
                        classname = '飞机'
                    if classname == 'ship':
                        classname = '船只'
                    if classname == 'helicopter':
                        classname = '直升机'
                    treedict[First_heading]['交通工具'][classname] = count
                elif classname in ['small-vehicle', 'large-vehicle']:
                    count = len(dec_results[classname])
                    if '车辆' not in treedict[First_heading]['交通工具']:
                        treedict[First_heading]['交通工具']['车辆'] = {}
                    if classname == 'small-vehicle':
                        classname = '小车'
                    if classname == 'large-vehicle':
                        classname = '大车'
                    treedict[First_heading]['交通工具']['车辆'][classname] = count
                elif classname == 'train':
                    if '火车' not in treedict[First_heading]['交通工具']:
                        treedict[First_heading]['交通工具']['火车'] = {}

                    for line in dec_results[classname]:
                        trainkind = find_train_kind(line[:-1], self.ori_image)
                        if trainkind not in treedict[First_heading]['交通工具']['火车']:
                            treedict[First_heading]['交通工具']['火车'][trainkind] = 0
                        treedict[First_heading]['交通工具']['火车'][trainkind] = treedict[First_heading]['交通工具']['火车'][
                                                                               trainkind] + 1

        print(treedict)

        return treedict


if __name__ == '__main__':
    ra = RailwayAlert()

    ## segment
    # scene_pairs = [
    #     ('data/geo_hazard/1_轨道变形/1_ori_class.png', 'data/geo_hazard/1_轨道变形/1_hzd.png'),
    #     ('data/geo_hazard/2_水漫线路/1_ori_class.png', 'data/geo_hazard/2_水漫线路/1_hzd.png'),
    #     ('data/geo_hazard/3_植被退化/1_ori_class.png', 'data/geo_hazard/3_植被退化/1_hzd.png'),
    #     ('data/geo_hazard/4_水域干涸/1_ori_class.png', 'data/geo_hazard/4_水域干涸/1_hzd.png'),
    #     ('data/geo_hazard/5_异物侵线/1_ori_class.png', 'data/geo_hazard/5_异物侵线/1_hzd.png'),
    #     ('data/geo_hazard/5_异物侵线/2_ori_class.png', 'data/geo_hazard/5_异物侵线/2_hzd.png'),
    #     ('data/geo_hazard/5_异物侵线/3_ori_class.png', 'data/geo_hazard/5_异物侵线/3_hzd.png'),
    #     ('data/geo_hazard/5_异物侵线/4_ori_class.png', 'data/geo_hazard/5_异物侵线/4_hzd.png'),
    # ]
    # for base_path, test_path in seg_pairs:
    #     print(test_path)
    #     ra.seg_alert(base_path, test_path)
    #     print()

    ## detect
    # dec_imgs = [
    #     'data/geo_hazard/6_汽车误入/1.png',
    #     'data/geo_hazard/6_汽车误入/2.png',
    # ]
    # for test_path in dec_imgs:
    #     print(test_path)
    #     ra.dec_alert(test_path, vis=True)

    ## tree
    scene_pairs = [
        ('data/geo_hazard/5_异物侵线/3_ori_class.png', 'data/geo_hazard/5_异物侵线/3_hzd.png'),
    ]
    for base_path, test_path in scene_pairs:
        ra.load_img(test_path)
        seg_res = ra.seg_alert(base_path)
        dec_res = ra.dec_alert()
        ra.semantic_tree(seg_res['seg_results'], dec_res['dec_results'])
