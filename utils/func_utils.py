import os
import torch
import numpy as np
import cv2
from datasets.DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms_poly


def normalize_img(image):
    image = (image / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    return image.astype(np.float32)


def preprocess(image, input_w, input_h):
    image = cv2.resize(image, (input_w, input_h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out_image = normalize_img(image)
    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    out_image = torch.from_numpy(out_image)
    return out_image


def decode_prediction(predictions, category, input_w, input_h, ori_w, ori_h, down_ratio):
    """
    解析 predictions 得到 dict 类型结果
    :param predictions:
    :param category:
    :param input_w: 模型输出尺寸
    :param input_h:
    :param ori_w: 初始图像尺寸
    :param ori_h:
    :param down_ratio:
    :return:
    """
    predictions = predictions[0, :, :]

    pts0 = {cat: [] for cat in category}
    scores0 = {cat: [] for cat in category}

    for pred in predictions:
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)
        tt = np.asarray([pred[2], pred[3]], np.float32)
        rr = np.asarray([pred[4], pred[5]], np.float32)
        bb = np.asarray([pred[6], pred[7]], np.float32)
        ll = np.asarray([pred[8], pred[9]], np.float32)
        # 4 中点 -> 4 顶点
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]
        clse = pred[11]
        pts = np.asarray([tr, br, bl, tl], np.float32)  # 4,2

        pts[:, 0] = pts[:, 0] * down_ratio / input_w * ori_w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * ori_h

        pts0[category[int(clse)]].append(pts)  # 类别添加 pts
        scores0[category[int(clse)]].append(score)  # score

    return pts0, scores0


def non_maximum_suppression(pts, scores):
    """
    :param pts: (n,4,2)
    :param scores: (n,)
    :return:
    """
    nms_item = np.concatenate([pts.reshape((-1, 8)),
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)  # n,9
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.05)  # 0.1
    return nms_item[keep_index]


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from datasets.config.railway import color_map
import shapely.geometry as sgeo


def valid_box(pred_poly, img_poly):
    inter = pred_poly.intersection(img_poly)
    x, y = inter.exterior.coords.xy
    inter = np.array([x, y], dtype=int).T  # (n,2)
    return inter


def plt_results(results, ori_image, vis=True, save_path=None):
    h, w, _ = ori_image.shape
    img_bounds = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=int)
    img_poly = sgeo.Polygon(img_bounds)

    plt.figure(figsize=(w / 100, h / 100))
    plt.imshow(ori_image[:, :, ::-1])
    plt.axis('off')
    ax = plt.gca()

    polygons = []
    colors = []

    for cat, result in results.items():
        if result is None:
            continue
        for pred in result:
            score = pred[-1]

            tl = np.asarray([pred[0], pred[1]], np.float32)
            tr = np.asarray([pred[2], pred[3]], np.float32)
            br = np.asarray([pred[4], pred[5]], np.float32)
            bl = np.asarray([pred[6], pred[7]], np.float32)

            box = np.asarray([tl, tr, br, bl], np.float32)  # 4,2
            cen_pts = np.mean(box, axis=0)

            # Polygon: matplotlib(plt) / shapely(sgeo) 
            poly = patches.Polygon(valid_box(sgeo.Polygon(box), img_poly))
            polygons.append(poly)
            colors.append(color_map[cat])

            plt.annotate('%s:%.3f' % (cat, score),
                         xy=cen_pts, xycoords='data', xytext=(+7, +10), textcoords='offset points',
                         color='white',
                         bbox=dict(facecolor='black', alpha=0.5))

    colors = list(map(lambda t: (t[0] / 255, t[1] / 255, t[2] / 255), colors))
    p = PatchCollection(polygons, facecolors=colors, linewidths=0, alpha=0.4)  # surface
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolors='none', edgecolors=colors, linewidths=2)  # edges
    ax.add_collection(p)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    if vis:
        plt.show()


def draw_results(results, ori_image):
    h, w, _ = ori_image.shape
    img_bounds = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=int)
    img_poly = sgeo.Polygon(img_bounds)

    for cat, result in results.items():
        if result is None:
            continue
        for pred in result:  # result, (n,9)
            score = pred[-1]
            tl = np.asarray([pred[0], pred[1]], np.float32)
            tr = np.asarray([pred[2], pred[3]], np.float32)
            br = np.asarray([pred[4], pred[5]], np.float32)
            bl = np.asarray([pred[6], pred[7]], np.float32)

            # 4边中点
            tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
            rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
            bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
            ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

            box = np.asarray([tl, tr, br, bl], np.float32)
            cen_pts = np.mean(box, axis=0)

            # 原始检测结果
            # 从 center 到 4边中点 连线 [起点, 终点]
            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 1, 1)
            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)

            # 原始检测结果，得到与 img 交集
            # inter box，计算 pred_box 和 rect_img 交集
            inter_box = valid_box(sgeo.Polygon(box), img_poly)  # poly pts

            # 使用 drawContours 画上多边形
            ori_image = cv2.drawContours(ori_image, [inter_box], -1, (255, 0, 255), 1, 1)
            # box = cv2.boxPoints(cv2.minAreaRect(box)) # 外界矩形
            # ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (0,255,0),1,1)

            cv2.putText(ori_image, '{:.2f} {}'.format(score, cat),
                        # (box[1][0], box[1][1]),
                        (cen_pts[0], cen_pts[1]),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
    return ori_image


def write_results(args,
                  model,
                  dsets,
                  down_ratio,
                  decoder,
                  result_path,
                  print_ps=False):
    results = {cat: {img_id: [] for img_id in dsets.img_ids} for cat in dsets.category}
    for index in range(len(dsets)):
        data_dict = dsets.__getitem__(index)
        image = data_dict['image'].cuda()
        img_id = data_dict['img_id']
        ori_w = data_dict['image_w']
        ori_h = data_dict['image_h']

        with torch.no_grad():
            pr_decs = model(image)

        decoded_pts = []
        decoded_scores = []
        # torch.cuda.synchronize(device)
        predictions = decoder.ctdet_decode(pr_decs)
        pts0, scores0 = decode_prediction(predictions, dsets.category,
                                          args.input_w, args.input_h,
                                          ori_w, ori_h,
                                          down_ratio)
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)

        # nms
        for cat in dsets.category:
            if cat == 'background':
                continue
            pts_cat = []
            scores_cat = []
            for pts0, scores0 in zip(decoded_pts, decoded_scores):
                pts_cat.extend(pts0[cat])
                scores_cat.extend(scores0[cat])
            pts_cat = np.asarray(pts_cat, np.float32)
            scores_cat = np.asarray(scores_cat, np.float32)
            if pts_cat.shape[0]:
                nms_results = non_maximum_suppression(pts_cat, scores_cat)
                results[cat][img_id].extend(nms_results)
        if print_ps:
            print('testing {}/{} data {}'.format(index + 1, len(dsets), img_id))

    for cat in dsets.category:
        if cat == 'background':
            continue
        with open(os.path.join(result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
            for img_id in results[cat]:
                for pt in results[cat][img_id]:
                    f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
