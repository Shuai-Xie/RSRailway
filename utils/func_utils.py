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
        # from mid_pts to box_pts
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]
        clse = pred[11]
        pts = np.asarray([tr, br, bl, tl], np.float32)

        pts[:, 0] = pts[:, 0] * down_ratio / input_w * ori_w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * ori_h

        pts0[category[int(clse)]].append(pts)  # 类别添加 pts
        scores0[category[int(clse)]].append(score)  # score

    return pts0, scores0


def non_maximum_suppression(pts, scores):
    nms_item = np.concatenate([pts[:, 0:1, 0],
                               pts[:, 0:1, 1],
                               pts[:, 1:2, 0],
                               pts[:, 1:2, 1],
                               pts[:, 2:3, 0],
                               pts[:, 2:3, 1],
                               pts[:, 3:4, 0],
                               pts[:, 3:4, 1],
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.05)  # 0.1
    return nms_item[keep_index]


def draw_results(results, ori_image):
    for cat, result in results.items():
        for pred in result:
            score = pred[-1]
            tl = np.asarray([pred[0], pred[1]], np.float32)
            tr = np.asarray([pred[2], pred[3]], np.float32)
            br = np.asarray([pred[4], pred[5]], np.float32)
            bl = np.asarray([pred[6], pred[7]], np.float32)

            tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
            rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
            bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
            ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

            box = np.asarray([tl, tr, br, bl], np.float32)
            cen_pts = np.mean(box, axis=0)

            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 1, 1)
            cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (255, 0, 255), 1, 1)
            # box = cv2.boxPoints(cv2.minAreaRect(box)) # 外界矩形
            # ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (0,255,0),1,1)
            cv2.putText(ori_image, '{:.2f} {}'.format(score, cat), (box[1][0], box[1][1]),
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
