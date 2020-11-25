"""
图像状态：
    有报警
    无报警
异常种类
    0.异物入侵
    1.边坡坍塌
"""
import torch
import torchvision.transforms as tr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.hrnet.hrnet import HRNet
from PIL import Image

label_colors = [
    (0, 0, 0),  # bg=0; 作为 0 类
    (0, 0, 255),  # rail=1
    (255, 0, 0),  # train=2
    (0, 255, 0),  # obstacle=3
]

trans_img = tr.Compose([
    tr.ToTensor(),
    tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_seg_model(model_path='models/model_best.pth'):
    model = HRNet(
        cfg_path='models/hrnet/cfg/seg_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        num_classes=3,
        use_pretrain=False
    )
    # resume = 'runs/railway/seg_hrnet18_fg_scale_cls3_epoch40_hrnet_Nov13_234125/model_best.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    model.load_state_dict(state_dict_)
    print('loaded seg model from {}, epoch {}'.format(model_path, checkpoint['epoch']))
    return model.eval().cuda()


@torch.no_grad()
def segment(model, image):
    pred = model(image).argmax(dim=1)
    pred = pred.cpu().numpy().squeeze().astype(np.uint8)
    print('segment done!')
    return pred


def color_code_target(target, label_colors):
    return np.array(label_colors, dtype='uint8')[target.astype('uint8')]


def filter_main_ccps(rail_msk, min_size=30 * 30):
    """
    只保留 size > min_size 的主要联通分量
    """
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(rail_msk)
    # ccp 计算结果是包含 bg 的

    # stats 五元组，外接矩形和面积
    cont_idxs = np.arange(nlabels, dtype=int)  # -bg
    discard_idxs = cont_idxs[stats[:, -1] <= min_size]
    # print(discard_idxs)
    for idx in discard_idxs:
        labels[labels == idx] = 0  # small ccp ->0，去噪

    labels[labels > 0] = 1  # 恢复成 binary rail_masl
    labels = labels.astype('uint8')
    return labels


def is_valid_center_vector(r1, r2):
    """
    r1,r2: 两个待合并连通分量 region
    r1,r2 中点组成的向量 夹角不在各个方向向量夹角范围内，不满足凸函数斜率性质，不合并

    两两可归并标准
        1.方向向量夹角变化在 turn_angle_thre 合理范围内
        2.潜在平行判断? 如何不合并
    """
    c1_x, c1_y = r1['center']
    c2_x, c2_y = r2['center']
    theta1, theta2 = r1['theta'], r2['theta']

    if c1_x - c2_x == 0:
        center_theta = 90
    else:
        center_theta = np.rad2deg(np.arctan((c1_y - c2_y) / (c1_x - c2_x)))

    # 在一条曲线轨迹上，即 theta 在二角范围内
    on_one_curve = theta1 <= center_theta <= theta2 or theta1 >= center_theta >= theta2
    if on_one_curve:
        return True

    # center 之间垂直距离很近，即两条平行线间距离
    center_d = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    if center_theta < min(theta1, theta2):  # 左侧
        d = center_d * np.sin(np.deg2rad(min(theta1, theta2) - center_theta))
    elif center_theta > max(theta1, theta2):
        d = center_d * np.sin(np.deg2rad(center_theta - max(theta1, theta2)))
    else:
        d = 1000  # 置为大值
    # print(d)

    return d < 40  # 满足的平行间距


def merge_railway_cnts(contours, turn_angle_thre=30):
    """
    Args:
        contours: 初始分割的分量
        turn_angle_thre: 子连通分量主方向变化范围，满足范围作为候选合并，再用 is_valid_center_vector(r1,r2) 判断

    Returns:
        合并后的大分量
    """
    # 存储 ccp 外接矩形信息，根据矩形主方向判断分量合并
    cont_min_rects = []
    for idx, cont in enumerate(contours):
        rect = cv2.minAreaRect(cont)
        # 做出 box
        rect_points = cv2.boxPoints(rect)
        rect_points = np.array(rect_points, dtype=int)

        # 做出 box 方向; 调整 w>h, w 为主方向
        # theta: bbox_w 与 x 轴正向夹角; 顺时针旋转为正，逆时针负
        cx, cy, bbox_w, bbox_h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        if bbox_w < bbox_h:
            bbox_w, bbox_h = bbox_h, bbox_w
            theta = theta + 90
        cont_min_rects.append({
            'idx': idx,
            'theta': theta,  # 以 x 轴正向为方向，顺时针为正，逆时针为负；
            'center': (cx, cy),  # 中点
        })

    # 判断子联通分量的合并情况
    merge_conts = []
    num_cont = len(contours)
    use_cont = [False] * num_cont

    for i in range(num_cont):
        if use_cont[i]:
            continue
        for j in range(num_cont):
            if i == j:
                continue
            r1, r2 = cont_min_rects[i], cont_min_rects[j]
            angle = abs(r1['theta'] - r2['theta'])
            if angle < turn_angle_thre and is_valid_center_vector(r1, r2):
                # 判断当前组合 能否和之前出现的合并
                num = len(merge_conts)
                find = False
                for idx in range(num):
                    if r1['idx'] in merge_conts[idx] and r2['idx'] not in merge_conts[idx]:
                        merge_conts[idx] += [r2['idx']]
                        find = True
                        break
                    elif r1['idx'] not in merge_conts[idx] and r2['idx'] in merge_conts[idx]:
                        merge_conts[idx] += [r1['idx']]
                        find = True
                        break
                if not find:
                    merge_conts.append([r1['idx'], r2['idx']])
                # 置为已用
                use_cont[i] = True
                use_cont[j] = True

    new_conts = []
    ori_cnt_idxs = set(range(len(contours)))
    for cnt_idxs in merge_conts:
        # 合并可归并的联通分量
        new_conts.append(np.vstack([contours[i] for i in cnt_idxs]))
        for idx in cnt_idxs:
            if idx in ori_cnt_idxs:
                ori_cnt_idxs.remove(idx)

    for idx in ori_cnt_idxs:  # remain
        new_conts.append(contours[idx])

    return new_conts


def postprocess(img, msk, scale=1, vis=False):
    """
    Args:
        img: opencv 读取原图
        msk: 语义分割结果
        scale: 对结果缩放，加快处理
        vis: True 使用 matplotlib 显示分割结果

    Returns:
        {
            'status': status,  # 图像状态: [正常，异常]
            'osbtacle': main_obs,  # 异常状态: [无, 异物入侵, 边坡滑坡]
        }
    """
    # 从分割结果 取出 rail_msk
    rail_msk = (msk == 1).astype('uint8')
    rail_msk = cv2.resize(rail_msk, dsize=None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST)
    rail_msk = filter_main_ccps(rail_msk)  # 过滤出主要联通分量

    # 分析 msk 子联通分量，对于异常分割，但是同一铁轨的分量进行合并
    contours, hierarchy = cv2.findContours(rail_msk,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    new_conts = merge_railway_cnts(contours)

    h, w = rail_msk.shape
    bound_rail = np.zeros((h, w, 3), dtype='uint8')

    for idx, cont in enumerate(new_conts):
        rect = cv2.minAreaRect(cont)
        # 做出 box
        rect_points = cv2.boxPoints(rect)
        rect_points = np.array(rect_points, dtype=int)
        cv2.fillPoly(bound_rail, [rect_points], (1, 1, 1))

    bound_rail = bound_rail[:, :, 0]

    # 绿色表示安全带
    safe_msk = bound_rail * msk  # 取出 bound_rail 内 rail / train
    safe_msk[(safe_msk == 0) & (bound_rail == 1)] = 3  # obstacle

    obstacle_msk = (safe_msk == 3).astype('uint8')
    kernel = np.ones((15, 15), np.uint8)  # 开操作，消去细长物
    obstacle_msk = cv2.morphologyEx(obstacle_msk, cv2.MORPH_OPEN, kernel)

    # 过滤掉 30*30 小物体
    obstacle_msk = filter_main_ccps(obstacle_msk)
    obstacle_color = color_code_target(obstacle_msk, label_colors)
    obstacle_img = img * np.repeat(obstacle_msk[:, :, np.newaxis], repeats=3, axis=-1)

    if vis:
        f, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs.flat[0].imshow(img)
        axs.flat[0].set_title('image')
        axs.flat[1].imshow(obstacle_img)
        axs.flat[1].set_title('obstacle')

        axs.flat[2].imshow(color_code_target(safe_msk, label_colors))
        axs.flat[2].set_title('railway segmentation')
        axs.flat[3].imshow(obstacle_color)
        axs.flat[3].set_title('obstacle segmentation')

        plt.show()

    obs_mask = obstacle_msk > 0

    if obs_mask.sum() == 0:
        status = '正常'
        main_obs = '无'
    else:
        status = '异常'
        main_obs_cls = np.sum(obstacle_msk[obs_mask]) / obs_mask.sum()
        main_obs = '异物入侵' if main_obs_cls < 4 else '边坡滑坡'

    return {
        'status': status,  # 图像状态: [正常，异常]
        'osbtacle': main_obs,  # 异常状态: [无, 异物入侵, 边坡滑坡]
    }


def state_assessment(model, img_path, vis=False):
    # ori_image = cv2.imread(img_path)[:, :, ::-1]
    ori_image = Image.open(img_path)
    image = trans_img(ori_image).unsqueeze(0).cuda()
    # 语义分割
    pred = segment(model, image)
    # 后处理
    res = postprocess(ori_image, pred, vis=vis)
    return res


if __name__ == '__main__':
    import os

    model_path = 'runs/railway/seg_hrnet18_fg_scale_cls3_epoch40_hrnet_Nov13_234125/model_best.pth'
    # model_path='models/model_best.pth'
    model = load_seg_model(model_path)
    # Note: 只加载一次

    # 测试图像所在 dir
    img_dir = 'data/obstacle/img'

    for img in os.listdir(img_dir):
        print(img)
        img_path = os.path.join(img_dir, img)
        res = state_assessment(model, img_path, vis=True)
        print(res)
        print()
