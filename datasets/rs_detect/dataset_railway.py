import os
import cv2
import numpy as np
from datasets.DOTA_devkit.ResultMerge_multi_process import mergebypoly
from datasets.rs_detect.base import BaseDataset


class Railway(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        """
        :param data_dir: /datasets/rs_detect/DOTA_data/train_split
        :param phase: train / test
        :param input_h:
        :param input_w:
        :param down_ratio:
        """
        super(Railway, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = [
            'plane',
            'baseball-diamond',
            'bridge',
            'ground-track-field',
            'small-vehicle',
            'large-vehicle',
            'ship',
            'tennis-court',
            'basketball-court',
            'storage-tank',
            'soccer-ball-field',
            'roundabout',
            'harbor',
            'swimming-pool',
            'helicopter',
            'train',  # 添加火车和轨道
            'rail',
        ]
        self.color_pans = [
            (204, 78, 210),
            (0, 192, 255),
            (0, 131, 0),
            (240, 176, 0),
            (254, 100, 38),
            (0, 0, 255),
            (182, 117, 46),
            (185, 60, 129),
            (204, 153, 255),
            (80, 208, 146),
            (0, 0, 204),
            (17, 90, 197),
            (0, 255, 255),
            (102, 255, 102),
            (255, 255, 0),
            (255, 0, 0),  # 红
            (0, 255, 0),  # 绿
        ]
        self.num_classes = len(self.category)
        self.cat_ids = {cat: i for i, cat in enumerate(self.category)}

        self.img_ids = self.load_img_ids()  # img lists
        self.image_path = os.path.join(data_dir, 'images')  # train
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        if self.phase == 'train':
            # image_set_index_file = os.path.join(self.data_dir, 'train.txt')  # 只有 train + rail
            image_set_index_file = os.path.join(self.data_dir, 'train_path.txt')  # DOTA 15 + 2 railway
        else:
            image_set_index_file = os.path.join(self.data_dir, 'val.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def load_image(self, img_id):
        """
        :param img_id: int, list idx; str, name
        """
        if isinstance(img_id, int):  # 如果是 index，索引得到 img_name
            img_id = self.img_ids[img_id]

        # 绝对路径 / img_name 读取图片
        if img_id.startswith('/'):  # 绝对路径
            img = cv2.imread(img_id)
        else:
            img = cv2.imread(os.path.join(self.image_path, img_id + '.png'))

        return img

    def load_annoFolder(self, img_id):  # img_id: P2805
        # 绝对路径 / img_name 读取 labelTxt path
        if img_id.startswith('/'):
            return img_id.replace('.png', '.txt').replace('images', 'labelTxt')
        else:
            return os.path.join(self.label_path, img_id + '.txt')

    def load_annotation(self, index):
        """
        read anns from txt
            717.0 76.0 726.0 78.0 722.0 95.0 714.0 90.0 small-vehicle 0
        :return
            annotation = {
                'pts': np.asarray(valid_pts, np.float32),  # n,4,2
                'cat': np.asarray(valid_cat, np.int32),  # n
                'dif': np.asarray(valid_dif, np.int32),  # n
            }
        """
        image = self.load_image(index)
        h, w, c = image.shape

        # valid
        valid_pts = []
        valid_cat = []
        valid_dif = []

        # read labelTxt
        with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                if len(obj) > 8:
                    # 4 pts -> int
                    x1 = min(max(float(obj[0]), 0), w - 1)
                    y1 = min(max(float(obj[1]), 0), h - 1)
                    x2 = min(max(float(obj[2]), 0), w - 1)
                    y2 = min(max(float(obj[3]), 0), h - 1)
                    x3 = min(max(float(obj[4]), 0), w - 1)
                    y3 = min(max(float(obj[5]), 0), h - 1)
                    x4 = min(max(float(obj[6]), 0), w - 1)
                    y4 = min(max(float(obj[7]), 0), h - 1)
                    # hbb 边界坐标
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = max(x1, x2, x3, x4)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = max(y1, y2, y3, y4)
                    # filter small instances, size < 10*10
                    if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                        valid_pts.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                        valid_cat.append(self.cat_ids[obj[8]])  # catname -> id
                        valid_dif.append(int(obj[9]))  # dif

        f.close()
        annotation = {
            'pts': np.asarray(valid_pts, np.float32),  # n,4,2
            'cat': np.asarray(valid_cat, np.int32),  # n
            'dif': np.asarray(valid_dif, np.int32),  # n
        }

        # self.debug(index, valid_pts, valid_cat, valid_dif)
        return annotation

    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)

    def debug(self, index, valid_pts, valid_cat, valid_dif):
        import matplotlib.pyplot as plt
        pts0 = np.asarray(valid_pts, np.float32)
        img = self.load_image(index)
        color = {
            't': (0, 0, 255),  # red
            'r': (0, 255, 0),  # green
            'b': (0, 255, 255),  # yellow
            'l': (255, 0, 0),  # blue
        }
        for i in range(pts0.shape[0]):
            pt = pts0[i, :, :]
            tl = pt[0, :]
            tr = pt[1, :]
            br = pt[2, :]
            bl = pt[3, :]
            cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), color['t'], 1, 1)
            cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), color['r'], 1, 1)
            cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), color['b'], 1, 1)
            cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), color['l'], 1, 1)
            # text: diff:cat
            cv2.putText(img, '{}:{}'.format(valid_dif[i], self.category[valid_cat[i]]),
                        (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 1, 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(img[:, :, ::-1])
        plt.show()


if __name__ == '__main__':
    data_dir = '/datasets/rs_detect/railway/train'
    phase = 'train'
    # 16:9
    dset = Railway(data_dir, phase, input_h=450, input_w=800, down_ratio=4)
    print(len(dset))

    for data_dict in dset:
        pass
