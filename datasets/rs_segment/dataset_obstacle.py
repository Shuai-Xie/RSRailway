from torch.utils.data import Dataset
import os
from PIL import Image
import datasets.rs_segment.transforms as tr
import numpy as np
from utils.misc import read_txt_as_list
import random


class RailwayObstacle(Dataset):
    """铁轨异物分割检测"""

    def __init__(self, root, split, base_size=None, crop_size=None):
        data_dir = os.path.join(root, split)

        self.img_paths = read_txt_as_list(os.path.join(data_dir, f'{split}_img_paths.txt'))
        self.target_paths = read_txt_as_list(os.path.join(data_dir, f'{split}_target_paths.txt'))

        self.num_class = 3
        self.transform = tr.get_transform(split, base_size, crop_size)

        self.label_names = ['bg', 'rail', 'train']
        self.label_colors = [
            (0, 0, 0),  # bg=0; 作为 0 类
            (0, 0, 255),  # rail=1
            (0, 255, 0),  # train=2
            (255, 0, 0),  # obstacle=3
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        target = Image.open(self.target_paths[index])

        sample = {
            'img': img,
            'target': target,
            'filename': os.path.basename(self.img_paths[index])
        }
        sample = self.transform(sample)
        return sample


def cal_cls_weights():
    from utils.calculate_class_weights import calculate_class_weights
    from torch.utils.data import DataLoader

    dset = RailwayObstacle(root='/datasets/rs_segment/railway',
                           split='train',
                           base_size=(960, 540), crop_size=(512, 512))
    dset.transform = tr.get_transform('test')

    dataloader = DataLoader(dset, batch_size=4, shuffle=False, num_workers=4)
    calculate_class_weights(dataloader, num_classes=3)


def vis_data():
    from utils.vis import plt_img_target
    import random

    dset = RailwayObstacle(root='/datasets/rs_segment/railway', split='train',
                           base_size=(960, 540), crop_size=(400, 400))
    # dset = RailwayObstacle(root='/datasets/rs_segment/railway', split='valid')
    print(len(dset))
    for _ in range(100):
        idx = random.randint(0, len(dset))
        sample = dset[idx]
        print(idx, sample['filename'])
        img, target = sample['img'], sample['target']
        img = tr.recover_color_img(img)
        target = target.numpy()
        target[target == tr.BG_INDEX] = 0
        plt_img_target(img, target, dset.label_colors)


if __name__ == '__main__':
    # cal_cls_weights()
    vis_data()
