"""
在 NWPU45 基础上 添加 dust 风沙类别
"""
import os
from torch.utils.data import Dataset
from datasets.config.railway import cls_label_names
import torchvision.transforms as tr
from PIL import Image
import random


class NWPU46(Dataset):

    def __init__(self, base_dir, split, num_per_class):
        """
        :param base_dir:
        :param split: train/test
        :param num_per_class: 每类样本选择数量
        """
        self.label_names = cls_label_names
        self.num_classes = 46

        self.img_paths = []
        self.target_paths = []

        self.transform = self.get_transform(split)

        random.seed(100)

        for idx, cat in enumerate(self.label_names):
            sub_dir = os.path.join(base_dir, cat)
            img_list = [os.path.join(sub_dir, p) for p in os.listdir(sub_dir) if p != '@eaDir']
            random.shuffle(img_list)
            num_per_class = min(num_per_class, len(img_list))

            if split == 'train':
                self.img_paths.extend(img_list[:num_per_class])  # train 取头部
            elif split == 'test':
                self.img_paths.extend(img_list[-num_per_class:])  # test 取尾部

            self.target_paths.extend([idx] * num_per_class)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        target = self.target_paths[idx]
        img = self.transform(img)
        return {
            'img': img,
            'target': target
        }

    def __len__(self):
        return len(self.target_paths)

    def get_transform(self, split):
        if split == 'train':
            return tr.Compose([
                tr.Resize(224),
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)),
                tr.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif split == 'test':
            return tr.Compose([
                tr.Resize(224),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


if __name__ == '__main__':
    from utils.misc import recover_color_img
    import matplotlib.pyplot as plt

    dset = NWPU46(base_dir='/datasets/rs_classify/NWPU_RESISC46', split='train', num_per_class=100)

    for sample in dset:
        img = sample['img']
        img = recover_color_img(img)
        plt.imshow(img)
        plt.show()
