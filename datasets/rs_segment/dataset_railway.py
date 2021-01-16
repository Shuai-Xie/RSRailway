from torch.utils.data import Dataset
import os
from PIL import Image
import datasets.rs_segment.transforms as tr
import numpy as np
from utils.misc import read_txt_as_list


class RailwaySeg(Dataset):

    def __init__(self, root, split, base_size=None, crop_size=None):
        data_dir = os.path.join(root, split)

        self.img_paths = read_txt_as_list(os.path.join(data_dir, f'{split}_img_paths.txt'))
        self.target_paths = read_txt_as_list(os.path.join(data_dir, f'{split}_target_paths.txt'))

        self.num_class = 7
        self.bg_idx = 0
        self.mapbg_fn = tr.mapbg(self.bg_idx)
        self.remap_fn = tr.remap(self.bg_idx)

        self.transform = tr.get_transform(split, base_size, crop_size)

        self.label_names = ['bg', 'rail', 'plant', 'buildings', 'road', 'land', 'water', 'train']
        self.label_colors = [
            (0, 0, 0),
            (0, 0, 255),  # rail=1
            (0, 255, 0),
            (255, 0, 0),
            (255, 0, 255),  # road 公路   粉
            (255, 255, 0),  # land 黄土地  黄
            (0, 255, 255),  # water=6
            (128, 128, 128),  # train=7
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        target = np.asarray(Image.open(self.target_paths[index]), dtype=int)
        target = self.mapbg_fn(target)
        target = Image.fromarray(target)

        sample = {
            'img': img,
            'target': target,
            'filename': os.path.basename(self.img_paths[index])
        }
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    from utils.misc import plt_img_target
    import random

    dset = RailwaySeg(root='/datasets/rs_segment/railway',
                      split='train',
                      base_size=(960, 540), crop_size=(512, 512))

    for _ in range(20):
        idx = random.randint(0, len(dset))
        sample = dset[idx]
        print(idx, sample['filename'])
        img, target = sample['img'], sample['target']
        img = tr.recover_color_img(img)
        target = target.numpy()
        target = dset.remap_fn(target)
        plt_img_target(img, target, dset.label_colors)
