from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import torch
import torch.nn.functional as F

BG_INDEX = 255


def mapbg(bg_idx):
    """
    image bg 转成 constants.BG_INDEX, 类别从 [0,..,C-1]
    """

    # bg 在首部，需要调整 实际类别 前移1位
    def map_headbg(target):
        target = target.astype(int)
        target -= 1  # 1->0
        target[target == -1] = BG_INDEX
        return target.astype('uint8')

    # bg 在尾部，直接替换为 constant 即可
    def map_other(target):
        target = target.astype(int)
        target[target == bg_idx] = BG_INDEX
        return target.astype('uint8')

    if bg_idx == 0:
        return map_headbg
    else:
        return map_other


def remap(bg_idx):
    """
    分割结果 -> 回归原始 bg idx，方面 vis
    """

    def remap_headbg(target):
        target = target.astype(int)
        target += 1
        target[target == BG_INDEX + 1] = bg_idx
        return target.astype('uint8')

    def remap_other(target):
        target = target.astype(int)
        target[target == BG_INDEX] = bg_idx
        return target.astype('uint8')

    if bg_idx == 0:
        return remap_headbg
    else:
        return remap_other


class Compose:  # 可以采用 默认的
    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, sample):
        for t in self.trans_list:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('

        for t in self.trans_list:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class RandomHorizontalFlip:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 0.5:
            img = img.transpose(0)
            target = target.transpose(0)

        sample['img'], sample['target'] = img, target
        return sample


class RandomVerticalFlip:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 0.5:
            img = img.transpose(1)
            target = target.transpose(1)

        sample['img'], sample['target'] = img, target
        return sample


class RandomRightAngle:
    """随机旋转直角, 90/180/270"""

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 0.5:
            k = random.randint(2, 4)
            img = img.transpose(k)
            target = target.transpose(k)

        sample['img'], sample['target'] = img, target
        return sample


class RandomDiagnoal:
    """随机对角线转换，主/副"""

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        if random.random() < 10:
            k = random.randint(5, 6)  # 闭区间
            img = img.transpose(k)
            target = target.transpose(k)

        sample['img'], sample['target'] = img, target
        return sample


class RandomRotate:
    def __init__(self, degree):  # 旋角上限
        self.degree = degree

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        target = target.rotate(rotate_degree, Image.NEAREST, fillcolor=BG_INDEX)

        sample['img'], sample['target'] = img, target
        return sample


class RandomFGCrop:
    """
    避免 crop 出 包含 bg 太多的 patch
    random base_num = grid_size **2
    """

    def __init__(self, base_size, crop_size, over_lap=0.5, fg_thre=0.1, scales=(0.8, 1.2)):
        self.base_size = min(base_size)
        self.crop_size = min(crop_size)
        self.scales = scales

        self.stride = int(self.crop_size * (1 - over_lap))
        self.grid_size = 1 + (base_size[0] - self.crop_size) // self.stride  # 宽度

        self.fg_thre = fg_thre
        self.bg_idx = 0

    def _binary_target(self, target):
        target[target != self.bg_idx] = 1
        return target

    def _rand_fg_coord(self, target):
        tmp = np.array(target)
        tmp = self._binary_target(tmp)
        tmp = torch.from_numpy(tmp).unsqueeze(0).unsqueeze(0).float()  # 为了用 avgpool
        fg_rates = F.avg_pool2d(tmp,
                                kernel_size=self.crop_size,
                                stride=self.stride).reshape(-1).numpy()  # 1D fg_rate

        top_idxs = np.argsort(fg_rates)[::-1]
        keep_num = np.sum(fg_rates >= self.fg_thre)
        choose = random.choice(top_idxs[:keep_num]) if keep_num > 0 else top_idxs[0]  # HZ v1
        return choose, fg_rates[choose]

    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        chose_idx, fg_rate = self._rand_fg_coord(target)

        # 从 chose_idx 还原 crop patch 坐标
        i, j = chose_idx // self.grid_size, chose_idx % self.grid_size
        x1, y1 = self.stride * j, self.stride * i

        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # resize
        short_size = random.randint(int(self.crop_size * self.scales[0]), int(self.crop_size * self.scales[1]))
        ow = oh = short_size
        # print('short_size:', short_size)
        # random scale
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)

        # scale 后短边 < 要 crop 尺寸，补图
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # img fill 0, 后面还有 normalize
            # target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=BG_INDEX)  # target fill bg_idx
            target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=BG_INDEX)  # target fill bg_idx
        else:
            x1 = random.randint(0, short_size - self.crop_size)
            y1 = random.randint(0, short_size - self.crop_size)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample['img'], sample['target'] = img, target
        return sample


class RandomScaleCrop:
    def __init__(self, base_size, crop_size, scales=(0.8, 1.2)):
        self.base_size = min(base_size)
        self.crop_size = min(crop_size)
        self.scales = scales

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        # 原图 scale
        short_size = random.randint(int(self.base_size * self.scales[0]), int(self.base_size * self.scales[1]))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # random scale
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)

        # scale 后短边 < 要 crop 尺寸，补图
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # img fill 0, 后面还有 normalize
            # target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=BG_INDEX)  # target fill bg_idx
            target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=BG_INDEX)  # target fill bg_idx

        # random crop
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample['img'], sample['target'] = img, target
        return sample


class ColorJitter:
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])

        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)

        sample['img'], sample['target'] = img, target
        return sample


class Normalize:
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        img = np.array(img).astype(np.float32)
        target = np.array(target).astype(np.float32)

        # img = img / 255. - 0.5  # [-0.5,0.5], mean=0.5, std=1. ?

        img /= 255.0
        img -= self.mean
        img /= self.std

        sample['img'], sample['target'] = img, target
        return sample


class ToTensor:
    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        target = np.array(target).astype(np.float32)

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).long()

        sample['img'], sample['target'] = img, target
        return sample


def get_transform(split, base_size=None, crop_size=None):
    if split == 'train':
        return Compose([
            # sampler
            # RandomScaleCrop(base_size, crop_size, scales=(0.5, 1.0)),
            RandomFGCrop(base_size, crop_size, scales=(0.5, 1.5)),
            # flip
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRightAngle(),
            RandomDiagnoal(),
            # color
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            # normal
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
    elif split == 'valid':
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
    elif split == 'test':
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


import torchvision.transforms as transforms


def get_test_transfrom(base_size=None):
    trans = [transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    if base_size is not None:
        trans.insert(0, transforms.Resize(base_size))
    return transforms.Compose(trans)


def recover_color_img(img):
    """
    cvt tensor image to RGB [note: not BGR]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.transpose(img, axes=[1, 2, 0])  # h,w,c
    img = img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    # img = img + 0.5
    img = (img * 255).astype('uint8')
    return img
