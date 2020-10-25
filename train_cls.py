import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, model_urls
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.resnet import resnet32
from datasets.rs_classify.NWPU46 import NWPU46
from utils.misc import get_curtime, AverageMeter

from tqdm import tqdm


def load_model(arch, num_classes):
    if arch == 'resnet32':
        model = resnet32(num_classes=num_classes)
    elif arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=num_classes)
        state_dict = load_state_dict_from_url(model_urls['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        # strict=False 只适用于 layer 增加/缺少 这样的情况;不适用于 layer(key) 相同而 praram shape 不同
        model.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError
    print(f'load {arch} model..')
    return model


def build_datasets():
    base_dir = '/datasets/rs_classify/NWPU_RESISC46'
    trainset = NWPU46(base_dir, split='train', num_per_class=500)
    testset = NWPU46(base_dir, split='test', num_per_class=200)
    return trainset, testset


def train():
    model.train()
    train_losses = AverageMeter()
    tbar = tqdm(trainloader)

    right_cnt, total_cnt = 0., 0.

    for sample in tbar:
        img, target = sample['img'].cuda(), sample['target'].cuda()

        # loss & optimize
        optimizer.zero_grad()
        output = model(img)  # B,C
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_losses.update(loss.item())

        # acc
        pred = output.argmax(dim=1)
        right_cnt += (pred == target).sum().item()
        total_cnt += bs
        acc = right_cnt / total_cnt
        tbar.set_description('Epoch {}, train loss: {:.3f}, acc: {:.3f}'.format(epoch, train_losses.avg, acc))

    writer.add_scalars('Loss', {'train': train_losses.avg}, epoch)
    writer.add_scalars('Acc', {'train': acc}, epoch)


@torch.no_grad()
def valid():
    model.eval()
    valid_losses = AverageMeter()
    tbar = tqdm(testloader)

    right_cnt, total_cnt = 0., 0.

    for sample in tbar:
        img, target = sample['img'].cuda(), sample['target'].cuda()

        # loss
        output = model(img)  # B,C
        loss = criterion(output, target)
        valid_losses.update(loss.item())

        # acc
        pred = output.argmax(dim=1)
        right_cnt += (pred == target).sum().item()
        total_cnt += bs
        acc = right_cnt / total_cnt
        tbar.set_description('Epoch {}, valid loss: {:.3f}, acc: {:.3f}'.format(epoch, valid_losses.avg, acc))

    writer.add_scalars('Loss', {'valid': valid_losses.avg}, epoch)
    writer.add_scalars('Acc', {'valid': acc}, epoch)

    return acc


if __name__ == '__main__':
    trainset, testset = build_datasets()

    bs = 128
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    arch = 'resnet50'
    model = load_model(arch, num_classes=46).cuda()
    # model = torch.nn.DataParallel(model)

    save_path = os.path.join('runs', 'railway', f'cls_{arch}_bs{bs}_{get_curtime()}')
    writer = SummaryWriter(save_path)

    num_epochs = 60

    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(trainloader))
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90])

    eval_interval = 1
    criterion = nn.CrossEntropyLoss().cuda()

    best_acc = 0.
    for epoch in range(1, num_epochs + 1):
        train()
        if epoch % eval_interval == 0:
            acc = valid()
            if acc >= best_acc:
                print('save model..')
                best_acc = acc
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'acc': acc
                }, os.path.join(save_path, 'model_best.pth'))
