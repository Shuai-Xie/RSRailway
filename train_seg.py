import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
from models.ctrseg_net import CTRSEG
from models.hrnet.hrnet import HRNet
from datasets.rs_segment.dataset_railway import RailwaySeg
from datasets.rs_segment.dataset_obstacle import RailwayObstacle
from datasets.rs_segment.config import train_rail_config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import SegmentationMetric
from utils.misc import *
import argparse

label_names = train_rail_config['label_names']


def parse_args():
    parser = argparse.ArgumentParser(description="TrainSeg")
    parser.add_argument('--epochs', default=40, type=int, help="CUDA device.")
    parser.add_argument('--num-classes', default=3, type=int, help="bg,rail,train")
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    args = parser.parse_args()
    return args


def get_eval_interval(cur_epoch, num_epochs):
    if cur_epoch > num_epochs * 0.9:
        return 1
    elif cur_epoch > num_epochs * 0.8:
        return 2
    elif cur_epoch > num_epochs * 0.6:
        return 5
    else:
        return num_epochs


def load_model():
    """
    实验发现，使用 dec model 的 backbone，只更新 seg_head 效果远远不够
    如果 freeze 原始参数，还是用 SENet 结构转换原始 feature
    """
    # load param of dec model
    resume = 'runs/dota/model_50.pth'  # dota demo
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['model_state_dict']
    # seg model
    model = CTRSEG(num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(state_dict_, strict=False)  # 将 key,v.shape 一致的参数 load
    print('load model')
    return model


def train():
    model.train()
    tbar = tqdm(trainloader)
    losses = AverageMeter()

    for sample in tbar:
        image, label = sample['img'].cuda(non_blocking=True), sample['target'].cuda(non_blocking=True)

        optimizer.zero_grad()

        output = model(image)
        loss = criterion(output, label.long())  # + lovasz_softmax(F.softmax(output,dim=1), label.long())
        loss.backward()

        optimizer.step()

        losses.update(loss.item())
        tbar.set_description('Epoch: {}, train loss: {:.3f}'.format(epoch, losses.avg))

    writer.add_scalar('Train/loss', losses.avg, epoch)


@torch.no_grad()
def valid():
    model.eval()
    tbar = tqdm(validloader)
    losses = AverageMeter()

    metrics.reset()

    for sample in tbar:
        image, label = sample['img'].cuda(non_blocking=True), sample['target'].cuda(non_blocking=True)
        output = model(image)
        loss = criterion(output, label.long())
        predict = torch.argmax(output, dim=1)
        metrics.add_batch(label.data.cpu().numpy(), predict.data.cpu().numpy())

        losses.update(loss.item())
        tbar.set_description('Epoch: {}, valid loss: {:.3f}'.format(epoch, losses.avg))

    ious = metrics.Intersection_over_Union_Class()
    mIoU = metrics.Mean_Intersection_over_Union(ious)
    fwIoU = metrics.Frequency_Weighted_Intersection_over_Union(ious)

    writer.add_scalar('Valid/loss', losses.avg, epoch)
    writer.add_scalars('Valid/eval', {
        'mIoU': mIoU,
        'fwIoU': fwIoU,
    }, epoch)
    writer.add_scalars('Valid/iou', {
        '{:0>2d}_{}'.format(i + 1, label_names[i]): ious[i] for i in range(args.num_classes)
    }, epoch)

    return fwIoU, mIoU, ious


def build_dataloaders():
    trainset = RailwayObstacle(root=f'/datasets/rs_segment/railway', split='train',
                               base_size=(960, 540), crop_size=(400, 400))
    validset = RailwayObstacle(root=f'/datasets/rs_segment/railway', split='valid')

    trainloader = DataLoader(trainset, args.batchsize, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    validloader = DataLoader(validset, args.batchsize, shuffle=False, pin_memory=True, num_workers=4)
    return trainloader, validloader


if __name__ == '__main__':
    args = parse_args()

    # data
    trainloader, validloader = build_dataloaders()

    # model
    model = HRNet(
        # cfg_path='models/hrnet/cfg/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
        # cfg_path='models/hrnet/cfg/seg_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        cfg_path='models/hrnet/cfg/seg_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        # cfg_path='models/hrnet/cfg/seg_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
        num_classes=args.num_classes,
        use_pretrain=True
    )
    model.cuda()
    print('build model...')

    # optimizer
    lr = 5e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # loss
    use_weight = False
    weights = torch.from_numpy(np.array(train_rail_config['weights'])).float().cuda() if use_weight else None
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255).cuda()

    # metric
    metrics = SegmentationMetric(args.num_classes)

    # writer
    save_dir = os.path.join('runs', 'railway',
                            f'seg_hrnet30_fg_scale_cls{args.num_classes}_epoch{args.epochs}_hrnet_{get_curtime()}')
    writer = SummaryWriter(log_dir=save_dir)
    print('create', save_dir)

    best_eval = 0.

    for epoch in range(1, args.epochs + 1):
        train()
        lr_scheduler.step()

        if epoch > 20:
            # if epoch % get_eval_interval(epoch, args.epochs) == 0:
            fwIoU, mIoU, ious = valid()
            if mIoU > best_eval:
                best_eval = mIoU
                state = {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'fwIoU': fwIoU,
                    'mIoU': mIoU,
                    'ious': ious
                }
                torch.save(state, os.path.join(save_dir, 'model_best.pth'))
                print('save checkpoint at', epoch)
