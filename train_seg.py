import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
from models.ctrseg_net import CTRSEG
from datasets.rs_segment.dataset_railway import RailwaySeg
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import SegmentationMetric
from utils.misc import *

num_classes = 7


def load_model():
    # load param of dec model
    resume = 'runs/dota/model_50.pth'  # dota demo
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['model_state_dict']
    # seg model
    model = CTRSEG(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict_, strict=False)  # 将 key,v.shape 一致的参数 load
    print('load model')
    return model


def train():
    # home = '/home/xs/data'
    home = '/datasets'
    dataset = RailwaySeg(root=f'{home}/rs_segment/railway',
                         split='train',
                         base_size=(960, 540), crop_size=(448, 448))
    dataloader = DataLoader(dataset,
                            batch_size=4, shuffle=True,
                            pin_memory=True, num_workers=4)

    # 实验发现，使用 dec model 的 backbone，只更新 seg_head 效果远远不够
    # 如果 freeze 原始参数，还是用 SENet 结构转换原始 feature
    # model = load_model().cuda()
    # train_params = [{'params': model.get_train_params(), 'lr': lr}]

    model = CTRSEG(num_classes=num_classes, pretrained=True).cuda()

    num_epochs = 100

    save_path = os.path.join('runs', 'railway', f'seg_cls7_res101_data100_{get_curtime()}')
    writer = SummaryWriter(log_dir=save_path)

    lr = 1e-3
    train_params = [{'params': model.parameters(), 'lr': lr}]
    optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dataloader))

    # Adam 比 SGD 效果差
    # lr = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    evaluator = SegmentationMetric(num_class=num_classes)

    eval_interval = 5

    best_FWIoU = 0.

    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.train()

    for epoch in range(1, num_epochs + 1):

        tbar = tqdm(dataloader)
        train_losses = AverageMeter()

        evaluation = epoch % eval_interval == 0

        for sample in tbar:
            image, target = sample['img'].cuda(), sample['target'].cuda()
            output = model(image)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            train_losses.update(loss.item())
            tbar.set_description('Epoch {}, Train loss: {:.3f}'.format(epoch, train_losses.avg))

            if evaluation:
                pred = torch.argmax(output, dim=1)
                evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())

        writer.add_scalar('loss', train_losses.avg, epoch)

        if evaluation:
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            writer.add_scalar('FWIoU', FWIoU, epoch)
            print('FWIoU:', FWIoU)

            if FWIoU > best_FWIoU:
                print('saving model...')
                best_FWIoU = FWIoU

                if isinstance(model, torch.nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()

                state = {
                    'epoch': epoch,
                    'FWIoU': FWIoU,
                    'state_dict': state_dict,
                }
                torch.save(state, os.path.join(save_path, 'model_best.pth'))


if __name__ == '__main__':
    train()
