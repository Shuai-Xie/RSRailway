import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.misc import get_curtime
from utils.loss import LossAll
import utils.func_utils as func_utils


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict


class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {
            'dota': ['train'],
            'railway': ['train']
        }
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']

        # 从 state_dict_ 解析出参数
        state_dict = {}
        for k in state_dict_:
            # 多 GPU 训练 key, module
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]

        # override model param
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()
        epoch = checkpoint['epoch']

        return model, optimizer, epoch

    def train_network(self, args):
        optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
        save_path = os.path.join('runs', args.dataset, f'{args.exp}_{get_curtime()}')
        writer = SummaryWriter(save_path)

        start_epoch = 1

        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            model, optimizer, start_epoch = self.load_model(self.model,
                                                            optimizer,
                                                            args.resume_train)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])  # todo: gpus
        self.model.cuda()

        criterion = LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]  # get dataset cls

        # datasets
        dsets = {
            x: dataset_module(data_dir=args.data_dir,
                              phase=x,  # train
                              input_h=args.input_h,
                              input_w=args.input_w,
                              down_ratio=self.down_ratio)
            for x in self.dataset_phase[args.dataset]
        }

        dsets_loader = {
            'train': torch.utils.data.DataLoader(dsets['train'],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 drop_last=True,
                                                 collate_fn=collater)
        }

        print('Starting training...')

        best_loss = 100.

        for epoch in range(start_epoch, args.num_epoch + 1):
            print('-' * 10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion,
                                        optimizer=optimizer)
            scheduler.step()  # 每个 epoch 变换一次 lr

            writer.add_scalar('train_loss', epoch_loss, global_step=epoch)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_model(os.path.join(save_path, 'model_best.pth'),
                                epoch,
                                self.model,
                                optimizer)

            # test 测试模型
            if 'test' in self.dataset_phase[args.dataset] and epoch % 5 == 0:
                mAP = self.dec_eval(args, dsets['test'])
                writer.add_scalar('mAP', mAP, global_step=epoch)

    def run_epoch(self, phase, data_loader, criterion, optimizer):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.

        # note: item
        tbar = tqdm(data_loader)
        for i, data_dict in enumerate(tbar):
            for name in data_dict:
                data_dict[name] = data_dict[name].cuda()
            if phase == 'train':
                optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])  # dict
                    loss = criterion(pr_decs, data_dict)
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)

            running_loss += loss.item()
            tbar.set_description('{} loss: {:.3f}'.format(phase, running_loss / (i + 1)))

        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss

    def dec_eval(self, args, dsets):
        result_path = 'result_' + args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model, dsets,
                                 self.down_ratio,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap
