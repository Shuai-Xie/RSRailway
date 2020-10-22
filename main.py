import argparse
import train
import test
import eval
from datasets.rs_detect.dataset_dota import DOTA
from datasets.rs_detect.dataset_railway import Railway
from models import ctrbox_net
import utils.decoder as decoder
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def parse_args(params=None):
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--exp', type=str, default='', help='Data directory')
    parser.add_argument('--num_epoch', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=608, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.18, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='../Datasets/dota', help='Data directory')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    args = parser.parse_args(params)
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA, 'railway': Railway}
    num_classes = {'dota': 15, 'railway': 17}
    heads = {
        'hm': num_classes[args.dataset],  # heatmap
        'wh': 10,  # box param
        'reg': 2,  # offset
        'cls_theta': 1,  # orientation class
    }
    down_ratio = 4
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_channels=256)

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    if args.phase == 'train':
        ctrbox_obj = train.TrainModule(dataset=dataset, num_classes=num_classes,
                                       model=model, decoder=decoder,
                                       down_ratio=down_ratio)
        ctrbox_obj.train_network(args)
    elif args.phase == 'test':  # see results
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes,
                                     model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes,
                                     model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)
