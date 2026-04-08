import warnings
from utils97.dataLoader import getDataLoader
from utils97.utils import getDatasetInfo, seed_torch
from utils97.logger import saveJSONFile

warnings.filterwarnings("ignore")
import argparse
from utils97.utils import getDevice

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch code for EGWA_OSDA')

    parser.add_argument('--log_name', type=str, default='EGWA_OSDA')
    parser.add_argument('--source_dataset', choices=['Houston13_7gt', 'PaviaU_7gt'], default='Houston13_7gt')
    parser.add_argument('--source_known_classes', type=list, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--target_dataset', choices=['Houston18_7gt', 'PaviaC_OS'], default='Houston18_7gt')
    parser.add_argument('--target_known_classes', type=list, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--target_unknown_classes', type=list, default=[9])
    parser.add_argument('--patch', type=int, default=7)
    parser.add_argument('--train_num', type=int, default=180)
    parser.add_argument('--few_train_num', type=int, default=150)
    parser.add_argument('--draw', type=str, default='True')

    parser.add_argument('--seed', type=int, default=83, metavar='S', help='random seed (default: 0)')

    ## Model Level
    parser.add_argument('--model', type=str, default='EGWA_OSDA')
    parser.add_argument('--net', type=str, default='DCRN_02', metavar='B', help='resnet50, efficientnet, densenet, vgg')
    parser.add_argument('--bottle_neck_dim', type=int, default=256, metavar='B', help='bottle_neck_dim for the classifier network.')
    parser.add_argument('--bottle_neck_dim2', type=int, default=500, metavar='B', help='bottle_neck_dim for the classifier network.')

    ## Iteration Level
    parser.add_argument('--warmup_iter', type=int, default=1000, metavar='S', help='warmup iteration for posterior inference')
    parser.add_argument('--training_iter', type=int, default=50, metavar='S', help='training_iter')
    parser.add_argument('--update_term', type=int, default=5, metavar='S', help='update term for posterior inference')

    ## Loss Level
    parser.add_argument('--threshold', type=float, default=0.97, metavar='fixmatch', help='threshold for fixmatch')
    parser.add_argument('--ls_eps', type=float, default=0.03, metavar='LR', help='label smoothing for classification')

    ## Optimization Level
    parser.add_argument('--update_freq_D', type=int,default=1, metavar='S', help='freq for D in optimization.')
    parser.add_argument('--update_freq_G', type=int, default=1, metavar='S', help='freq for G in optimization.')
    parser.add_argument('--batch', type=int, default=64, metavar='N', help='input batch size for training (default: 32)')
    parser.add_argument('--scheduler', type=str, default='cos', help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='label smoothing for classification')
    parser.add_argument('--e_lr', type=float, default=0.002, metavar='LR', help='label smoothing for classification')
    parser.add_argument('--g_lr', type=float, default=0.1, metavar='LR', help='label smoothing for classification')
    parser.add_argument('--opt_clip', type=float, default=0.1, metavar='LR', help='label smoothing for classification')
    ## etc:
    parser.add_argument('--exp_code', type=str, default='Test', metavar='S', help='random seed (default: 0)')
    parser.add_argument('--result_dir', type=str, default='results', metavar='S', help='random seed (default: 0)')
    parser.add_argument('--set_gpu', type=int, default=0, help='gpu setting 0 or 1')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disable cuda')

    try:
        args = parser.parse_args()
    except:
        args, _ = parser.parse_known_args()

    args.device = getDevice(args.set_gpu)

    if args.source_dataset == 'PaviaU_7gt' and args.target_dataset == 'PaviaC_OS':
        args.source_known_classes = [1,2,3,4,5,6,7]
        args.target_known_classes = [1,2,3,4,5,6,7]
        args.target_unknown_classes = [9]
    elif args.source_dataset == 'Houston13_7gt' and args.target_dataset == 'Houston18_7gt':
        args.source_known_classes = [1,2,3,4,5,6]
        args.target_known_classes = [1,2,3,4,5,6]
        args.target_unknown_classes = [7]

    return args

if __name__ == '__main__':

    args = parse_args()
    seed_torch(args.seed)
    saveJSONFile(
        f'logs/{args.log_name}/{args.log_name} {args.source_dataset}-{args.target_dataset} seed={args.seed}.json', {
            'args': str(args)
        })
    saveJSONFile(f'time/{args.log_name}', {
        'temp': 1
    })

    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    data_loader: dict = getDataLoader(args, source_info, target_info)

    if args.model == 'EGWA_OSDA':
        from models.model_EGWA_OSDA import EGWA_OSDA
        model = EGWA_OSDA(args, source_info, target_info, len(args.source_known_classes) + 1, data_loader)

    import time
    model.train_init()
    model.test(0)
    model.build_model()
    model.train()
