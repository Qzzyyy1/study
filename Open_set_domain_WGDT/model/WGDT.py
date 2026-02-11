import torch
import torch.nn as nn
from torchmetrics import Accuracy

from utils.dann import DomainDiscriminator, DomainAdversarialLoss
from utils.meter import OpensetDomainMetric
from utils.Trainer import Trainer
from utils.dataLoader import CombinedLoader
from utils.utils import mergeArgs
from .DCRN import DCRN
from .Anchor import Anchor
from .Radius import Radius

class Model(nn.Module):
    def __init__(self, args, source_info, target_info, device, in_channels, patch, known_num_classes, anchor_type, radius_loss_type, radius_init, radius_margin, alpha, domain_loss_weight, radius_loss_weight, pseudo_label_weight, pseudo_label_threshold):
        super().__init__()

        self.args = args
        self.source_info = source_info
        self.target_info = target_info
        self.device = device

        self.in_channels = in_channels
        self.patch = patch
        self.known_num_classes = known_num_classes
        self.alpha = alpha
        self.domain_loss_weight = domain_loss_weight
        self.radius_loss_weight = radius_loss_weight
        self.pseudo_label_weight = pseudo_label_weight
        self.pseudo_label_threshold = pseudo_label_threshold

        self.feature_encoder = DCRN(in_channels, patch, known_num_classes)
        self.classifier = nn.Linear(288, known_num_classes)
        self.disc_encoder = DomainDiscriminator(in_feature=288, hidden_size=64)
        self.domain_adv = DomainAdversarialLoss(self.disc_encoder)
        self.anchor = {
            'anchor': lambda: Anchor(known_num_classes, anchor_weight=args.anchor_weight, alpha=self.alpha),
        }[anchor_type]()
        self.radius = Radius(radius_init, radius_margin, radius_loss_type)

        self.source_oa = Accuracy()
        self.metric = OpensetDomainMetric(self.known_num_classes, self.args)
        self.prediciton_all = []

    def pre_train_step(self, batch):
        x, y = batch
        loss = self(x, y)['loss']

        return loss

    def pre_train_optimizer(self):
        optimizer = torch.optim.SGD([
            {'params': self.feature_encoder.parameters()},
            {'params': self.classifier.parameters()},
            {'params': self.anchor.parameters()}
        ], lr=0.001, momentum=0.9, weight_decay=5e-4)

        return optimizer

    def train_step(self, batch):
        [source_x, source_y], [target_x, target_y] = batch

        source_out = self(source_x, source_y)
        target_out = self(target_x)

        # 获取 gamma (用于筛选) 和 distance (用于计算损失)
        gamma = target_out['gamma']
        distance = target_out['distance']
        
        min_gamma = gamma.min(1)[0]
        min_distance = distance.min(1)[0].detach()
        weight = (min_distance.max() - min_distance) / (min_distance.max() - min_distance.min() + 1e-6)

        loss_disc = self.domain_adv(source_out['features'], target_out['features'], w_t=weight)
        loss_radius = self.radius(min_gamma.detach(), weight=weight)

        # ========== 伪标签自训练（基于度量学习） ==========
        loss_pseudo = torch.tensor(0.0, device=self.device)
        loss_pseudo_anchor = torch.tensor(0.0, device=self.device)
        loss_pseudo_tuplet = torch.tensor(0.0, device=self.device)
        num_pseudo = 0

        # 1. 筛选高置信度的已知类样本
        with torch.no_grad():
            # ✅ 修改 1: 使用 gamma 作为置信度评分，与测试阶段保持一致
            score, pseudo_labels = torch.min(gamma, 1) 
            
            radius_value = self.radius.radius.item()
            # 筛选条件：gamma 小于半径 (已知类) 且 gamma 小于严格阈值 (高置信)
            confident_mask = (score < radius_value) & (score < self.pseudo_label_threshold)
            num_pseudo = confident_mask.sum().item()

        # 2. 计算伪标签损失
        if num_pseudo > 0:
            # 获取对应的距离矩阵 (Batch_Pseudo, Num_Classes)
            # 注意：虽然筛选用 gamma，但计算损失必须用 distance
            confident_distances = distance[confident_mask]
            confident_pseudo_labels = pseudo_labels[confident_mask]

            # --- A. Anchor Loss ---
            # 获取样本到其“伪标签锚点”的距离
            true_distances = torch.gather(confident_distances, 1, confident_pseudo_labels.view(-1, 1)).view(-1)
            loss_pseudo_anchor = torch.mean(true_distances)

            # --- B. Tuplet Loss (向量化优化版) ---
            # ✅ 修改 2: 移除列表推导式，使用掩码屏蔽掉 true_class
            # 创建一个全 1 掩码
            mask = torch.ones_like(confident_distances, dtype=torch.bool)
            # 将 true class 的位置设为 False
            mask.scatter_(1, confident_pseudo_labels.view(-1, 1), False)
            
            # 选出非伪标签类别的距离 (Flatten 后 reshape)
            # Shape: [Num_Pseudo, Num_Classes - 1]
            other_distances = confident_distances[mask].view(num_pseudo, -1)

            # 计算 Tuplet 公式: log(1 + sum(exp(true - other)))
            # true_distances.unsqueeze(1) shape: [Num_Pseudo, 1]
            # other_distances shape: [Num_Pseudo, Num_Classes - 1]
            tuplet = torch.exp(true_distances.unsqueeze(1) - other_distances)
            loss_pseudo_tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

            # 汇总伪标签损失
            loss_pseudo = self.alpha * loss_pseudo_anchor + loss_pseudo_tuplet

        # ========== 损失汇总 ==========
        loss_dic = dict(
            loss_anchor = source_out['loss_anchor'] * self.alpha,
            loss_tuplet = source_out['loss_tuplet'],
            loss_disc = loss_disc * self.domain_loss_weight,
            loss_radius = loss_radius * self.radius_loss_weight,
            loss_pseudo = loss_pseudo * self.pseudo_label_weight
        )

        self.source_oa.update(source_out['prediction'], source_y)

        return {
            'loss': sum(value for value in loss_dic.values()),
            'information': {
                **loss_dic, 
                'num_pseudo': num_pseudo,
                # 为了防止 log 报错，转换一下 tensor
                'loss_pseudo_anchor': loss_pseudo_anchor.item(),
                'loss_pseudo_tuplet': loss_pseudo_tuplet.item()
            }
        }
    
    def train_epoch_end(self):
        dic = {
            'source_oa': self.source_oa.compute()
        }
        self.source_oa.reset()

        return dic

    def train_optimizer(self):
        momentum = 0.9
        l2_decay = 5e-4
        optimizer = torch.optim.SGD([
            {'params': self.feature_encoder.parameters()},
            {'params': self.classifier.parameters()},
            {'params': self.anchor.parameters()}
        ], lr=self.args.lr_encoder, momentum=momentum, weight_decay=l2_decay)
        optimizer_critic = torch.optim.SGD(self.disc_encoder.parameters(), lr=self.args.lr_domain, momentum=momentum, weight_decay=l2_decay)
        optimizer_raiuds = torch.optim.SGD(self.radius.parameters(), lr=1e-4, momentum=momentum, weight_decay=l2_decay)

        return optimizer, optimizer_critic, optimizer_raiuds

    def test_step(self, batch):
        x, y = batch

        out = self(x)
        prediction = out['prediction']
        gamma = out['gamma']

        score, prediction = torch.min(gamma, 1)
        radius_value = self.radius.radius.item()  # 将Parameter转为标量
        prediction[score > radius_value] = self.known_num_classes

        self.metric.update(prediction, y)
    
    def test_end(self):
        self.metric.finish()

    def prediction_step(self, batch):
        x = batch

        out = self(x)
        gamma = out['gamma']
        score, prediction = torch.min(gamma, 1)

        radius_value = self.radius.radius.item()  # 将Parameter转为标量
        prediction[score > radius_value] = self.known_num_classes
        self.prediciton_all.append(prediction)

    def prediction_end(self):
        from utils.draw import drawPredictionMap

        drawPredictionMap(self.prediciton_all, f'{self.args.log_name} {self.args.target_dataset}', self.target_info, known_classes=self.args.target_known_classes, unknown_classes=self.args.target_unknown_classes, draw_background=False)

    def forward(self, x, y=None):
        features = self.feature_encoder(x)['features']
        logits = self.classifier(features)
        anchor_out = self.anchor(logits, y)

        return {
            'features': features,
            'logits': logits,
            **anchor_out
        }


def run_model(model: Model, data_loader: dict):
    trainer = Trainer(model, model.device)

    if model.args.pre_train == 'True':
        trainer.train('pre_train', data_loader['source']['train'], model.args.pre_train_epochs)
    trainer.train('train', CombinedLoader([data_loader['source']['train'], data_loader['target']['train']]), model.args.epochs)

    trainer.test('test', data_loader['target']['test'])
    
    if model.args.draw == 'True':
        trainer.test('prediction', data_loader['target']['all'])

def get_model(args, source_info, target_info):
    from utils.utils import getDevice

    model_args = {
        'args': args,
        'source_info': source_info,
        'target_info': target_info, 
        'device': getDevice(args.device),
        'in_channels': args.pca if hasattr(args, 'pca') and args.pca > 0 else source_info.bands_num,
        'patch': args.patch,
        'known_num_classes': len(args.source_known_classes),
        'anchor_type': args.anchor_type,
        'radius_loss_type': args.radius_loss_type,
        'radius_init': args.radius_init,
        'radius_margin': args.radius_margin,
        'alpha': args.alpha,
        'domain_loss_weight': args.domain_loss_weight,
        'radius_loss_weight': args.radius_loss_weight,
        'pseudo_label_weight': args.pseudo_label_weight,
        'pseudo_label_threshold': args.pseudo_label_threshold,
    }
    return Model(**model_args)

def parse_args():
    import argparse

    # seed
    # PaviaU_7gt -> PaviaC_OS [7, 43, 45, 46, 66, 67, 77, 81, 88, 98]
    # Houston13_7gt -> Houston18_OS [1, 23, 30, 35, 52, 64, 68, 72, 73, 91]
    # HyRank_source -> HyRank_target [6, 13, 20, 40, 49, 58, 60, 67, 81, 91]
    # Yancheng_ZY ->'Yancheng_GF [15, 18, 19, 24, 37, 50, 51, 62, 77, 80]

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='WGDT')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_num', type=int, default=180)
    parser.add_argument('--few_train_num', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patch', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--source_dataset', choices=['Houston13_7gt', 'PaviaU_7gt', 'HyRank_source', 'Yancheng_ZY'], default='PaviaU_7gt')
    parser.add_argument('--target_dataset', choices=['Houston18_OS', 'PaviaC_OS', 'HyRank_target', 'Yancheng_GF'], default='PaviaC_OS')
    parser.add_argument('--anchor_type', type=str, choices=['anchor'], default='anchor'),
    parser.add_argument('--radius_loss_type', type=str, choices=['MarginMSELoss'], default='MarginMSELoss')
    parser.add_argument('--radius_init', type=float, default=0.0)

    parser.add_argument('--pre_train', type=str, default='True')
    parser.add_argument('--pre_train_epochs', type=int, default=20)

    parser.add_argument('--draw', type=str, default='False')

    # 伪标签自训练参数
    parser.add_argument('--pseudo_label_weight', type=float, default=0.5, help='伪标签损失权重')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.3, help='伪标签置信度阈值（距离）')

    args = parser.parse_args()

    mergeArgs(args, args.target_dataset)

    return args