from __future__ import print_function
from utils97 import utils1
from utils97 import utils
from utils97.utils1 import OptimWithSheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.function import HLoss
from models.basenet import *
from utils97.utils1 import inverseDecayScheduler, CosineScheduler, StepScheduler, ConstantScheduler
from models.function import CrossEntropyLoss
from models.function import BetaMixture1D
import copy

from models.basenet import *

from train import parse_args
from utils97.dataLoader import getDataLoader
from utils97.utils import getDatasetInfo

class Model(nn.Module):
    def __init__(self, args, source_info, target_info, num_class):
        super().__init__()

        self.model = 'UADAL'
        self.args = args
        self.info = {
            'source': source_info,
            'target': target_info
        }

        self.all_num_class = num_class
        self.known_num_class = num_class - 1
        self.device = self.args.device

        self.build_model_init()
        self.ent_criterion = HLoss()
        self.bmm_model = self.cont = self.k = 0
        self.bmm_model = BetaMixture1D()
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)
        self.bmm_update_cnt = 0

    def build_model_init(self):
        self.G, self.E, self.C = utils1.get_model_init(self.args, self.info, known_num_class=self.known_num_class, all_num_class=self.all_num_class)
        self.G.to(self.args.device)
        self.E.to(self.args.device)
        self.C.to(self.args.device)

        scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75, max_iter=self.args.warmup_iter)

        if 'vgg' == self.args.net:
            for name, param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_w_g = OptimWithSheduler(optim.SGD(params, lr=self.args.g_lr * self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_w_e = OptimWithSheduler(optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_w_c = OptimWithSheduler(optim.SGD(self.C.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)

    def build_model(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)

        _, _, self.E, self.DC = utils.get_model(self.args, self.info, known_num_class=self.known_num_class, all_num_class=self.all_num_class, domain_dim=3)

        self.DC.apply(weights_init_bias_zero)

        self.E.to(self.args.device)
        self.DC.to(self.args.device)

        SCHEDULER = {'cos': CosineScheduler, 'step': StepScheduler, 'id': inverseDecayScheduler, 'constant':ConstantScheduler}
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75, max_iter=self.num_batches*self.args.training_iter)
        scheduler_dc = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75, max_iter=self.num_batches*self.args.training_iter*self.args.update_freq_D)

        if 'vgg' == self.args.net:
            for name,param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_g = OptimWithSheduler(optim.SGD(params, lr=self.args.g_lr * self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_c = OptimWithSheduler(optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_dc = OptimWithSheduler(optim.SGD(self.DC.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_dc)
        scheduler_e = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,max_iter=self.num_batches*self.args.training_iter)
        self.opt_e = OptimWithSheduler(optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_e)

    def network_initialization(self):
        if 'resnet' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()
        elif 'vgg' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()

    def freeze_GE(self):
        self.G_freezed = copy.deepcopy(self.G)
        self.E_freezed = copy.deepcopy(self.E)

    def compute_probabilities_batch(self, out_t, unk=1):
        ent_t = self.ent_criterion(out_t)
        batch_ent_t = (ent_t - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss + 1e-6)
        batch_ent_t[batch_ent_t >= 1 - 1e-4] = 1 - 1e-4
        batch_ent_t[batch_ent_t <=  1e-4] = 1e-4
        B = self.bmm_model.posterior(batch_ent_t.clone().cpu().numpy(), unk)
        B = torch.FloatTensor(B)
        return B


    def forward(self, img_s, label_s, img_t, label_t):
        alpha = 1

        out_t_free = self.E_freezed(self.G_freezed(img_t)).detach()
        w_unk_posterior = self.compute_probabilities_batch(out_t_free, 1)
        w_k_posterior = 1 - w_unk_posterior
        w_k_posterior = w_k_posterior.to(self.args.device)
        w_unk_posterior = w_unk_posterior.to(self.args.device)

        #########################################################################################################
        for d_step in range(self.args.update_freq_D):
            self.opt_dc.zero_grad()
            feat_s = self.G(img_s).detach()
            out_ds = self.DC(feat_s)

            label_ds = Variable(torch.zeros(img_s.size()[0], dtype=torch.long).to(self.args.device))
            label_ds = nn.functional.one_hot(label_ds, num_classes=3)
            loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))  # self.criterion(out_ds, label_ds)

            label_dt_known = Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
            label_dt_known = nn.functional.one_hot(label_dt_known, num_classes=3)
            label_dt_unknown = 2 * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
            label_dt_unknown = nn.functional.one_hot(label_dt_unknown, num_classes=3)
            feat_t = self.G(img_t).detach()
            out_dt = self.DC(feat_t)
            label_dt = w_k_posterior[:, None] * label_dt_known + w_unk_posterior[:, None] * label_dt_unknown
            loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
            
            loss_D = 0.5*(loss_ds + loss_dt)
            if self.args.opt_clip >0.0:
                torch.nn.utils.clip_grad_norm_(self.DC.parameters(), self.args.opt_clip)
        #########################################################################################################
        for _ in range(self.args.update_freq_G):
            self.opt_g.zero_grad()
            self.opt_c.zero_grad()
            self.opt_e.zero_grad()
            feat_s = self.G(img_s)
            out_ds = self.DC(feat_s)
            loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))
            feat_t = self.G(img_t)
            out_dt = self.DC(feat_t)
            label_dt = w_k_posterior[:, None] * label_dt_known - w_unk_posterior[:, None] * label_dt_unknown
            loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))
            loss_G = alpha * (- loss_ds - loss_dt)
            #########################################################################################################
            out_Es = self.E(feat_s)
            label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
            label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
            label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
            loss_cls_Es = CrossEntropyLoss(label=label_s_onehot,
                                            predict_prob=F.softmax(out_Es, dim=1))
            out_Cs = self.C(feat_s)
            label_Cs_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
            label_Cs_onehot = label_Cs_onehot * (1 - self.args.ls_eps)
            label_Cs_onehot = label_Cs_onehot + self.args.ls_eps / (self.all_num_class)
            loss_cls_Cs = CrossEntropyLoss(label=label_Cs_onehot, predict_prob=F.softmax(out_Cs, dim=1))

            label_unknown = (self.known_num_class) * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
            label_unknown = nn.functional.one_hot(label_unknown, num_classes=self.all_num_class)
            label_unknown_lsr = label_unknown * (1 - self.args.ls_eps)
            label_unknown_lsr = label_unknown_lsr + self.args.ls_eps / (self.all_num_class)

            feat_t_aug = self.G(img_t)
            out_Ct = self.C(feat_t)
            out_Ct_aug = self.C(feat_t_aug)

            loss_cls_Ctu = alpha*CrossEntropyLoss(label=label_unknown_lsr, predict_prob=F.softmax(out_Ct_aug, dim=1),
                                            instance_level_weight=w_unk_posterior)

            pseudo_label = torch.softmax(out_Ct.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            targets_u_onehot = nn.functional.one_hot(targets_u, num_classes=self.all_num_class)
            mask = max_probs.ge(self.args.threshold).float()
            loss_ent_Ctk = CrossEntropyLoss(label=targets_u_onehot,
                                            predict_prob=F.softmax(out_Ct_aug, dim=1),
                                            instance_level_weight=mask)
            loss = loss_cls_Es + loss_cls_Cs + 0.5*loss_G + 0.5 * loss_ent_Ctk + 0.2 * loss_cls_Ctu

if __name__ == '__main__':
    args = parse_args()
    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    data_loader: dict = getDataLoader(args, source_info, target_info)

    model = Model(args, source_info, target_info, len(args.source_known_classes) + 1)
    model.freeze_GE()
    model.build_model()
    source_x, source_y = next(iter(data_loader['source']['train']))
    target_x, target_y = next(iter(data_loader['target']['train']))
    from thop import profile
    flops, params = profile(model, inputs=[source_x.to(args.device),source_y.to(args.device),target_x.to(args.device),target_y.to(args.device)])
    print(f'flops: {flops}\nparams: {params}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'total_params: {total_params}')