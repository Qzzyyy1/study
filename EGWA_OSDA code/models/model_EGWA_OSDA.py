from __future__ import print_function
import argparse
import time
import os
import datetime
from utils97 import utils1
from utils97 import utils
from utils97.utils1 import OptimWithSheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.function import HLoss, BCELossForMultiClassification, SupConLoss
from models.function import BetaMixture1D
from models.function import CrossEntropyLoss
from models.basenet import *
import copy
from utils97.utils1 import inverseDecayScheduler, CosineScheduler, StepScheduler, ConstantScheduler
from tqdm import tqdm
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
import json
import data_augment
from train import parse_args

from utils97.logger import saveJSONFile

class EGWA_OSDA():
    def __init__(self, args, source_info, target_info, num_class, data_loader:dict):
        self.model = 'EGWA_OSDA'
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
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)
        self.bmm_update_cnt = 0

        self.src_train_loader = data_loader['source']['train']
        self.target_train_loader = data_loader['target']['train']
        self.target_test_loader = data_loader['target']['test']
        self.draw_loader = data_loader['target']['all']
        self.num_batches = min(len(self.src_train_loader), len(self.target_train_loader))

        self.cutoff = True


    def build_model_init(self):
        self.G, self.E, self.C ,self.H= utils1.get_model_init(self.args,self.info, known_num_class=self.known_num_class, all_num_class=self.all_num_class)
        self.G.to(self.args.device)
        self.E.to(self.args.device)
        self.C.to(self.args.device)
        self.H.to(self.args.device)

        scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                  max_iter=self.args.warmup_iter)

        params = list(self.G.parameters())

        self.opt_w_g = OptimWithSheduler(optim.SGD(params, lr=self.args.g_lr * self.args.e_lr, weight_decay=5e-4, momentum=0.9,
                               nesterov=True), scheduler)
        self.opt_w_e = OptimWithSheduler(optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_w_c = OptimWithSheduler(optim.SGD(self.C.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)


    def build_model(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)

        _, _, self.E, self.DC,self.H = utils1.get_model(self.args,self.info, known_num_class=self.known_num_class, all_num_class=self.all_num_class, domain_dim=2)

        self.DC.apply(weights_init_bias_zero)

        self.E.to(self.args.device)
        self.DC.to(self.args.device)
        self.H.to(self.args.device)

        SCHEDULER = {'cos': CosineScheduler, 'step': StepScheduler, 'id': inverseDecayScheduler, 'constant':ConstantScheduler}
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter)
        scheduler_dc = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter*self.args.update_freq_D)

        params = list(self.G.parameters())

        self.opt_g = OptimWithSheduler(
            optim.SGD(params, lr=self.args.g_lr * self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_c = OptimWithSheduler(
            optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_dc = OptimWithSheduler(
            optim.SGD(self.DC.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_dc)

        scheduler_e = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                     max_iter=self.num_batches*self.args.training_iter)
        self.opt_e = OptimWithSheduler(
            optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler_e)

    def network_initialization(self):
        if 'resnet' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()
        if 'DCRN_02' in self.args.net:
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

    def train_init(self):
        print('train_init starts')
        t1 = time.time()
        epoch_cnt =0
        step=0
        while step < self.args.warmup_iter + 1:
            self.G.train()
            self.E.train()
            self.C.train()
            epoch_cnt +=1
            for batch_idx, (img_s, label_s) in enumerate(self.src_train_loader):

                img_s = Variable(img_s.to(self.args.device))
                label_s = Variable(label_s.to(self.args.device))
                step += 1
                if step >= self.args.warmup_iter + 1:
                    break

                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()
                feat_s = self.G(img_s)
                out_s = self.E(feat_s)

                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                loss_s = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s, dim=1))

                out_Cs = self.C(feat_s)
                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.all_num_class)
                loss_Cs = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                loss = loss_s + loss_Cs

                loss.backward()
                self.opt_w_g.step()
                self.opt_w_e.step()
                self.opt_w_c.step()
                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()

        duration = str(datetime.timedelta(seconds=time.time() - t1))[:7]
        print('train_init end with duration: %s'%duration)


    def train(self, input_dim=None):
        print('Train Starts')
        t1 = time.time()

        for epoch in range(1, self.args.training_iter):
            joint_loader = zip(self.src_train_loader, self.target_train_loader)
            alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch) / float(self.args.training_iter)))))) - 1)
            for batch_idx, ((img_s,label_s), (img_t, label_t)) in enumerate(joint_loader):

                self.G.train()
                self.C.train()
                self.DC.train()
                self.E.train()

                img_s = Variable(img_s.to(self.args.device))
                label_s = Variable(label_s.to(self.args.device))
                img_t = Variable(img_t.to(self.args.device))

                out_t_free = self.E_freezed(self.G_freezed(img_t)).detach()
                w_unk_H = self.compute_H_batch(out_t_free, 1)
                w_k_H = 1 - w_unk_H
                w_k_H = w_k_H.to(self.args.device)
                w_unk_H = w_unk_H.to(self.args.device)
                if self.cutoff:
                    w_unk_H[w_unk_H < self.args.threshold] = 0.0
                    w_k_H[w_k_H < self.args.threshold] = 0.0

                #########################################################################################################
                for d_step in range(self.args.update_freq_D):
                    self.opt_dc.zero_grad()

                    feat_s = self.G(img_s).detach()
                    out_ds = self.DC(feat_s)
                    label_ds = Variable(torch.zeros(img_s.size()[0], dtype=torch.long).to(self.args.device))
                    label_ds = nn.functional.one_hot(label_ds, num_classes=2)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))

                    label_dt = Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_dt = nn.functional.one_hot(label_dt, num_classes=2)
                    feat_t = self.G(img_t).detach()
                    out_dt = self.DC(feat_t)
                    label_dt = w_k_H[:, None] * label_dt
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt, dim=1))

                    loss_D = 0.5*(loss_ds + loss_dt)
                    loss_D.backward()

                    if self.args.opt_clip >0.0:
                        torch.nn.utils.clip_grad_norm_(self.DC.parameters(), self.args.opt_clip)
                    self.opt_dc.step()
                    self.opt_dc.zero_grad()


                #########################################################################################################
                for _ in range(self.args.update_freq_G):
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()

                    img_t1 = img_t.cpu()
                    img_t_w = torch.FloatTensor(data_augment.random_flip(
                        img_t1))
                    img_t_s = torch.FloatTensor(
                        data_augment.Crop_and_resize_batch(img_t1, 7 // 2))
                    img_t1 = torch.cat([img_t_w, img_t_s], dim=0)
                    img_t1 = Variable(img_t1.to(self.args.device))
                    feat_t1 = self.G(img_t1)
                    feat_t1 = self.H(feat_t1)
                    tdl_loss_fn = (SupConLoss(temperature=0.1)).to(self.args.device)
                    tdl_loss = tdl_loss_fn(feat_t1)

                    feat_s = self.G(img_s)
                    out_ds = self.DC(feat_s)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))

                    feat_t = self.G(img_t)
                    out_dt = self.DC(feat_t)
                    label_dt = w_k_H[:, None] * label_dt
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
                                                    instance_level_weight=w_unk_H)

                    pseudo_label = torch.softmax(out_Ct.detach(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    targets_u_onehot = nn.functional.one_hot(targets_u, num_classes=self.all_num_class)
                    mask = max_probs.ge(self.args.threshold).float()
                    loss_ent_Ctk = CrossEntropyLoss(label=targets_u_onehot,
                                                    predict_prob=F.softmax(out_Ct_aug, dim=1),
                                                    instance_level_weight=mask)

                    loss = loss_cls_Es + loss_cls_Cs + 0.60*loss_G + loss_ent_Ctk  + 0.18 * loss_cls_Ctu  + 0.10 *tdl_loss
                    loss.backward()
                    self.opt_g.step()
                    self.opt_c.step()
                    self.opt_e.step()
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()

            if (epoch%self.args.update_term==0):
                self.test(epoch)
        test_start = time.time()
        self.test(self.args.training_iter)
        test_end = time.time()
        print("test_time",test_end-test_start)

    def compute_H_batch(self, out_t, unk=1):
        ent_t = self.ent_criterion(out_t)
        batch_ent_t = (ent_t - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss + 1e-6)
        batch_ent_t[batch_ent_t >= 1 - 1e-4] = 1 - 1e-4
        batch_ent_t[batch_ent_t <= 1e-4] = 1e-4
        B = torch.FloatTensor(batch_ent_t.cpu().numpy())
        return B

    def freeze_GE(self):
        self.G_freezed = copy.deepcopy(self.G)
        self.E_freezed = copy.deepcopy(self.E)

    def test(self, epoch):
        from time import time
        test_start = time()
        self.G.eval()
        self.C.eval()
        self.E.eval()
        total_pred_t = np.array([])
        total_label_t = np.array([])
        all_ent_t = torch.Tensor([])

        oa_meter = Accuracy(task='MULTICLASS',num_classes=self.all_num_class).to(self.device)
        aa_meter = MulticlassAccuracy(self.all_num_class, average=None).to(self.device)
        known_meter = Accuracy(task='MULTICLASS',num_classes=self.all_num_class).to(self.device)
        unknown_meter = Accuracy(task='MULTICLASS',num_classes=self.all_num_class).to(self.device)

        with torch.no_grad():
            for batch_idx, (img_t, label_t) in enumerate(self.target_test_loader):

                img_t, label_t = Variable(img_t.to(self.args.device)), Variable(label_t.to(self.args.device))

                feat_t = self.G(img_t)
                out_t = F.softmax(self.C(feat_t), dim=1)

                pred = out_t.data.max(1)[1]
                pred_numpy = pred.cpu().numpy()
                total_pred_t = np.append(total_pred_t, pred_numpy)
                total_label_t = np.append(total_label_t, label_t.cpu().numpy())

                oa_meter.update(pred, label_t)
                aa_meter.update(pred, label_t)
                known_mask = label_t < self.known_num_class
                unknown_mask = label_t == self.known_num_class
                if known_mask.sum().item() > 0:
                    known_meter.update(pred[known_mask], label_t[known_mask])
                if unknown_mask.sum().item() > 0:
                    unknown_meter.update(pred[unknown_mask], label_t[unknown_mask])

                out_Et = self.E(feat_t)
                ent_Et = self.ent_criterion(out_Et)
                all_ent_t = torch.cat((all_ent_t, ent_Et.cpu()))

            max_target_label = int(np.max(total_label_t) + 1)
            m = utils1.extended_confusion_matrix(total_label_t, total_pred_t, true_labels=list(range(max_target_label)),
                                                pred_labels=list(range(self.all_num_class)))
            cm = m
            cm = cm.astype(np.float64) / np.sum(cm, axis=1, keepdims=True)
            acc_os_star = sum([cm[i][i] for i in range(self.known_num_class)]) / self.known_num_class
            acc_unknown = sum(
                [cm[i][self.known_num_class] for i in range(self.known_num_class, int(np.max(total_label_t) + 1))]) / (
                                      max_target_label - self.known_num_class)
            acc_os = (acc_os_star * (self.known_num_class) + acc_unknown) / (self.known_num_class + 1)
            acc_hos = (2 * acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)

            oa = oa_meter.compute().item()
            aa = aa_meter.compute().mean().item()
            classes_acc = aa_meter.compute().tolist()
            known_acc = known_meter.compute().item()
            unknown_acc = unknown_meter.compute().item()
            harmony_acc = (2 * known_acc * unknown_acc) / (known_acc + unknown_acc)
            save_dict = {
                f'epoch_{epoch} oa': oa,
                f'epoch_{epoch} aa': aa,
                f'epoch_{epoch} classes_acc': classes_acc,
                f'epoch_{epoch} harmony_acc': harmony_acc,
                f'epoch_{epoch} acc_os_star': acc_os_star,
                f'epoch_{epoch} acc_unknown': acc_unknown,
                f'epoch_{epoch} acc_os': acc_os,
                f'epoch_{epoch} acc_hos': acc_hos,
            }
            print(json.dumps(save_dict, indent=4))
            saveJSONFile(
                f'logs/{self.args.log_name}/{self.args.log_name} {self.args.source_dataset}-{self.args.target_dataset} seed={self.args.seed}.json',
                save_dict, a=True)
            test_end = time()
            saveJSONFile(f'time/{self.args.log_name}', {
                'test': test_end - test_start
            }, a=True)
            self.G.train()
            self.C.train()
            self.E.train()
            self.freeze_GE()

    def draw(self):
        self.G.eval()
        self.C.eval()
        self.E.eval()

        prediction_list = []

        with torch.no_grad():
            for img_t in self.draw_loader:

                img_t = Variable(img_t.to(self.args.device))

                feat_t = self.G(img_t)
                out_t = F.softmax(self.C(feat_t), dim=1)

                pred = out_t.data.max(1)[1]
                prediction_list.append(pred.cpu())

        from utils97.draw import drawPredictionMap

        drawPredictionMap(prediction_list, f'{self.args.log_name} {self.args.target_dataset}', self.info['target'], self.args.target_known_classes, self.args.target_unknown_classes, False)