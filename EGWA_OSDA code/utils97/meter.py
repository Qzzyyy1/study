import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics import F1Score
import json

from .logger import saveJSONFile

class AverageMeter:
    def __init__(self):
        self.reset()

    def __str__(self):
        return '%f' % self.value()

    def reset(self):
        self.sum = 0
        self.count = 0

    def add(self, value, count=1):
        self.sum += value
        self.count += count

    def value(self):
        return self.sum / self.count

class ConfusionMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def add(self, output, target):
        pred = output.max(1)[1].cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        matrix = confusion_matrix(target, pred, labels=np.arange(self.num_classes))
        self.matrix += matrix

    def value(self):
        return self.getOA(), self.getAA()

    def __str__(self):
        return "OA: %.2f%%\nAA: %.2f%%\nClass: %s" % \
               (self.getOA()*100, self.getAA()*100, np.around(self.getClassesAccuracy()*100, 2))

    def getOA(self):
        return np.diag(self.matrix).sum() / self.matrix.sum()

    def getAA(self):
        AA_matrix = self.matrix / self.matrix.sum(1, keepdims=True)
        AA = np.diag(AA_matrix).sum() / self.num_classes
        return AA

    def getClassesAccuracy(self):
        return np.diag(self.matrix / self.matrix.sum(1, keepdims=True))
    
class F1ScoreMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.prediction = []
        self.target = []

    def add(self, prediction, target):
        self.prediction.extend(prediction.cpu().detach().tolist())
        self.target.extend(target.cpu().detach().tolist())

    def value(self, average):
        # ['micro', 'macro']
        f1 = F1Score(self.num_classes, average=average)
        return f1(torch.tensor(self.prediction), torch.tensor(self.target))

class FeatureAndGTGather:
    def __init__(self):
        self.reset()

    def reset(self):
        self.feature_list = []
        self.gt_list = []

    def add(self, feature, gt):
        self.feature_list.append(feature.cpu().detach().numpy())
        self.gt_list.append(gt.cpu().detach().numpy())

    def getFeatureAndGT(self):
        return np.concatenate(self.feature_list, 0), np.concatenate(self.gt_list)
    
def computeOpensetDomainResult(prediction: torch.tensor or np.array or list, label: torch.tensor or np.array or list, known_num_classes: int):
    from torchmetrics import Accuracy
    from torchmetrics.classification import MulticlassAccuracy

    label = torch.tensor(label)
    prediction = torch.tensor(prediction)
    known_mask = label < known_num_classes
    unknown_mask = label == known_num_classes

    device = label.device

    oa_meter = Accuracy().to(device)
    aa_meter = MulticlassAccuracy(known_num_classes + 1, average=None).to(device)
    known_meter = Accuracy().to(device)
    unknown_meter = Accuracy().to(device)

    oa = oa_meter(prediction, label)
    aa = aa_meter(prediction, label).mean() # os
    classes_acc = aa_meter(prediction, label)
    oa_known = known_meter(prediction[known_mask], label[known_mask])
    aa_known = classes_acc[:-1].mean() # os_star
    unknown = unknown_meter(prediction[unknown_mask], label[unknown_mask])
    hos = (2 * aa_known * unknown) / (aa_known + unknown)
    hoa = (2 * oa_known * unknown) / (oa_known + unknown)

    return {
        'oa': oa.item(),
        'aa': aa.item(),
        'classes_acc': classes_acc.tolist(),
        'oa_known': oa_known.item(),
        'aa_known': aa_known.item(),
        'unknown': unknown.item(),
        'hos': hos.item(),
        'hoa': hoa.item(),
    }

class PredictionTargetGather:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prediction_list = []
        self.target_list = []

    def update(self, prediction, target):
        assert prediction.shape == target.shape, 'Error: The prediction and target shapes are different.'

        self.prediction_list.append(prediction)
        self.target_list.append(target)

    def get(self):
        return torch.cat(self.prediction_list), torch.cat(self.target_list)

class OpensetDomainMetric:
    def __init__(self, known_num_classes, args):

        self.known_num_classes = known_num_classes
        self.save_path = f'logs/{args.log_name}/{args.log_name} {args.source_dataset}-{args.target_dataset} seed={args.seed}.json'

        self.reset()

    def reset(self):
        self.gather = PredictionTargetGather()
        self.save_dict = None

    def update(self, prediction, target):
        self.gather.update(prediction, target)

    def compute(self):
        self.save_dict = computeOpensetDomainResult(*self.gather.get(), self.known_num_classes)
        return self.save_dict
    
    def save(self, a=False):
        saveJSONFile(self.save_path, self.save_dict, a=a)

    def print(self):
        print(json.dumps(self.save_dict, indent=4))

if __name__ == '__main__':
    a = ConfusionMeter(10)
    print(a)
