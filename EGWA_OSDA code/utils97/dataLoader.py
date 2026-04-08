import torch
from torch.utils.data import Subset, DataLoader

from .myDataset import FeatureDataset, AllDataset, LabelDataset
from .splitData import initDataset, getSourceTrainIndex, getTargetIndex


def getDataLoader(args, source_info, target_info, collate_fn=None, drop_last=False):
    source_dataset: dict = initDataset(args, source_info, known_classes=args.source_known_classes)
    target_dataset: dict = initDataset(args, target_info, known_classes=args.target_known_classes, unknown_classes=args.target_unknown_classes)

    source_index = getSourceTrainIndex(source_dataset['gt'], args, source_info)
    target_index: dict = getTargetIndex(target_dataset['gt'], args, target_info)

    source_all_dataset = AllDataset(source_dataset['data'], args, source_info)
    target_all_dataset = AllDataset(target_dataset['data'], args, target_info)
    source_label_dataset = LabelDataset(source_dataset['data'], source_dataset['gt'], args, source_info)
    target_label_dataset = LabelDataset(target_dataset['data'], target_dataset['gt'], args, target_info)

    source_train_dataset = Subset(source_label_dataset, source_index)
    target_foreground_dataset = Subset(target_label_dataset, target_index['all_index_list'])
    target_known_dataset = Subset(target_label_dataset, target_index['known_index_list'])
    target_unknown_dataset = Subset(target_label_dataset, target_index['unknown_index_list'])

    return {
        'source': {
            'all': DataLoader(source_all_dataset, batch_size=args.batch, shuffle=False),
            'train': DataLoader(source_train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=drop_last)
        },
        'target': {
            'all': DataLoader(target_all_dataset, batch_size=args.batch, shuffle=False),
            'train': DataLoader(target_foreground_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=drop_last),
            'test':  DataLoader(target_foreground_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn),
            'known': DataLoader(target_known_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn),
            'unknown': DataLoader(target_unknown_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
        }
    }