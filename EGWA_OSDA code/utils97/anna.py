import torch
from torch import nn

def forward_FDA(rois: torch.tensor, targets: torch.tensor, classifier: nn.Module, args):
    criterion_bce = nn.BCELoss()

    unk_pixel_list = []
    k_pixel_list = []
    # loss_mining_dict = {}
    bs, c, h, w = rois.size()
    p_xk_x = 1.0

    for idx, roi in enumerate(rois):
        roi_label = targets[idx]  # bs x 1 (0-24)
        roi_flatten = roi.view(c, -1).permute(1, 0)

        classifier.apply(fix_bn)
        # constant > 0 is grl
        pixel_logits = classifier(roi_flatten)
        pixel_scores = pixel_logits.softmax(-1)
        classifier.apply(enable_bn)

        sorted_value, sorted_index = pixel_scores[:, :-1].sort(-1, descending=True)
        k_mask = (sorted_index[:, :1] == roi_label).sum(-1).bool()
        unk_mask = ~((sorted_index[:, :args.topk] == roi_label).sum(-1).bool())

        if k_mask.any() and unk_mask.any() and k_mask.sum() > unk_mask.sum():
            unk_pixels = pixel_logits[unk_mask]
            k_pixels = pixel_logits[k_mask]
            unk_pixel_list.append(unk_pixels)
            k_pixel_list.append(k_pixels)

    loss_mining_unk = 0
    if len(unk_pixel_list) > 1:

        num_lk = len(torch.cat(k_pixel_list))
        num_pu = len(torch.cat(unk_pixel_list))

        p_xu_x = num_pu / (num_pu + num_lk)
        p_xk_x = num_lk / (num_pu + num_lk)

        mined_scores = torch.cat(unk_pixel_list).softmax(-1)[:, -1]
        loss_mining_unk = p_xu_x * criterion_bce(mined_scores, torch.tensor([args.mining_th] * len(mined_scores)).to(args.device))
        # loss_mining_dict.update(loss_mining_s=loss_mining_unk)

    return loss_mining_unk, p_xk_x

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def enable_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()