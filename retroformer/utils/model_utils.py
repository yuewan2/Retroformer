import copy
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def reallocate_batch(batch, location='cpu'):
    batch = list(batch)
    for i in range(len(batch)):
        batch[i] = batch[i].to(location)
    return tuple(batch)


def validate(model, val_iter, pad_idx=1):
    pred_token_list, gt_token_list, pred_infer_list, gt_infer_list = [], [], [], []
    pred_arc_list, gt_arc_list = [], []
    pred_brc_list, gt_brc_list = [], []
    model.eval()
    for batch in tqdm(val_iter):
        src, tgt, gt_context_alignment, gt_nonreactive_mask, graph_packs = batch
        bond, _ = graph_packs

        # Infer:
        with torch.no_grad():
            scores, atom_rc_scores, bond_rc_scores, context_alignment = \
                model(src, tgt, bond)
            context_alignment = F.softmax(context_alignment[-1], dim=-1)

        # Atom-level reaction center accuracy:
        pred_arc = (atom_rc_scores.squeeze(2) > 0.5).bool()
        pred_arc_list += list(~pred_arc.view(-1).cpu().numpy())
        gt_arc_list += list(gt_nonreactive_mask.view(-1).cpu().numpy())

        # Bond-level reaction center accuracy:
        if bond_rc_scores is not None:
            pred_brc = (bond_rc_scores > 0.5).bool()
            pred_brc_list += list(pred_brc.view(-1).cpu().numpy())

        pair_indices = torch.where(bond.sum(-1) > 0)
        rc = ~gt_nonreactive_mask
        gt_bond_rc_label = (rc[[pair_indices[1], pair_indices[0]]] & rc[[pair_indices[2], pair_indices[0]]])
        gt_brc_list += list(gt_bond_rc_label.view(-1).cpu().numpy())

        # Token accuracy:
        pred_token_logit = scores.view(-1, scores.size(2))
        _, pred_token_label = pred_token_logit.topk(1, dim=-1)
        gt_token_label = tgt[1:].view(-1)
        pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
        gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

    pred_tokens = torch.cat(pred_token_list).view(-1)
    gt_tokens = torch.cat(gt_token_list).view(-1)

    if bond_rc_scores is not None:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               np.mean(np.array(pred_brc_list) == np.array(gt_brc_list)), \
               (pred_tokens == gt_tokens).float().mean().item()
    else:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               0, \
               (pred_tokens == gt_tokens).float().mean().item()
   