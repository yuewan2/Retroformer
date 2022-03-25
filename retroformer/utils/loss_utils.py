import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean', apply_logsoftmax=True, ignore_index=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.apply_logsoftmax = apply_logsoftmax
        self.ignore_idx = ignore_index

    def forward(self, logits, label):
        if logits.shape != label.shape and self.ignore_idx != -1:
            logits = logits[label != self.ignore_idx]
            label = label[label != self.ignore_idx]

        # Apply Label Smoothing:
        with torch.no_grad():
            if logits.shape != label.shape:
                new_label = torch.zeros(logits.shape)
                indices = torch.Tensor([[torch.arange(len(label))[i].item(),
                                         label[i].item()] for i in range(len(label))]).long()
                value = torch.ones(indices.shape[0])
                label = new_label.index_put_(tuple(indices.t()), value).to(label.device)
                label = label * (1 - self.smoothing) + self.smoothing / logits.shape[-1]
                label = label / label.sum(-1)[:, None]

            elif self.ignore_idx != -1:  # for context alignment loss
                label_lengths = (label != 2).sum(dim=-1)
                valid_indices = label_lengths != 0

                exist_align = (label == 1).sum(dim=-1) > 0
                smoothed_logits_addon = self.smoothing / label_lengths
                smoothed_logits_addon[smoothed_logits_addon > 1] = 0

                tmp = label.clone()
                tmp = tmp * (1 - self.smoothing) + smoothed_logits_addon.unsqueeze(1)
                tmp[label == 2] = 0

                label = tmp[valid_indices & exist_align]
                logits = logits[valid_indices & exist_align]

            else:
                label = label * (1 - self.smoothing) + self.smoothing / logits.shape[-1]
                label = label / label.sum(-1)[:, None]

        if self.apply_logsoftmax:
            logs = self.log_softmax(logits)
        else:
            logs = logits

        loss = -torch.sum(logs * label, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
