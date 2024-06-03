import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util import generate_mask_with_seq_length


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0, weight=None):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

        self.weight = weight

    def _smooth_targets(self, targets, vocab_size, mask):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # target: (B*T)
        # mask: (B, T)
        # smooth_dist: (B*T, C)
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(dim=1, index=targets.unsqueeze(1), value=1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        smooth_dist = smooth_dist * mask.float().unsqueeze(-1)
        return smooth_dist.detach()

    def forward(self, logits, targets, len_targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param logits: predicted by model, (B, T, C)
        :param targets: target indices, (B, T)
        :param len_targets: (B)
        :return:
        """
        B, T, C = logits.shape
        log_probs = F.log_softmax(logits, dim=-1).view(-1, C)
        mask = generate_mask_with_seq_length(len_targets)
        # log_probs: (B*T, C)
        if self.smoothing > 0:
            smooth_targets = self._smooth_targets(targets=targets.view(-1), vocab_size=C, mask=mask.view(-1))
            # targets: (B*T, C)
            loss = self.criterion(log_probs, smooth_targets)
        else:
            if self.weight is not None:
                log_probs = log_probs * self.weight
            loss = -torch.gather(log_probs, dim=1, index=targets.view(-1, 1))
            loss = (loss * mask.view(-1, 1)).sum()
            # targets: (B*T)
        loss = loss / mask.float().sum()

        preds = log_probs.max(1)[1]
        # (B*T)
        pred_seqs = preds.view(B, T).transpose(0, 1)
        # (T, B)
        num_corrects = int(preds.eq(targets.view(-1)).masked_select(mask.view(-1)).float().sum().item())
        num_words = len_targets.sum().item() 

        return loss, pred_seqs, num_corrects, num_words


class MaskedMSELoss(nn.Module):
    '''
        MSELoss with mask
    '''
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, source, target, len_target):
        # (B, T, C)
        distance = self.criterion(source, target)
        distance = distance.sum(-1)
        # (B, T)
        mask = generate_mask_with_seq_length(len_target).float()
        loss = (distance * mask).mean() / mask.sum()
        return loss
