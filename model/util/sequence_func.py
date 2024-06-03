import time

import numpy as np
import torch

__all__ = [
    'generate_mask_with_seq_length', 'generate_subsequent_mask',
]

def generate_mask_with_seq_length(sequence_length: torch.Tensor, max_len: int=None):
    '''
        Arguments:
            sequence_length: [B]
        Return:
            mask: [B, max_len], if padding, will be False

    '''
    device = sequence_length.device
    if max_len is None:
        max_len = sequence_length.max().item()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long().to(device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    mask = seq_range_expand < seq_length_expand
    return mask.detach()


def generate_subsequent_mask(size: int):
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

