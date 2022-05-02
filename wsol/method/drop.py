import torch
import torch.nn as nn

__all__ = ['AttentiveDrop']

class AttentiveDrop(nn.Module):
    def __init__(self, drop_threshold=0.7, drop_prob=1.):
        super(AttentiveDrop, self).__init__()
        if not (0 <= drop_threshold <= 1):
            raise ValueError("Drop threshold must be in range [0, 1].")
        self.drop_threshold = drop_threshold
        self.drop_prob = drop_prob
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, input_):
        m_avg = torch.mean(input_, dim=1, keepdim=True)
        thres = (self.maxpool(m_avg) * self.drop_threshold).expand_as(m_avg)

        dropped_mask = torch.where(m_avg > thres, torch.zeros_like(m_avg), torch.ones_like(m_avg))
        random_tensor = torch.rand_like(m_avg, dtype=torch.float32)
        dropped_mask = torch.where(random_tensor <= self.drop_prob,
                                   dropped_mask, torch.ones_like(dropped_mask))
        erased_input = input_ * dropped_mask

        return erased_input
