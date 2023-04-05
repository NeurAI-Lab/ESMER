import torch
import torch.nn.functional as F
import numpy as np

 # Rampup Methods
def sigmoid_rampup(current, rampup_length, phase_shift=-5.0, min_val=0, max_val=1.):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return max_val
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return min_val + (max_val - min_val) * float(np.exp(phase_shift * phase * phase))


def linear_rampup(current, rampup_length, start_val=0):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return start_val + (1.0 - start_val) * current / rampup_length


def log_rampup(current, rampup_length, warmup_length=3, min_val=0.50, max_val=0.7):

    if current <= warmup_length:
        return min_val
    elif current >= rampup_length:
        return max_val
    else:
        return min_val + (max_val - min_val) * (np.log(current - warmup_length) / + np.log(rampup_length - warmup_length))


# Rampdown Methods
def cosine_rampdown(current, rampdown_length, warmup_length=3, min_val=0.1, max_val=0.5):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""

    if current < warmup_length:
        return min_val

    elif warmup_length <= current <= rampdown_length:
        return min_val + (max_val - min_val) * float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

    return max_val


def fixed_rampdown(current, rampdown_length, warmup_length=210, min_val=0.1, max_val=0.5):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""

    if current < warmup_length:
        return max_val

    elif warmup_length <= current <= rampdown_length:
        return min_val + (max_val - min_val) * float(.5 * (np.cos(np.pi * (current - warmup_length) / (rampdown_length - warmup_length)) + 1))

    else:
        return min_val



num_steps = 4
vals = [cosine_rampdown(x, num_steps - 1, warmup_length=0, min_val=0.1, max_val=0.95) for x in range(num_steps)]
