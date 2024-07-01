import os
import random
from typing import Dict

import torch
import numpy as np

def freeze_subnet(subnet):
    for p in subnet.parameters():
        p.requires_grad = False

def to_cpu(data: Dict[str, torch.Tensor] | torch.Tensor):
    if type(data) == dict:
        for k, v in data.items():
            v.to('cpu')
    else:
        data.to('cpu')

    return data

def set_deterministic(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True