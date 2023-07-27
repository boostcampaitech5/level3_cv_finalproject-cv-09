import torch
import numpy as np
import random
import requests

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
def request_weight(path):
    with open(path, 'rb') as f:
        files = {'files': f}
        res = requests.post(
            f"http://127.0.0.1:30008/weight/",
            files=files,
        )
    
    return res.status_code