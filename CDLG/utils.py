import torch
import numpy as np
from loguru import logger
import os
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork
import logging
#读数据
def dataloader(args):
    transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(args.device)])
    dataset = Planetoid(root=args.datapath, name=args.dataset, transform=transform)
    data = dataset[0]
    return data, dataset.num_classes

def get_logger(name):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # 定义handler的输出格式（formatter）
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 创建一个handler，用于输出到控制台,决定将日志记录分配至正确的目的地
    ch = logging.StreamHandler() # 将日志写入控制台
    ch.setLevel(logging.DEBUG) # 指定将被分派到相应目标的最低严重性
    #给handler添加formatter
    ch.setFormatter(formatter)
    #给logger添加handler
    logger.addHandler(ch)

    # 创建一个handler，用于写入日志文件
    if name is not None:
        fh = logging.FileHandler(f'{name}.log')
        #给handler添加formatter
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        # 给logger添加handler
        logger.addHandler(fh)
    return logger


def set_rng_seed(seed):
    random.seed(seed) #为python设置随机种子
    np.random.seed(seed)  #为numpy设置随机种子
    torch.manual_seed(seed)   #为CPU设置随机种子
    torch.cuda.manual_seed(seed)   #为当前GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)   #为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

