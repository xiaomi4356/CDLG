import random
import sys

import numpy as np
import torch
import torch.nn as nn
from model import DisenEncoder, pretext_loss, LogReg, acc, cos_loss
from utils import *
from torch_geometric.utils import dropout_adj
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CDG')
    parser.add_argument('--dataset', type=str, default='Pubmed')
    parser.add_argument('--datapath', type=str, default='../dataset')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--enc_lr', type=float, default=0.001, help='the learning rate of total model')
    parser.add_argument('--enc_weight_decay', type=float, default=0.0001, help='weight_decay of total model')
    parser.add_argument('--k', type=int, default=5, help='channels')
    parser.add_argument('--x_dim', type=int, default=32, help='dimension of each channels')
    parser.add_argument('--routit', type=int, default=4, help='iteration of disentangle')
    parser.add_argument('--gpu', type=int, default=-1)

    parser.add_argument('--de_rate1', type=float, default=0.3, help='dropout edges rate for view1')
    parser.add_argument('--de_rate2', type=float, default=0.3, help='dropout edges rate for view2')
    parser.add_argument('--df_rate1', type=float, default=0.3, help='dropout features rate for view1')
    parser.add_argument('--df_rate2', type=float, default=0.2, help='dropout features rate for view2')
    parser.add_argument('--m', type=int, default=10, help='inter negative samples')
    parser.add_argument('--n', type=int, default=4, help='intra negative samples')
    parser.add_argument('--log_epoch', type=int, default=200, help='epoch for logreg')
    parser.add_argument('--log_lr', type=float, default=0.01, help='the learning rate of LogReg for classification')
    parser.add_argument('--log_weight_decay', type=float, default=0.0001, help='weight_decay of LogReg classification')
    parser.add_argument('--log_trails', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1, help='fix random seed if needed')

    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:1' if args.gpu >= -1 else 'cpu')

    return args


def Encoder(args, logger):
    if args.seed > 0:
        #seed=random.randint(1, 1000)
        seed=356
        set_rng_seed(seed)
    logger.info(f'seed:{seed}')
    # load data
    data, nclass = dataloader(args)
    #generate two views
    edge_index_1 = dropout_adj(data.edge_index, p=args.de_rate1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=args.de_rate2)[0]
    x_1 = drop_feature(data.x, args.df_rate1)
    x_2 = drop_feature(data.x, args.df_rate2)
    #model and optim
    model = DisenEncoder(data.x.size(1), args).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.enc_lr, weight_decay=args.enc_weight_decay)

    #train
    loss_list = []
    best_loss, step = 1e6, 0
    for epoch in range(args.epoch):
        model.train()
        optim.zero_grad()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        # loss = pretext_loss(z1, z2, args.k, args.n, args.m)
        loss = cos_loss(z1, z2, args.k, args.n, args.m)
        loss_list.append(loss.item())

        if loss < best_loss:
            best_loss = loss
            step = epoch
            torch.save(model.state_dict(), 'best_cdg_Pubmed_tsne.pth')

        loss.backward()
        optim.step()
        logger.info(f'Epoch:{epoch:03d}, loss:{loss:.4f}')
    logger.info(f'step:{step:03d}, best_loss:{best_loss:.4f}')

    # obtain embeddings
    model.load_state_dict(torch.load('best_cdg_Pubmed_tsne.pth'))
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    return z, loss_list

def classification(z, args, logger):
    #load data
    data, nclass = dataloader(args)
    #logreg, optim and loss
    log = LogReg(args.k*args.x_dim, nclass).to(args.device)
    opt = torch.optim.Adam(log.parameters(), lr=args.log_lr, weight_decay=args.log_weight_decay)
    F_loss = nn.CrossEntropyLoss()
    #train for classification
    best_acc, best_log, log_step = 0, None, 0
    for i in range(args.log_epoch):
        log.train()
        opt.zero_grad()
        train_ret =log(z)
        loss = F_loss(train_ret[data.train_mask], data.y[data.train_mask])
        train_acc = acc(train_ret, data.y, data.train_mask)
        loss.backward()
        opt.step()

        with torch.no_grad():
            log.eval()
            val_ret = log(z)
            val_acc = acc(val_ret, data.y, data.val_mask)
            if val_acc > best_acc:
                best_acc = val_acc
                log_step = i
                torch.save(log.state_dict(), 'best_log.pth')
        logger.info(f'Log_epoch:{i:03d}, log_loss:{loss:.4f}, train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}')


    log.load_state_dict(torch.load('best_log.pth'))
    test_ret = log(z)
    test_acc = acc(test_ret, data.y, data.test_mask)
    logger.info(f'test_acc:{test_acc}, log_step:{log_step}')

    return test_acc

def main(args):
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{time.strftime("%Y-%m-%d,%H:%M")}'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(log_name)
    logger.info(f'args: {args}')
    logger.info(f'================================Encoder run=================================')
    z, loss_list = Encoder(args, logger)
    logger.info(f'================================Encoder run ends=================================')
    test_acc_list = []
    for i in range(args.log_trails):
        logger.info(f'================================Log runs {i+1}=================================')
        test_acc = classification(z, args, logger)
        test_acc_list.append(test_acc)
        logger.info(f'model runs {i+1} times, mean_acc={np.mean(test_acc_list)}+-{np.std(test_acc_list)}')
    logger.info(f'test_acc_lis:{test_acc_list}')
    logger.info(f'loss_lis:{loss_list}')
    logger.info(f'================================Log runs ends=================================')

if __name__ == "__main__":
    args = get_args()
    main(args)
