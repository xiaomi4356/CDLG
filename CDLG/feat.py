import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model import DisenEncoder
from utils import *
import scipy
import argparse
import seaborn as sns
from matplotlib import patches

parser = argparse.ArgumentParser(description='CDG')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--datapath', type=str, default='../dataset')
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--enc_lr', type=float, default=0.001, help='the learning rate of total model')
parser.add_argument('--enc_weight_decay', type=float, default=0.0001, help='weight_decay of total model')
parser.add_argument('--k', type=int, default=6, help='channels')
parser.add_argument('--x_dim', type=int, default=8, help='dimension of each channels')
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


data, nclass = dataloader(args)
model = DisenEncoder(data.x.size(1), args).to(args.device)
model.load_state_dict(torch.load('best_cdg_cora.pth'))
model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)
z=z.cpu().detach().numpy()
print(z)
#compute person
cor = np.zeros((z.shape[1], z.shape[1]))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        cof = scipy.stats.pearsonr(z[:, i], z[:, j])[0]
        # cof = scipy.stats.kendalltau(z[:, i], z[:, j])[0]
        cor[i][j] = cof

print(cor)
def plot_corr(data, args):

    config = {
        "font.family": 'serif',  # sans-serif/serif/cursive/fantasy/monospace
        "font.size": 12,  # medium/large/small
        'font.style': 'normal',  # normal/italic/oblique
        'font.weight': 'normal',  # bold
        "mathtext.fontset": 'cm',  # 'cm' (Computer Modern)
        "font.serif": ['Times New Roman'],  # 'Simsun'宋体
        "axes.unicode_minus": False,  # 用来正常显示负号
    }
    plt.rcParams.update(config)

    ax = sns.heatmap(data, vmin=0.0, vmax=1.0, cmap="YlGnBu")

    rect1 = patches.Rectangle(xy=(0,0), width=args.x_dim, height=args.x_dim, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect1)
    rect2 = patches.Rectangle(xy=(args.x_dim, args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect2)
    rect3 = patches.Rectangle(xy=(2*args.x_dim, 2*args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect3)
    rect4 = patches.Rectangle(xy=(3*args.x_dim, 3*args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect4)
    rect5 = patches.Rectangle(xy=(4*args.x_dim, 4*args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect5)
    rect6 = patches.Rectangle(xy=(5*args.x_dim, 5*args.x_dim), width=args.x_dim, height=args.x_dim, linewidth=1.7, linestyle='--', edgecolor='r', facecolor='none')
    ax.add_patch(rect6)

    plt.savefig('club_feat_fig.eps', bbox_inches='tight', pad_inches=0.1, dpi=800)
    plt.savefig('club_feat_fig.png', bbox_inches='tight', pad_inches=0.1, dpi=800)
    plt.close()
plot_corr(np.abs(cor),args)
