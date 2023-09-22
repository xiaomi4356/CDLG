import random
import sys
from model import DisenEncoder
from utils import *
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

seed=12345
set_rng_seed(seed)

parser = argparse.ArgumentParser(description='CDG')
parser.add_argument('--k', type=int, default=6, help='channels')
parser.add_argument('--x_dim', type=int, default=32, help='dimension of each channels')
parser.add_argument('--routit', type=int, default=4, help='iteration of disentangle')

args = parser.parse_args()
args.argv = sys.argv

args.device = torch.device('cuda:1')
################################################################################
transform = T.Compose([T.NormalizeFeatures()])
dataset_cora = Planetoid(root='../dataset', name='Cora', transform=transform)
data_cora = dataset_cora[0]
model = DisenEncoder(data_cora.x.size(1), args)
model.load_state_dict(torch.load('best_cdg_Cora_tsne.pth'))
model.eval()
with torch.no_grad():
    z_cora = model(data_cora.x, data_cora.edge_index)
    z_cora_d = TSNE(n_components=2).fit_transform(z_cora.detach().cpu().numpy())
x_cora = TSNE(n_components=2).fit_transform(data_cora.x.detach().cpu().numpy())

######################################################################################
transform = T.Compose([T.NormalizeFeatures()])
dataset_Citeseer = Planetoid(root='../dataset', name='Citeseer', transform=transform)
data_Citeseer = dataset_Citeseer[0]

model = DisenEncoder(data_Citeseer.x.size(1), args)
model.load_state_dict(torch.load('best_cdg_Citeseer_tsne.pth'))
model.eval()
with torch.no_grad():
    z_Citeseer = model(data_Citeseer.x, data_Citeseer.edge_index)
    z_Citeseer_d = TSNE(n_components=2).fit_transform(z_Citeseer.detach().cpu().numpy())

x_citeseer = TSNE(n_components=2).fit_transform(data_Citeseer.x.detach().cpu().numpy())
###########################################################################################
seed=69761
print(seed)
set_rng_seed(seed)
transform = T.Compose([T.NormalizeFeatures()])
dataset_Pubmed = Planetoid(root='../dataset', name='Pubmed', transform=transform)
data_Pubmed = dataset_Pubmed[0]

parser = argparse.ArgumentParser(description='CDG')
parser.add_argument('--k', type=int, default=5, help='channels')
parser.add_argument('--x_dim', type=int, default=32, help='dimension of each channels')
parser.add_argument('--routit', type=int, default=4, help='iteration of disentangle')

args = parser.parse_args()
args.argv = sys.argv
model = DisenEncoder(data_Pubmed.x.size(1), args)
model.load_state_dict(torch.load('best_cdg_Pubmed_tsne.pth'))
model.eval()
with torch.no_grad():
    z_Pubmed = model(data_Pubmed.x, data_Pubmed.edge_index)
    z_Pubmed_d = TSNE(n_components=2, perplexity=300).fit_transform(z_Pubmed.detach().cpu().numpy())

x_pubmed = TSNE(n_components=2).fit_transform(data_Pubmed.x.detach().cpu().numpy())
##############################################################################################

plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8,4.8))
# fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=0.1, hspace=0.2)  # 调整子图间距

axs[1, 0].scatter(z_cora_d[:, 0], z_cora_d[:, 1], s=2, c=data_cora.y, cmap="Set2")
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 0].set_title("Embeddings by CLDGE for Cora", y=-0.15)
axs[1, 0].axis('off')

axs[0, 0].scatter(x_cora[:, 0], x_cora[:, 1], s=2, c=data_cora.y, cmap="Set2")
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 0].set_title("Raw features for Cora", y=-0.15)
axs[0, 0].axis('off')

axs[1, 1].scatter(z_Citeseer_d[:, 0], z_Citeseer_d[:, 1], s=2, c=data_Citeseer.y, cmap="Set2")
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 1].set_title("Embeddings by CLDGE for Citeseer", y=-0.15)
axs[1, 1].axis('off')

axs[0, 1].scatter(x_citeseer[:, 0], x_citeseer[:, 1], s=2, c=data_Citeseer.y, cmap="Set2")
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[0, 1].set_title("Raw features for Citeseer", y=-0.15)
axs[0, 1].axis('off')

axs[1, 2].scatter(z_Pubmed_d[:, 0], z_Pubmed_d[:, 1], s=2, c=data_Pubmed.y, cmap="Set2")
axs[1, 2].set_xticks([])
axs[1, 2].set_yticks([])
axs[1, 2].set_title("Embeddings by CLDGE for Pubmed", y=-0.15)
axs[1, 2].axis('off')

axs[0, 2].scatter(x_pubmed[:, 0], x_pubmed[:, 1], s=2, c=data_Pubmed.y, cmap="Set2")
axs[0, 2].set_xticks([])
axs[0, 2].set_yticks([])
axs[0, 2].set_title("Raw features for Pubmed", y=-0.15)
axs[0, 2].axis('off')

plt.savefig("tsne.eps", dpi=600,bbox_inches='tight', pad_inches=0.1,  format='eps')
plt.savefig("tsne.png", bbox_inches='tight', pad_inches=0.1,  dpi=600)
plt.show()
