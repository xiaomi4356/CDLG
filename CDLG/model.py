import torch.nn as nn
import torch.nn.functional as F
import torch

class DisenEncoder(nn.Module):
    def __init__(self, in_dim, args):
        super(DisenEncoder, self).__init__()
        self.k = args.k
        self.routit = args.routit
        self.linear = nn.Linear(in_dim, args.k * args.x_dim)


    def forward(self, x, src_trg):
        x = self.linear(x)
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k

        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)  # neighbors' feature
        c = x  # node-neighbor attention aspect factor
        for t in range(self.routit):
            p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)  # update node-neighbor attention aspect factor
            p = F.softmax(p, dim=1)
            p = p.view(-1, 1).repeat(1, delta_d).view(m, k, delta_d)

            weight_sum = (p * z).view(m, d)  # weight sum (node attention * neighbors feature)
            c = c.index_add_(0, trg, weight_sum)  # update output embedding

            c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d)  # embedding normalize aspect factor
        return c

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        # for m in self.modules():
        #     self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        # ret = F.normalize(ret)
        # ret = F.dropout(ret, p=0.1)
        return ret


def pretext_loss(z1, z2, k, n, m):
    N = z1.size(0)
    z1 = z1.view(N, k, -1)
    z2 = z2.view(N, k, -1)
    pos_loss = torch.log(torch.sigmoid(torch.mul(z1, z2).sum(dim=2))).sum()

    #compute negative loss between nodes
    neg_intra_loss, neg_inter_loss = 0, 0
    for i in range(m):
        inter_index = torch.randperm(N)
        loss = torch.log(1-torch.sigmoid(torch.mul(z1, z2[inter_index]).sum(dim=2))).sum()
        neg_inter_loss = neg_inter_loss + loss
    #compute negative loss between channels
    for i in range(n):
        intra_index = torch.randperm(k)
        neg_intra_loss = torch.log(1-torch.sigmoid(torch.mul(z1, z2[:, intra_index, :]).sum(dim=2))).sum()
        neg_intra_loss = neg_intra_loss +loss

    loss = -1/(N*k)*(pos_loss + 1/m*neg_inter_loss +1/n*neg_intra_loss)
    return loss

def cos_loss(z1, z2, k, n, m):
    N = z1.size(0)
    z1 = z1.view(N, k, -1)
    z2 = z2.view(N, k, -1)
    z1_norm = torch.norm(z1, dim=2)
    z2_norm = torch.norm(z2, dim=2)
    a = torch.mul(z1, z2).sum(dim=2)
    b = torch.mul(z1_norm, z2_norm)
    pos_loss = torch.log(torch.sigmoid(torch.mul(a, 1/b))).sum()

    #compute negative loss between nodes
    neg_intra_loss, neg_inter_loss = 0, 0
    for i in range(m):
        inter_index = torch.randperm(N)
        z2_neg = z2[inter_index]
        z2_neg_norm = torch.norm(z2_neg, dim=2)
        m_neg = torch.mul(z1, z2_neg).sum(dim=2)
        n_neg = torch.mul(z1_norm, z2_neg_norm)
        loss = torch.log(1-torch.sigmoid(torch.mul(m_neg, 1/n_neg))).sum()
        neg_inter_loss = neg_inter_loss + loss
    #compute negative loss between channels
    for i in range(n):
        intra_index = torch.randperm(k)
        z2_neg_intra = z2[:, intra_index, :]
        z2_neg_intra_norm = torch.norm(z2_neg_intra, dim=2)
        m_neg_intra = torch.mul(z1, z2_neg_intra).sum(dim=2)
        n_neg_intra = torch.mul(z1_norm, z2_neg_intra_norm)
        neg_intra_loss = torch.log(1-torch.sigmoid(torch.mul(m_neg_intra, 1/n_neg_intra))).sum()
        neg_intra_loss = neg_intra_loss +loss

    loss = -1/(N*k)*(pos_loss + 1/m*neg_inter_loss +1/n*neg_intra_loss)
    return loss

def acc(ret, y, mask):
    preds = torch.argmax(ret, dim=1)
    correct = preds[mask] == y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc


