import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Conv1d, MaxPool1d, Linear
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.nn import GCN, GAT

class PSTL(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers, in_dim, hidden_dim,
                 num_layer, out_dim, mdp_flag, dropout):
        super(PSTL, self).__init__()
        self.mdp_flag = mdp_flag
        self.dgcnn = DGCNN(train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6)
        self.sfgcn = STGCN(in_dim, hidden_dim, num_layer, out_dim, dropout)

        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin3 = Linear(out_channels, 1)
        self.att_1 = Linear(out_channels, 16)
        self.att_2 = Linear(out_channels, 16)
        self.query = Linear(16, 1)

    def forward(self, data, adj, nfm_view, dp_view):
        emb_1 = self.dgcnn(data)
        emb_2 = self.sfgcn(adj, nfm_view, dp_view)

        target_nodes = data.target_nodes
        N = len(target_nodes)
        index_1 = torch.range(0, N - 1, 2).to(torch.long)
        index_2 = torch.range(1, N, 2).to(torch.long)
        src_nodes = target_nodes[index_1]
        dst_nodes = target_nodes[index_2]
        target_edges = torch.stack([src_nodes, dst_nodes]).T.tolist()
        idx = []
        for u, v in target_edges:
            idx.append(self.mdp_flag.__getitem__((u, v)))
        idx = torch.tensor(idx)
        emb_2 = emb_2[idx.long(), :]

        att_1 = self.query(F.tanh(self.att_1(emb_1)))
        att_2 = self.query(F.tanh(self.att_2(emb_2)))
        alpha_t = torch.exp(att_1) / (torch.exp(att_1) + torch.exp(att_2))
        alpha_f = torch.exp(att_2) / (torch.exp(att_1) + torch.exp(att_2))
        x = alpha_t * emb_1 + alpha_f * emb_2

        output_1 = self.lin1(emb_1)
        output_2 = self.lin2(emb_2)
        output_3 = self.lin3(x)
        score = torch.sigmoid(output_3)

        return output_1, output_2, output_3, score


class DGCNN(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(in_channels, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        feature_size = train_dataset.data.u_feature.shape[1] + train_dataset.data.v_feature.shape[1]
        self.lin1 = Linear(dense_dim + feature_size, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)

        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, data.u_feature, data.v_feature], 1)
        x = self.lin1(x)
        return x


# cite from DTI-MGNN
class STGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layer, out_dim, dropout):
        super(STGCN, self).__init__()
        self.SGAT1 = GAT(in_channels=in_dim, hidden_channels=hidden_dim, num_layers=num_layer, out_channels=out_dim,
                         dropout=dropout)
        self.SGAT2 = GAT(in_channels=in_dim, hidden_channels=hidden_dim, num_layers=num_layer, out_channels=out_dim,
                         dropout=dropout)
        self.CGCN = GCN(in_channels=in_dim, hidden_channels=hidden_dim, num_layers=num_layer, out_channels=out_dim,
                        dropout=dropout)
        self.a = nn.Parameter(torch.zeros(size=(out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(out_dim)
        self.tanh = nn.Tanh()

    def forward(self, adj, aug_feat1, aug_feat2):
        emb1 = self.SGAT1(aug_feat1, adj)
        com1 = self.CGCN(aug_feat1, adj)
        com2 = self.CGCN(aug_feat2, adj)
        emb2 = self.SGAT2(aug_feat2, adj)
        Xcom = (com1 + com2) / 2

        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        return emb


class Attention(torch.nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
