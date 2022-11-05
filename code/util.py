import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import interp
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,accuracy_score,precision_score,recall_score,f1_score


def load_data(data_name, random_seed):
    if data_name == 'mda':
        path = '../data/mda/HMDD V2.0'

        M_FSM = np.loadtxt(path + '/M_FSM.txt')
        M_GSM = np.loadtxt(path + '/M_GSM.txt')
        m_features = M_FSM
        for i in range(M_FSM.shape[0]):
            for j in range(M_FSM.shape[1]):
                if m_features[i][j] == 0:
                    m_features[i][j] = M_GSM[i][j]
        m_features = torch.tensor(m_features, dtype=torch.float)

        D_SSM1 = np.loadtxt(path + '/D_SSM1.txt')
        D_SSM2 = np.loadtxt(path + '/D_SSM2.txt')
        D_GSM = np.loadtxt(path + '/D_GSM.txt')
        D_SSM = (D_SSM1 + D_SSM2) / 2
        d_features = D_SSM
        for i in range(D_SSM.shape[0]):
            for j in range(D_SSM.shape[1]):
                if d_features[i][j] == 0:
                    d_features[i][j] = D_GSM[i][j]
        d_features = torch.tensor(d_features, dtype=torch.float)

        edges_all = pd.read_csv(path + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
        edges_all['miRNA'] -= 1
        edges_all['disease'] -= 1
        pos_edges = edges_all.loc[edges_all['label'] == 1]
        neg_edges_all = edges_all.loc[edges_all['label'] == 0]
        neg_edges = neg_edges_all.sample(n=pos_edges.shape[0], random_state=random_seed, axis=0)
        pos_edges_df = pos_edges.iloc[:, :2]
        neg_edges_df = neg_edges.iloc[:, :2]
        pos_edges_df.reset_index(drop=True, inplace=True)
        neg_edges_df.reset_index(drop=True, inplace=True)

        return m_features, d_features, pos_edges_df, neg_edges_df


def construct_mdp_semantic_graph(m_features, d_features, all_mdps):
    all_mdps = all_mdps.T
    X_mdp = torch.cat([m_features[all_mdps[:, 0]], d_features[all_mdps[:, 1]]], 1)

    data_concat = Data(X_mdp)
    data_concat.pos = data_concat.x
    k = int(data_concat.num_edges / data_concat.num_nodes) + 1
    trans = KNNGraph(k, loop=False, force_undirected=True)
    mdp_semantic_graph = trans(data_concat)
    data_concat.pos, mdp_semantic_graph.pos = None, None

    return mdp_semantic_graph


def create_augmented_views(feature, mask_ratio, dropout_ratio):
    nfm_view = drop_out(feature, mask_ratio)
    dp_view = drop_node(feature, dropout_ratio)
    return nfm_view, dp_view


# cite from MERIT
def drop_out(input_feat, drop_percent):
    aug_input_feat = input_feat
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat


# cite from GRAND and GRPAMDA
def drop_node(feats, drop_rate, training=True):
    # not delete nodes for non-training
    n = feats.shape[0]
    # np.ones(): Returns a new array of the given shape and data type, with the element's value set to 1
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:
        # 从伯努利分布中抽取二元随机数(0 或者 1)。1. - drop_rates用于抽取上述二元随机值的概率
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


class PSTLMDA(InMemoryDataset):
    def __init__(self, feature_set, edge_set, args, fold, num_hops=1, split='train'):
        self.feature_set = feature_set
        self.edge_set = edge_set
        self.args = args
        self.num_hops = num_hops
        self.fold = fold

        super(PSTLMDA, self).__init__('../data/MDA')
        index = ['train', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def processed_file_names(self):
        return ['PSTLMDA_train_data_fold_{}.pt'.format(self.fold),
                'PSTLMDA_test_data_fold_{}.pt'.format(self.fold)]

    def process(self):
        train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index = self.edge_set
        train_pos_edge_index = torch.tensor(train_pos_edge_index.values).t().contiguous()
        train_neg_edge_index = torch.tensor(train_neg_edge_index.values).t().contiguous()
        test_pos_edge_index = torch.tensor(test_pos_edge_index.values).t().contiguous()
        test_neg_edge_index = torch.tensor(test_neg_edge_index.values).t().contiguous()
        edge_index = train_pos_edge_index

        Arow, Acol = toIndex(self.feature_set, edge_index)

        train_pos_list = extract_enclosing_subgraphs(self.feature_set, self.num_hops, train_pos_edge_index, Arow, Acol, 1)
        train_neg_list = extract_enclosing_subgraphs(self.feature_set, self.num_hops, train_neg_edge_index, Arow, Acol, 0)
        test_pos_list = extract_enclosing_subgraphs(self.feature_set, self.num_hops, test_pos_edge_index, Arow, Acol, 1)
        test_neg_list = extract_enclosing_subgraphs(self.feature_set, self.num_hops, test_neg_edge_index, Arow, Acol, 0)

        torch.save(self.collate(train_pos_list + train_neg_list), self.processed_paths[0])
        torch.save(self.collate(test_pos_list + test_neg_list), self.processed_paths[1])


# follow SEAL's application in IGMC
def extract_enclosing_subgraphs(feature_set, num_hops, link_index, Arow, Acol, y):
    m_features, d_features = feature_set

    data_list = []
    for u, v in tqdm(link_index.t().tolist(), desc='Extracting enclosing subgraphs for mdps...'):
        target_nodes = torch.tensor([u, v])
        u_nodes, v_nodes = [u], [v]
        u_dist, v_dist = [0], [0]
        u_visited, v_visited = set([u]), set([v])
        u_fringe, v_fringe = set([u]), set([v])
        for dist in range(1, num_hops + 1):
            v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
            u_fringe = u_fringe - u_visited
            v_fringe = v_fringe - v_visited
            u_visited = u_visited.union(u_fringe)
            v_visited = v_visited.union(v_fringe)
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = u_nodes + list(u_fringe)
            v_nodes = v_nodes + list(v_fringe)
            u_dist = u_dist + [dist] * len(u_fringe)
            v_dist = v_dist + [dist] * len(v_fringe)
        subgraph = Arow[u_nodes][:, v_nodes]
        subgraph[0, 0] = 0

        u, v, r = sp.find(subgraph)
        v += len(u_nodes)
        node_labels = [x * 3 for x in u_dist] + [x + 1 for x in v_dist]
        max_node_label = 2 * num_hops + 1

        u, v = torch.LongTensor(u), torch.LongTensor(v)
        edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)

        x = F.one_hot(torch.tensor((node_labels)), max_node_label + 1).to(torch.float)
        u_features = m_features[u_nodes]
        v_features = d_features[v_nodes]
        node_features = [u_features[0], v_features[0]]
        u_feature, v_feature = node_features

        sub_data = Data(x, edge_index, y=y, target_nodes=target_nodes)
        sub_data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
        sub_data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
        data_list.append(sub_data)
    return data_list


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return sp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return sp.csc_matrix((data, indices, indptr), shape=shape)


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def toIndex(feature_set, edge_index):
    m_features, d_features = feature_set
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(row.size)
    adj_train = sp.csr_matrix((data, [row, col]),
                              shape=[m_features.shape[0], d_features.shape[0]])
    Arow = SparseRowIndexer(adj_train)
    Acol = SparseColIndexer(adj_train.tocsc())

    return Arow, Acol


def evaluate_auc_ap(y_pred, y_true):
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise ValueError('Both y_pred and y_true need to be torch.Tensor.')
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    pred_test = [0 if j < 0.5 else 1 for j in y_pred]
    acc_test = accuracy_score(y_true, pred_test)
    pre_test = precision_score(y_true, pred_test)
    recall_test = recall_score(y_true, pred_test)
    f1_test = f1_score(y_true, pred_test)

    result = {'AUC': auc, 'AP': ap, 'ACC': acc_test, 'PRE': pre_test, 'REC': recall_test, 'F1': f1_test}
    return result
