import logging
from util import *
from config import parser
from pstlmda import PSTL

from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader


import warnings
warnings.filterwarnings('ignore')


def train(model, train_loader, adj, nfm_view, dp_view, device, optimizer, train_dataset):
    model.train()

    total_loss = 0
    for data in tqdm(train_loader, ncols=70):
        data = data.to(device)
        optimizer.zero_grad()
        logits_1, logits_2, logits_3, _ = model(data, adj, nfm_view, dp_view)
        loss_1 = BCEWithLogitsLoss()(logits_1.view(-1), data.y.to(torch.float))
        loss_2 = BCEWithLogitsLoss()(logits_2.view(-1), data.y.to(torch.float))
        loss_3 = BCEWithLogitsLoss()(logits_3.view(-1), data.y.to(torch.float))
        loss = loss_1 + loss_2 + loss_3
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(args, loader, adj, nfm_view, dp_view, model, device, test_dataset):
    model.eval()

    y_pred, y_true = [], []
    total_loss = 0
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        logits_1, logits_2, logits_3, score = model(data, adj, nfm_view, dp_view)
        loss_1 = BCEWithLogitsLoss()(logits_1.view(-1), data.y.to(torch.float))
        loss_2 = BCEWithLogitsLoss()(logits_2.view(-1), data.y.to(torch.float))
        loss_3 = BCEWithLogitsLoss()(logits_3.view(-1), data.y.to(torch.float))
        loss = loss_1 + loss_2 + loss_3
        total_loss += loss.item() * data.num_graphs
        y_pred.append(score.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return evaluate_auc_ap(y_pred, y_true)


def run(args, feature_set, mdp_semantic_graph, mdp_flag, edge_set, i, device):
    nfm_view, dp_view = create_augmented_views(mdp_semantic_graph.x, args.mask_ratio, args.dropout_ratio)

    train_dataset = eval('PSTLMDA')(feature_set, edge_set, args, fold=i, num_hops=1, split='train')
    test_dataset = eval('PSTLMDA')(feature_set, edge_set, args, fold=i, num_hops=1, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    model = PSTL(train_dataset, train_dataset[0].num_features, args.hidden_channel_t, args.out_channel, args.num_layers_t,
                   mdp_semantic_graph.x.shape[1], args.hidden_channel_s, args.num_layers_s, args.out_channel, mdp_flag, dropout=0.2).to(
        device)
    logger.info(model)

    adj = mdp_semantic_graph.edge_index.to(device)
    nfm_view = nfm_view.to(device)
    dp_view = dp_view.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    schedular = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_val_auc = 0
    patience = 0
    loss_train = []
    test_results = []
    for epoch in range(1, args.epochs):
        logger.info(f'Epoch: {epoch:02d}')
        schedular.step()
        loss = train(model, train_loader, adj, nfm_view, dp_view, device, optimizer, train_dataset)
        results = test(args, test_loader, adj, nfm_view, dp_view, model, device, test_dataset)
        val_auc, val_ap = results['AUC'], results['AP']
        val_acc, val_pre = results['ACC'], results['PRE']
        val_rec, val_f1 = results['REC'], results['F1']
        loss_train.append(loss)
        logger.info(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}, '
            f' Acc:{val_acc:.4f}, Pre:{val_pre:.4f}, REC:{val_rec:.4f}, F1:{val_f1:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_results = results
            patience = 0
        else:
            patience += 1

        if patience >= args.patience:
            logger.info('Early Stop!')
            break

    test_auc, test_ap = test_results['AUC'], test_results['AP']
    test_acc, test_pre = test_results['ACC'], test_results['PRE']
    test_rec, test_f1 = test_results['REC'], test_results['F1']

    logger.info(f'Fold: {i:02d}, TEST_AUC: {test_auc:.4f}, TEST_AP: {test_ap:.4f}, '
                f' Acc:{test_acc:.4f}, Pre:{test_pre:.4f}, REC:{test_rec:.4f}, F1:{test_f1:.4f}')

    return [test_auc, test_ap, test_acc, test_pre, test_rec, test_f1]


# follow the broad outline of BSAL
if __name__ == '__main__':
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        r'D:\PSTLMDA\results\PSTLMDA\Log_PSTLMDA.txt')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if args.cuda else 'cpu')

    logger.info('loading miRNA-disease dataset...')
    m_features, d_features, pos_edges_df, neg_edges_df = load_data(args.dataset, args.seed)
    all_mdps = torch.cat(
        [torch.tensor(pos_edges_df.values).t().contiguous(), torch.tensor(neg_edges_df.values).t().contiguous()], 1)

    mdp_flag = torch.sparse_coo_tensor(all_mdps, np.array(range(all_mdps.shape[1])),
                                       [m_features.shape[0], d_features.shape[0]])

    mdp_semantic_graph = construct_mdp_semantic_graph(m_features, d_features, all_mdps)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    i = 0
    res = []
    for train_idx, test_idx in kf.split(pos_edges_df):
        i += 1
        train_pos = pos_edges_df.iloc[train_idx]
        train_neg = neg_edges_df.iloc[train_idx]
        test_pos = pos_edges_df.iloc[test_idx]
        test_neg = neg_edges_df.iloc[test_idx]
        edge_set = [train_pos, train_neg, test_pos, test_neg]
        feature_set = [m_features, d_features]
        results = run(args, feature_set, mdp_semantic_graph, mdp_flag, edge_set, i, device)
        res.append(results)

    auc, ap, acc, pre, rec, f1 = 0, 0, 0, 0, 0, 0
    for j in range(len(res)):
        auc += res[j][0]
        ap += res[j][1]
        acc += res[j][2]
        pre += res[j][3]
        rec += res[j][4]
        f1 += res[j][5]

    for i in range(len(res)):
        logger.info(f'Fold: {i + 1:2d}, Test AUC: {res[i][0]:.4f}, Test AP: {res[i][1]:.4f}, '
                    f'Test ACC:{res[i][2]:.4f}, Test PRE:{res[i][3]:.4f}, Test REC:{res[i][4]:.4f},Test F1:{res[i][5]:.4f}')

    logger.info("AUC: avg {:.4f} std{:.4f}".format((auc / 5), np.std([i[0] for i in res])))
    logger.info("AP: avg {:.4f} std{:.4f}".format((ap / 5), np.std([i[1] for i in res])))
    logger.info("ACC: avg {:.4f} std{:.4f}".format((acc / 5), np.std([i[6] for i in res])))
    logger.info("PRE: avg {:.4f} std{:.4f}".format((pre / 5), np.std([i[7] for i in res])))
    logger.info("REC: avg {:.4f} std{:.4f}".format((rec / 5), np.std([i[8] for i in res])))
    logger.info("F1: avg {:.4f} std{:.4f}".format((f1 / 5), np.std([i[9] for i in res])))
