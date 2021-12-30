# utils for vgae.
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import sys
from retrying import retry
from sklearn.metrics import roc_auc_score, average_precision_score

def load_data(dataset, task='link_prediction', feat_norm=True):
    if task == 'link_prediction':
        names = ['graph', 'feature']
        objects = []
        for i in range(len(names)):
            with open("data/{}.{}".format(dataset, names[i]), 'rb') as f:
                objects.append(pkl.load(f))
        adj = objects[0]
        if feat_norm == True:
            feature = preprocess_feature(objects[1])
        else:
            # feature = torch.FloatTensor(np.array(objects[1].todense()))
            feature = sparse_to_torch_sparse_tensor(objects[1])
        return adj, feature
    elif task == 'classification':
        names = ['graph', 'feature','labels']
        objects = []
        for i in range(len(names)):
            with open("data/{}.{}".format(dataset, names[i]), 'rb') as f:
                objects.append(pkl.load(f))
        adj = objects[0]
        if feat_norm == True:
            feature = preprocess_feature(objects[1])
        else:
            # feature = torch.FloatTensor(np.array(objects[1].todense()))
            feature = sparse_to_torch_sparse_tensor(objects[1])
        labels = labels_encode(objects[2])
        return adj, feature, labels

def preprocess_feature(feature):
    rowsum = np.array(feature.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feature = r_mat_inv.dot(feature)
    feature = torch.FloatTensor(np.array(feature.todense()))
    return feature

def preprocess_graph_diff(adj, n_diff, alpha):
    adj = sp.coo_matrix(adj)
    adj_ = propagation_prob(adj, alpha)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_diff = np.power(adj_normalized, n_diff)
    return sparse_to_torch_sparse_tensor(adj_diff)

def preprocess_graph_diff_nth(adj, n_diff, alpha):
    adj_diff = sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    adj_ = propagation_prob(adj, alpha)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    for i in range(1, n_diff+1):
        adj_diff += np.power(adj_normalized, i)
    return sparse_to_torch_sparse_tensor(adj_diff)

def propagation_prob(adj, alpha):
    if  alpha == 0.5:
        adj = adj + sp.eye(adj.shape[0])
    else:
        adj = (alpha) * adj + (1-alpha) * sp.eye(adj.shape[0])
    return adj

def sparse_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def remove_diag(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    return adj

def mask_test_edges(adj, test, val):
    adj = remove_diag(adj)
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / test))
    num_val = int(np.floor(edges.shape[0] / val))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score_vgae(emb, adj_orig, edges_pos, edges_neg):
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def labels_encode(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    labels = torch.LongTensor(np.where(labels_onehot)[1])
    return labels