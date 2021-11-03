# utils in baseline experinments
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pickle as pkl
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp

def load_data(dataset, task='link_prediction', feat_norm=True, simple_diffusion_mode=True):
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
            feature = torch.FloatTensor(np.array(objects[1].todense()))
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
            feature = torch.FloatTensor(np.array(objects[1].todense()))
        labels = labels_encode(objects[2])
        return adj, feature, labels

# def preprocess_feature(feature):
#     rowsum = np.array(feature.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     feature = r_mat_inv.dot(feature)
#     feature = torch.FloatTensor(np.array(feature.todense()))
#     return feature

def remove_diag(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    return adj

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# def mask_test_edges(adj, test, val):
#     adj = remove_diag(adj)
#     assert np.diag(adj.todense()).sum() == 0

#     adj_triu = sp.triu(adj)
#     adj_tuple = sparse_to_tuple(adj_triu)
#     edges = adj_tuple[0]
#     edges_all = sparse_to_tuple(adj)[0]
#     num_test = int(np.floor(edges.shape[0] / test))
#     num_val = int(np.floor(edges.shape[0] / val))

#     all_edge_idx = list(range(edges.shape[0]))
#     np.random.shuffle(all_edge_idx)
#     val_edge_idx = all_edge_idx[:num_val]
#     test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
#     test_edges = edges[test_edge_idx]
#     val_edges = edges[val_edge_idx]
#     train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

#     def ismember(a, b, tol=5):
#         rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
#         return np.any(rows_close)

#     test_edges_false = []
#     while len(test_edges_false) < len(test_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], edges_all):
#             continue
#         if test_edges_false:
#             if ismember([idx_j, idx_i], np.array(test_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(test_edges_false)):
#                 continue
#         test_edges_false.append([idx_i, idx_j])

#     val_edges_false = []
#     while len(val_edges_false) < len(val_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], train_edges):
#             continue
#         if ismember([idx_j, idx_i], train_edges):
#             continue
#         if ismember([idx_i, idx_j], val_edges):
#             continue
#         if ismember([idx_j, idx_i], val_edges):
#             continue
#         if val_edges_false:
#             if ismember([idx_j, idx_i], np.array(val_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(val_edges_false)):
#                 continue
#         val_edges_false.append([idx_i, idx_j])
#     assert ~ismember(test_edges_false, edges_all)
#     assert ~ismember(val_edges_false, edges_all)
#     assert ~ismember(val_edges, train_edges)
#     assert ~ismember(test_edges, train_edges)
#     assert ~ismember(val_edges, test_edges)

#     data = np.ones(train_edges.shape[0])

#     adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
#     adj_train = adj_train + adj_train.T

#     return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
#     adj_rec = np.dot(emb, emb.T)
#     preds = []
#     pos = []
#     for e in edges_pos:
#         preds.append(sigmoid(adj_rec[e[0], e[1]]))
#         pos.append(adj_orig[e[0], e[1]])

#     preds_neg = []
#     neg = []
#     for e in edges_neg:
#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         neg.append(adj_orig[e[0], e[1]])

#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)

#     return roc_score, ap_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):

    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score

def edge_split(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True):
        
    adj = remove_diag(adj)
    assert np.diag(adj.todense()).sum() == 0
    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))

    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()
    
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        node1 = edge[0]
        node2 = edge[1]
        g.remove_edge(node1, node2)
        
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print ("Not enough removable edges to perform full train-test split!")
        print ("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print ("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
            
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
            
        val_edges_false.add(false_edge)

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue
        train_edges_false.add(false_edge)
        
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    adj_train = nx.adjacency_matrix(g)

    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false