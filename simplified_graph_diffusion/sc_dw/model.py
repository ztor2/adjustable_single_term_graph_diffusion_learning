# models for baseline experiments
from sc_dw.utils import *
import numpy as np
import time
from sklearn.manifold import spectral_embedding
from node2vec import Node2Vec

def spectral_clustering_scores(train_test_split, random_state=0, dim=16):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split

    start_time = time.time()
    sc_scores = {}

    spectral_emb = spectral_embedding(adj_train, n_components=dim, random_state=random_state)
    sc_score_matrix = np.dot(spectral_emb, spectral_emb.T)

    runtime = time.time() - start_time
    sc_test_roc, sc_test_ap = get_roc_score(test_edges, test_edges_false, sc_score_matrix, apply_sigmoid=True)
    sc_val_roc, sc_val_ap = get_roc_score(val_edges, val_edges_false, sc_score_matrix, apply_sigmoid=True)

    # Record scores
    sc_scores['test_roc'] = sc_test_roc
    # sc_scores['test_roc_curve'] = sc_test_roc_curve
    sc_scores['test_ap'] = sc_test_ap

    sc_scores['val_roc'] = sc_val_roc
    # sc_scores['val_roc_curve'] = sc_val_roc_curve
    sc_scores['val_ap'] = sc_val_ap

    sc_scores['runtime'] = runtime
    return sc_scores

def deepwalk_scores(train_test_split, dim=16, walk_len=80, num_walk=10, window=10):
    start_time = time.time()
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
    
    G_train = nx.from_scipy_sparse_matrix(adj_train)
    model_train = Node2Vec(G_train, dimensions=dim, walk_length=walk_len, num_walks=num_walk)
    n2v_train = model_train.fit(window=window, min_count=1)
    edge_emb = n2v_train.wv

    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_emb = edge_emb[str(node_index)]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    n2v_score_matrix = np.dot(emb_matrix, emb_matrix.T)
    runtime = time.time() - start_time

#     if len(val_edges) > 0:
#         n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
#     else:
#         n2v_val_roc = None
#         n2v_val_roc_curve = None
#         n2v_val_ap = None
#         # Test set scores
#         n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)
    n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, n2v_score_matrix, apply_sigmoid=True)
    n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, n2v_score_matrix, apply_sigmoid=True)

    # Record scores
    n2v_scores = {}
    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap
    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap
    n2v_scores['runtime'] = runtime

    return n2v_scores