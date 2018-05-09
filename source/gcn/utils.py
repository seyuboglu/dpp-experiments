import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def shuffle_negatives(Y):
    n_negatives = np.sum(Y[:,0])
    Y[:,0] = np.zeros(Y.shape[0])
    new_indices = np.random.choice(range(Y.shape[0]), size=int(n_negatives), replace=False)
    while np.sum(Y[new_indices,1]) > 0.5: 
        new_indices = np.random.choice(range(Y.shape[0]), size=int(n_negatives), replace=False)
    Y[new_indices,0] = 1.0


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def inverse_sample_mask(idx, l):
    mask = np.ones(l)
    mask[idx] = 0
    return np.array(mask, dtype=np.bool)


def format_data(X, XS, Y, graph, idx_train, idx_validate, idx_test):
    X = X.astype(np.float32)
    XS = XS.astype(np.float32)
    Y = Y.astype(np.int32)

    features = sp.coo_matrix(X).tolil()
    side_features = XS #sp.coo_matrix(XS).tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    train_mask = sample_mask(idx_train, Y.shape[0])
    val_mask = sample_mask(idx_validate, Y.shape[0])
    test_mask = sample_mask(idx_test, Y.shape[0])

    y_train = np.zeros(Y.shape)
    y_val = np.zeros(Y.shape)
    y_test = np.zeros(Y.shape)
    y_train[train_mask, :] = Y[train_mask, :]
    y_val[val_mask, :] = Y[val_mask, :]
    y_test[test_mask, :] = Y[test_mask, :]

    save_to_excel("y_train.csv", y_train)
    save_to_excel("y_val.csv", y_val)
    save_to_excel("y_test.csv", y_test)
    save_to_excel("y_train.csv", y_train)
    save_to_excel("y_val.csv", y_val)
    save_to_excel("y_test.csv", y_test)

    return adj, features, side_features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def format_data_no_split(X, XS, Y, graph):
    X = X.astype(np.float32)
    XS = XS.astype(np.float32)
    Y = Y.astype(np.int32)

    features = sp.coo_matrix(X).tolil()
    side_features = XS #sp.coo_matrix(XS).tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    train_mask = sample_mask(idx_train, Y.shape[0])
    val_mask = sample_mask(idx_validate, Y.shape[0])
    test_mask = sample_mask(idx_test, Y.shape[0])

    y_train = np.zeros(Y.shape)
    y_val = np.zeros(Y.shape)
    y_test = np.zeros(Y.shape)
    y_train[train_mask, :] = Y[train_mask, :]
    y_val[val_mask, :] = Y[val_mask, :]
    y_test[test_mask, :] = Y[test_mask, :]

    save_to_excel("y_train.csv", y_train)
    save_to_excel("y_val.csv", y_val)
    save_to_excel("y_test.csv", y_test)
    save_to_excel("y_train.csv", y_train)
    save_to_excel("y_val.csv", y_val)
    save_to_excel("y_test.csv", y_test)

    return adj, features, side_features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_file(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    return load_data(x, y, tx, ty, allx, ally, graph, test_idx_reorder, dataset=dataset_str)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, side_features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['side_features']: side_features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def save_to_excel(filepath, array):
    return
    if isinstance(array, sp.spmatrix):
        array = array.todense()
    np.savetxt(filepath, array, delimiter=',', fmt='%s')
