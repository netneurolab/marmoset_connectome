import numpy as np
import pandas as pd
import networkx as nx
import brainconn as bc
import matplotlib.pyplot as plt
import json

def save_pickle_file(path, data):
    import pickle
    with open(f'{path:}.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=-1)


def load_pickle_file(path):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_marmoset_data():
    [conn_mat, labels, dist_mat] = load_pickle_file('data/script_20181015_20181020_162509.pickle')
    # nodes_num = conn_mat.shape[0]
    g_conn = nx.from_numpy_matrix(conn_mat, create_using=nx.DiGraph())
    # conn_mat_bd = bc.utils.binarize(conn_mat, copy=True)
    for it1 in range(dist_mat.shape[0]):
        for it2 in range(dist_mat.shape[1]):
            if (it1, it2) in g_conn.edges:
                g_conn[it1][it2]['distance'] = dist_mat[it1, it2]

    return conn_mat, labels, dist_mat, g_conn
    

def prepare_coord_xyz():
    """
    get the xyz coordinates for plotting the marmoset brain wiring
    we leave this here because there may be better ways than mean value of injections
    :return:
    """
    [conn_mat, labels, dist_mat] = load_pickle_file('data/script_20181015_20181020_162509.pickle')
    df_coord = pd.read_csv('data/marmoset_coord.csv')
    label_coord = []
    for idx, label in enumerate(labels):
        label_coord.append(np.array(df_coord[df_coord.abbrev == label][['x', 'y', 'z']].mean()))
        # label_coord.append(df_coord[df_coord.abbrev == label][['x', 'y', 'z']].values[0])
    # print(label_coord)
    x = [_[0] for _ in label_coord]
    y = [_[1] for _ in label_coord]
    z = [_[2] for _ in label_coord]

    return x, y, z


def generate_new_labels():
    conn_mat, labels, dist_mat, g_conn = load_marmoset_data()
    with open("data/flat.json", "r") as f:
        flat = json.load(f)
    labels_flat = list(flat['areas'].keys())

    def strip_name(s):
        ret = s.replace('/', '-')
        if '(' in ret:
            ret = ret.split('(')[0]
        return ret

    labels_conv = {strip_name(_):_ for _ in labels_flat}

    # for label in labels:
    #     print(f"{label} -> {labels_conv[label]}")

    labels_new = [labels_conv[_] for _ in labels]
    
    save_pickle_file('data/new_labels', labels_new)
    
    
def plot_flatmap(values, **kwargs):
    assert len(values) == 55
    [img, rgba_list] = load_pickle_file('data/flatmap_res.pickle')
    
    # prepare background
    bg = np.zeros(img.shape)
    bg_sum = np.sum(img == (0, 0, 0, 255), axis=2)
    bg[np.nonzero(bg_sum==3)] = (255, 255, 255, 255)
    bg[np.nonzero(bg_sum==4)] = (0, 0, 0, 255)
    
    # prepare overlay
    mat = np.full((img.shape[0], img.shape[1]), np.nan)
    for it, color in enumerate(rgba_list):
        nz = np.nonzero(np.all(img[:, :] == color, axis=2))
        mat[nz] = values[it]
    
    # plot output
    fig, ax = plt.subplots()
    ax.imshow(bg.astype(np.uint8))
    ax.axis('off')
    im = plt.imshow(mat, **kwargs)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    # cb.set_ticks([])
    return fig, ax, cb

def corrcoef_matrices(X, Y):
    # correlation coefficient between matrices
    numerator = np.mean((X - X.mean()) * (Y - Y.mean()))
    denominator = X.std() * Y.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result
    
def annot_reg(X, y, reg_line_color='red', **kwargs):
    if np.ndim(X) == 1: X = X.reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(n_jobs=-1)
    res = reg.fit(X, y)
    yhat = reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    # adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    ax = plt.gca()
    ax.plot(X, yhat, color=reg_line_color, **kwargs)
    ax.annotate(f'Linear reg:\n'
                f'y = {res.coef_[0]:.2f} * X + {res.intercept_:.2f}, r^2 = {r_squared:.2f}\n',
                xy=(.1, .8), xycoords=ax.transAxes)

    import scipy.stats as stats
    r, p_val = stats.pearsonr(X.ravel(), y)
    ax.annotate("Pearson r = {:.2f} p={:.1e}".format(r, p_val),
                xy=(.1, .95), xycoords=ax.transAxes)