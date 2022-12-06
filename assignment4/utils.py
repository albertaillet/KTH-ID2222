# imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# typing
from numpy import ndarray
from typing import Optional


def load_data(filename: str) -> list[tuple[int, int]]:
    '''Load data from file and return it as a list of frozensets.'''
    with open(f'data/{filename}.dat', newline='') as csvfile:
        edges_reader = list(csv.reader(csvfile, delimiter=','))
        values = len(edges_reader[0])
        if values == 2:
            edges = [(int(a), int(b)) for a, b in edges_reader]
        else:
            edges = [(int(a), int(b)) for a, b, c in edges_reader]
    return edges


def get_adj_mtx(edges_list: list[tuple[int, int]]) -> ndarray:
    max_idx = max(max(*edges_list))
    A = np.zeros(shape=(max_idx, max_idx), dtype=int)
    for i, j in edges_list:
        A[i - 1, j - 1] = 1
        A[j - 1, i - 1] = 1
    return A


def get_laplacian(A: ndarray) -> ndarray:
    D_inv = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    # D_inv = np.linalg.inv(np.sqrt(D))
    L = D_inv @ A @ D_inv
    return L


def spectral_clustering(L: ndarray, k: Optional[int] = None) -> ndarray:

    # get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # compute the number of eigenvectors k
    if k is None:
        k = int(len(eigenvalues) - np.argmax(np.diff(eigenvalues)))
        print(f'Number of clusters using the heuristic: {k}')

    # exctract the k biggest eigenvectors in X
    X = eigenvectors[:, -k:]

    # renormalize X rows to get Y
    Y = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))

    # cluster the rows of Y using k-means clustering
    kmeans = KMeans(n_clusters=k).fit(Y)
    labels = kmeans.labels_
    return labels  # type: ignore


def draw_graph(A: ndarray, labels: ndarray, ax: plt.Axes) -> None:
    import networkx as nx

    graph = nx.from_numpy_array(A)
    pos = nx.spring_layout(graph)
    ax.set_title('Clusters')
    nx.draw_networkx(graph, pos=pos, node_size=10, node_color=labels, with_labels=False, cmap=plt.cm.Set3, ax=ax)  # type: ignore


def plot_fiedler(L: ndarray, ax) -> None:
    # get eigenvalues and eigenvectors
    _, eigenvectors = np.linalg.eigh(L)

    # extract the Fiedler vector (second smallest eigenvector)
    fiedler_eigenvector = eigenvectors[:, 1]

    ax.plot(np.arange(len(fiedler_eigenvector)), sorted(fiedler_eigenvector))
    ax.set_title('Fiedler vector')
    ax.grid()


def plot_eigengap(L: ndarray, ax: plt.Axes) -> None:
    # get eigenvalues and eigenvectors
    w, _ = np.linalg.eigh(L)

    # plot the eigengap
    ax.plot(np.arange(len(w) - 1), sorted(np.diff(w)), 'o')
    ax.set_title('Eigengap')
    ax.grid()


def plot_sparsity_pattern(L: ndarray, ax: plt.Axes) -> None:
    ax.set_title('Sparsity pattern of the Laplacian matrix')
    ax.spy(L)


def plots(L: ndarray, A: ndarray, labels: ndarray) -> None:
    _, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_sparsity_pattern(L, axs[0, 0])
    plot_eigengap(L, axs[0, 1])
    plot_fiedler(L, axs[1, 0])
    draw_graph(A, labels, axs[1, 1])
    plt.show()
