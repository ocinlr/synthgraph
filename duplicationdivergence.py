"""
This module contains the implementation of the duplication-divergence model,
both considering only duplication and divergence, and also considering the
possibility of connecting the duplicated node to other nodes not connected to
it.

Algorithmic description
-----------------------

It is based on the following paper:

Pastor-Satorras R, Smith E, Solé RV. *Evolving protein interaction networks
through gene duplication*. J Theor Biol. 2003 May 21;222(2):199-210. doi:
10.1016/s0022-5193(03)00028-6.

The model starts from a complete graph with two nodes and then repeats this process:

1. A node is chosen at random and duplicated, creating a new node.

2. With probability delta, each of the edges of the duplicated node may be removed.

3. With probability alpha, the duplicated node may become connected to any of the nodes \
not connected to it before the duplication.

|

**API Reference**
-----------------
"""

import multiprocessing as mp
import os
import pickle as pk
import random
from statistics import mean
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

random_generator = None


def timestamp(msg="", prefix=''):
    """
    This function prints a timestamp with a message and a prefix (both optional)

    :param msg: Message to be shown after timestamp
    :type msg: str
    :param prefix: Message to be shown before timestamp
    :type prefix: str

    """
    print(time.strftime(f"{prefix}%Y-%m-%d %H:%M:%S"), msg, flush=True)


def _set_generator(seed):
    global random_generator
    """Procedure to fix the seed for random sample procedures

    :param seed: random seed
    :type seed: int
    
    """
    random_generator = default_rng(seed)


def _sample_binomial(n, p):
    """Sample size considering a binomial experiment

    :param n: Number of trials
    :type n: int
    :param p: Success probability
    :type n: float

    :return ans: The sampled value. Number of successes
    :rtype ans: int 
    """
    global random_generator
    ans = random_generator.binomial(n, p)
    return ans


def extended_duplication_divergence_graph(n, delta, alpha, seed=None,
                                          track_evolution=False, verbose=False):
    """Duplication-Divergence model where there is a possibility to connect a
    duplicated node to other edges not connected to it

    :param n: Number of nodes of the final network
    :type n: int
    :param delta: Probability of removing an edge from the duplicated node
    :type delta: float
    :param alpha: Probability of connecting the duplicated node with any of the
        other nodes not connected to it, when the duplication occurred
    :type alpha: float
    :param seed: (optional) Seed for the random number generator. Default: None
    :type seed: float
    :param track_evolution: (Optional) Whether topological parameters are stored
        and return at the end of the simulation. The returned metrics are:
        current appended node, number of edges, average square clustering,
        transitivity (triangle count) and degree histogram.
        BEWARE: This highly increases the computational complexity of the model.
        Use at your own discretion.
    :type track_evolution: bool
    :param verbose: (optional) Whether the progress should be displayed on the
        screen using tqdm. Default value: False
    :type verbose: bool

    :return g: The resulting graph, 0-indexed. If track_evolution is set to True,
        then the return value is a tuple (graph, list(metrics))
    :rtype g: networkx.Graph
    """
    _set_generator(seed)
    metrics = []
    g = nx.complete_graph(2)
    iterable = tqdm(range(2, n)) if verbose else range(2, n)
    for i in iterable:
        j = random.randrange(i)     # the node to be duplicated
        deg = len(g[j])             # degree of node j
        preserved = max(1, _sample_binomial(deg, 1 - delta))  # how many deletions
        new_edges = _sample_binomial(i - deg, alpha)          # how many new edges

        # first, compute the preserved edges
        list_preserved = random.sample(list(g[j]), preserved)
        # then, compute the new edges
        list_new = random.sample(list(set(g.nodes()) - set(g[j])), new_edges)

        # finally, add everything to the graph g
        g.add_node(i)
        for item in list_preserved:
            g.add_edge(item, i)
        for item in list_new:
            g.add_edge(item, i)
        if track_evolution:
            metrics.append((i + 1, *characterize_graph(g)))
    ans = (g, metrics) if track_evolution else g
    return ans


def duplication_divergence_graph(n, delta, seed=None,
                                 track_evolution=False, verbose=False):
    """Duplication-Divergence model where there is a possibility to connect a
    duplicated node to other edges not connected to it

    :param n: Number of nodes of the final network
    :type n: int
    :param delta: Probability of removing an edge from the duplicated node
    :type delta: float
    :param seed: (optional) Seed for the random number generator. Default: None
    :type seed: float
    :param track_evolution: (Optional) Whether topological parameters are stored
        and return at the end of the simulation. The returned metrics are:
        current appended node, number of edges, average square clustering,
        transitivity (triangle count) and degree histogram.
        BEWARE: This highly increases the computational complexity of the model.
        Use at your own discretion.
    :type track_evolution: bool
    :param verbose: (optional) Whether the progress should be displayed on the
        screen using tqdm. Default value: False
    :type verbose: bool

    :return g: The resulting graph, 0-indexed. If track_evolution is set to True,
        then the return value is a tuple (graph, list(metrics))
    :rtype g: networkx.Graph
    """
    _set_generator(seed)
    g = nx.complete_graph(2)
    metrics = []
    iterable = tqdm(range(2, n)) if verbose else range(2, n)
    for i in iterable:
        j = random.randrange(i)     # the node to be duplicated
        deg = len(g[j])             # degree of node j
        preserved = max(1, _sample_binomial(deg, 1 - delta))  # how many deletions

        # first, compute the preserved edges
        list_preserved = random.sample(list(g[j]), preserved)

        # finally, add everything to the graph g
        g.add_node(i)
        for item in list_preserved:
            g.add_edge(item, i)
        if track_evolution:
            metrics.append((i + 1, *characterize_graph(g)))
    ans = (g, metrics) if track_evolution else g
    return ans


def characterize_graph(g):
    """
    Characterize a graph by computing the number of edges, the average square
    clustering coefficient, the transitivity and the degree distribution. For
    the latter, the degree distribution is returned as a list of integers
    where the i-th element is the number of nodes with degree i.

    :param g: The graph to characterize
    :type g: networkx.Graph

    :return (edg, squ, tri, deg): A tuple containing the number of edges, the
        average square clustering coefficient, the transitivity and the degree
        distribution of the graph.
    :rtype: tuple
    """
    edg = g.number_of_edges()
    squ = mean(nx.square_clustering(g).values())
    tri = nx.transitivity(g)
    deg = nx.degree_histogram(g)
    return edg, squ, tri, deg


def characterize_complement(n, delta, alpha=0.0):
    """
    Characterize the complement of a duplication-divergence graph by computing
    the number of edges, the average square clustering coefficient, the
    transitivity and the degree distribution. For the latter, the degree
    distribution is returned as a list of integers where the i-th element is
    the number of nodes with degree i. If alpha is different from 0 or not
    given, the extended duplication-divergence graph is used instead.

    The complement of a graph is the graph with the same nodes and all the
    edges that are not in the original graph. This excludes self-loops and
    multiple edges.

    :param n: The number of nodes in the graph
    :type n: int
    :param delta: The probability of removing an edge after the duplication
        process
    :type delta: float
    :param alpha: The probability of adding an edge between the duplicated
        node and a random node not connected to the original node
    :type alpha: float

    :return (edg, squ, tri, deg): A tuple containing the number of edges, the
        average square clustering coefficient, the transitivity and the degree
        distribution of the graph.
    """

    if alpha == 0.0:
        g = nx.complement(duplication_divergence_graph(n, delta))
    else:
        g = nx.complement(extended_duplication_divergence_graph(n, delta, alpha))
    edg = g.number_of_edges()
    squ = mean(nx.square_clustering(g).values())
    tri = nx.transitivity(g)
    deg = nx.degree_histogram(g)
    return edg, squ, tri, deg


def _plot_distribution(g, dist_type='ccdf', display=True, label='plot'):
    """ Plot a distribution of a graph. It can be one of the following:
    - probability density distribution
    - clustering coefficient distribution

    Parameters
    g : an undirected graph
    dist_type : 'pdf','cdf','ccdf'
    """
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = ['\\usepackage[cmbright]{sfmath}']
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'cmbright'
    # plt.rcParams['font.size'] = 15
    # dictionary node:degree in_values = sorted(set(in_degrees.values()))
    degrees = dict(g.degree())
    values = sorted(set(degrees.values()))
    hist = [list(degrees.values()).count(x) for x in values]
    cum_hist = nx.utils.cumulative_distribution(hist)
    fig = plt.figure(figsize=(6.5, 6.5))
    plt.xlabel('$k$')
    if dist_type == 'pdf':
        items = list(zip(values, hist))
        plt.ylabel('$p(k)$')
        plt.title("Probability Density Function")
    elif dist_type == 'cdf':
        items = [*zip(values, cum_hist)]
        plt.ylabel('$P(K<k)$')
        plt.title("Cumulative Probability Function")
    elif dist_type == 'ccdf':
        items = list(zip(values, list(1 - np.array(cum_hist))))
        plt.ylabel('$P(K \\geq k)$')
        plt.title("Complementary Cumulative Probability Function")
    else:
        raise ValueError("distribution type should be one of ['pdf', 'cdf', 'ccdf']")
    ax = fig.add_subplot(111)
    ax.plot([k for (k, v) in items], [v for (k, v) in items])
    ax.set_aspect('auto')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(axis='both')
    print(f"Number of nodes: {g.number_of_nodes()}")
    print(f"Number of edges: {g.number_of_edges()}")
    print(f"Number of connected components: {nx.number_connected_components(g)}")
    if display:
        plt.show()
    else:
        plt.savefig('distribution_'+label+'_'+dist_type+'.pdf', dpi=600)
    plt.close()
    return items


def _my_boxplot(x, y, title='', x_label="", y_label="", x_log=False, y_log=True,
                edge_color='black', fill_color='lightblue', filename=None):
    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(y, labels=x, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.grid(which='both', axis='y', zorder=10)
    if filename is not None:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()
    plt.close()


def _my_violin_plot(x, y, title='', x_label="", y_label="", x_log=False, y_log=False, filename=None):
    plt.figure(figsize=(8, 5))
    plt.boxplot(y, labels=x, patch_artist=True, zorder=2.5)
    plt.violinplot(y, showmeans=False, showmedians=False, showextrema=False)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.grid(which='both', axis='y', zorder=10)
    if filename is not None:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()
    plt.close()


def _test1():
    """Testing function: This function creates a network using the
    extended_duplication_divergence_model and characterizes it.
    """
    n = 100
    delta = [0.25, 0.5, 0.75]
    alpha = 0.01
    for d in delta:
        g = extended_duplication_divergence_graph(n, delta=d, alpha=alpha)
        print(f"Delta: {d}")
        print(f"Number of nodes: {g.number_of_nodes()}")
        print(f"Number of edges: {g.number_of_edges()}")
        print(f"Number of connected components: {nx.number_connected_components(g)}")
        print(f"Average number of triangles: {mean(nx.triangles(g).values())}")
        print(f"Average square clustering: {mean(nx.square_clustering(g).values())}")
        print("-----------------------------------")
    timestamp("test1 ends...")


def _test2():
    """This function characterizes the complementary graph on the 
    model for different values of p
    """
    n = 50
    lo = 0.0
    hi = 1.0
    intervals = 20
    repetitions = 100
    num_threads = 7

    # p = 0 is irrelevant, isolated nodes
    x = list(np.linspace(lo, hi, intervals+1))[1:] 
    if not os.path.isfile("test_results/complement.pkl"):
        edges, square_coeff, tri_clust, deg_dist, prob = [], [], [], [], []
        pool = mp.Pool(num_threads)
        for p in x:
            timestamp(f"Starting interval @ {p:0.2f}...")
            ans = pool.starmap_async(characterize_complement, [(n, p)] * repetitions).get()
            timestamp("Analyzing data...")
            a, b, c, d = zip(*ans)
            edges.append(a)
            square_coeff.append(b)
            tri_clust.append(c)
            # relative degree distribution: averaging the degree distribution and append it to deg_dist
            mx_size = max([len(t) for t in d])
            deg = [0] * mx_size
            for arr in d:
                mx_sum = sum(arr)
                for i in range(len(arr)):
                    deg[i] += arr[i] / mx_sum
            for i in range(mx_size):
                deg[i] /= repetitions
            deg_dist.append(deg)
            prob.append(round(p, 2))
        pk.dump((edges, square_coeff, tri_clust, deg_dist, prob),
                open("test_results/complement.pkl", "wb"))
    else:
        edges, square_coeff, tri_clust, deg_dist, prob = pk.load(open("test_results/complement.pkl", "rb"))

    timestamp("Presenting plots...")
    _my_boxplot(prob, edges, title="Edges as a function of p", x_label='Probability',
                y_label='Number of Edges', filename='test_results/edges.pdf')
    _my_violin_plot(prob, square_coeff, title='Square Clustering Coefficient',
                    x_label='Probability', y_label='Average Square Clustering',
                    y_log=False, filename='test_results/square_coeff.pdf')
    _my_violin_plot(prob, tri_clust, title='Triangle Clustering Coefficient',
                    x_label='Probability', y_label='Average Triangle Clustering',
                    y_log=False, filename='test_results/tri_coeff.pdf')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Degree distribution")
    for i, p in enumerate(x):
        if i % 2 == 1:      # % 8 only 0.1, 0.5, 0.9 or % 4 also for 0.3 and 0.7
            n = len(deg_dist[i])
            tmp = np.array(nx.utils.cumulative_distribution(deg_dist[i]))
            ax1.plot(np.linspace(0, n, n + 1), tmp, "-o", label=f"p={p:0.1f}", alpha=0.7)
            ax2.plot(np.linspace(0, n, n + 1), np.ones(n+1) - tmp, "-o", label=f"p={p:0.1f}", alpha=0.7)
            
    ax1.grid(which='both', axis='y', zorder=10)
    ax2.grid(which='both', axis='y', zorder=10)
    ax1.set_title("Cumulative degree distribution")
    ax2.set_title("Complementary cumulative degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    # plt.xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.legend()
    ax2.legend()
    ax1.set_xlim(0, n)
    ax2.set_xlim(0, n)
    plt.tight_layout()
    # plt.show()
    plt.savefig("test_results/degree_distribution.pdf")
    timestamp("test2 ends...")


def _test3():
    """This function creates a network using the
    duplication_divergence_model and characterizes it.
    """
    n = 100
    delta = 0.5
    g = duplication_divergence_graph(n, delta=delta)
    print(f"Number of nodes: {g.number_of_nodes()}")
    print(f"Number of edges: {g.number_of_edges()}")
    print(f"Number of connected components: {nx.number_connected_components(g)}")
    print(f"Average number of triangles: {mean(nx.triangles(g).values())}")
    print(f"Average square clustering: {mean(nx.square_clustering(g).values())}")
    timestamp("test3 ends...")


def _test4():
    """ This function tries several values of alpha, for the same delta
    """
    n = 100
    delta = 0.5
    for alpha in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]:
        g = extended_duplication_divergence_graph(n, delta=delta, alpha=alpha)
        print(f"Alpha: {alpha}")
        print(f"Number of nodes: {g.number_of_nodes()}")
        print(f"Number of edges: {g.number_of_edges()}")
        print(f"Number of connected components: {nx.number_connected_components(g)}")
        print(f"Average number of triangles: {mean(nx.triangles(g).values())}")
        print(f"Average square clustering: {mean(nx.square_clustering(g).values())}")
        print("-----------------------------------")
    timestamp("test4 ends...")


if __name__ == '__main__':
    _test1()
    _test2()
    _test3()
    _test4()
