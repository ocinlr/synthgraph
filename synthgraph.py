"""
    This module contains the code for generating synthetic graphs using the
    duplication-divergence model. This process resembles the evolution of
    interactions between proteins in an organism. This evolution is
    thought to occur in three steps:

        1. Duplication: A protein duplicates itself. This is done by
        duplicating a vertex in the graph and copying all its edges.

        2. Divergence: The two proteins diverge from each other. This is
        done by randomly removing edges from the duplicated vertex with a
        probability delta.

        3. Evolution: The duplicated protein develops new interactions. This
        is done by adding new edges to the duplicated vertex with a
        probability alpha

Graph generation
----------------
    Two methods are provided for generating synthetic graphs in such a manner:

        1. duplication_divergence_graph: This method generates a single graph
        using the duplication-divergence model (steps 1 and 2)

        2. extended_duplication_divergence_graph: This method generates a single
        graph using the duplication-divergence model (steps 1, 2, and 3)

    The duplication-divergence model is parameterized by the following
    parameters:

        - n: The number of vertices in the graph

        - delta: The probability of an edge being removed during divergence

        - alpha: The probability of an edge being added during evolution

        - seed: The seed for the random number generator

    Additional optional parameters are provided for controlling verbosity:

        - track_evolution: If set to True, a list of topological parameters \
        are stored and return at the end of the simulation. The returned \
        metrics are: current appended node, number of edges, average square \
        clustering, transitivity (triangle count) and degree histogram.

        - verbose: If set to True, the method will display the progress of \
        the node aggregation process using the tqdm library.

Calibration methods
-------------------
    Besides, this module implements methods for calibrating the alpha and
    delta parameters given a target number of edges. This method uses a
    modified binary search algorithm to find the correct value for the
    parameter. The algorithm works as follows:

        1. Given the expected number of edges, the method creates partitions
        of the search space [0, 1] for a parameter (alpha or delta).

        2. For each partition, the method generates several graphs using the
        duplication-divergence model with the given parameters.

        3. The method computes the average number of edges for each partition.

        4. If the behaviour of the average number of edges is monotonic, the
        method narrows the search space to the partition including the target
        number of edges and repeats the procedure. Otherwise, the method stops
        and returns the average of the limits of the partition with the
        expected number of edges.

    Keep in mind that the calibration method finds an approximate value for
    the parameter, not the exact value. This is because the number of edges
    varies from one graph to another, even if the parameters are the same.
    For the extended duplication-divergence model, one of the parameters
    should be fixed, either alpha or delta.

    The calibration methods are parameterized by the following parameters,
    where applicable:

        - n: The number of vertices in the graph.
        - target_edges: The target number of edges.
        - lo_alpha: The lower bound for the alpha parameter.
        - hi_alpha: The upper bound for the alpha parameter.
        - lo_delta: The lower bound for the delta parameter.
        - hi_delta: The upper bound for the delta parameter.
        - intervals: The number of partitions for the search space.
        - repetitions: The number of graphs generated for each partition.
        - seed: The seed for the random number generator.
        - verbose: If set to True, the method will display the progress of \
        the node aggregation process using the tqdm library.

Using an existing network
-------------------------
    Support has been added for generating a graph using the
    duplication-divergence model from an existing graph or the
    intersection of two graphs. Those methods are:

        1. synth_graph_from_network: This method generates a graph using the \
        duplication-divergence model from an existing graph.

        2. synth_graph_from_intersection: This method generates a graph using \
        the duplication-divergence model from the intersection of two graphs.

    The methods are parameterized by the following parameters:

        - g1: The graph to be used as a template.
        - g2: The second graph to be used as a template (for the intersection method only).
        - alpha: The probability of an edge being added during evolution.
        - lo_delta: The lower bound for the delta parameter.
        - hi_delta: The upper bound for the delta parameter.
        - all other optional parameters like the calibration functions: \
        target_edges, lo_alpha, hi_alpha, intervals, repetitions, seed, verbose.
        - seed: The seed for the random number generator.
        - track_evolution: If set to True, a list of topological parameters \
        are stored and return at the end of the simulation. The returned \
        metrics are: current appended node, number of edges, average square \
        clustering, transitivity (triangle count) and degree histogram.

|

**API Reference**
-----------------
"""
from multiprocessing import Pool
from duplicationdivergence import *


class _DeltaGraph(object):
    """
    This class is used to create a graph with the given number of edges as
    static parameter and delta as dynamic parameter (using the __call__
    method).
    """
    __slots__ = ['n']

    def __init__(self, n):
        """
        :param n: The number of vertices in the graph.
        """
        self.n = n

    def __call__(self, delta):
        """
        This method generates a graph with the given number of edges and
        delta.

        :param delta: The probability of an edge being removed during divergence.
        :type delta: float

        :return graph: A graph generated using the duplication-divergence model.
        :rtype: networkx.Graph
        """
        return duplication_divergence_graph(self.n, delta).number_of_edges()


class _DeltaGivenAlphaGraph(object):
    """
    This class is used to generate graphs using the duplication-divergence
    model with a given value for the delta parameter and a given value for
    the alpha parameter.
    """
    __slots__ = ['n', 'alpha']

    def __init__(self, n, alpha):
        """
        Constructor for the _DeltaGivenAlphaGraph class.

        :param n: The number of vertices in the graph.
        :type n: int
        :param alpha: The alpha parameter for the duplication-divergence model.
        :type alpha: float
        """
        self.n = n
        self.alpha = alpha

    def __call__(self, delta):
        """
        This method generates a graph using the extended duplication-divergence
        model with the given values for the alpha and delta parameters.

        :param delta: The delta parameter for the duplication-divergence model.
        :type delta: float

        :return edges: The number of edges in the generated graph.
        :rtype edges: int
        """
        # Return the number of edges in the graph.
        return extended_duplication_divergence_graph(self.n, delta, self.alpha).number_of_edges()


class _AlphaGivenDeltaGraph(object):
    """
    This class represents a graph generated using the duplication-divergence
    model with a fixed value for the delta parameter. The class implements
    the __call__ method, which returns the number of edges in the graph.
    """
    __slots__ = ["n", "delta"]

    def __init__(self, n, delta):
        """
        This method initializes the object.

        :param n: The number of vertices in the graph.
        :type n: int
        :param delta: The delta parameter for the duplication-divergence model.
        :type delta: float
        """

        self.n = n
        self.delta = delta

    def __call__(self, alpha):
        """
        This method generates a graph using the duplication-divergence model
        with the given parameters and returns the number of edges in the
        graph.

        :param alpha: The alpha parameter for the duplication-divergence model.
        :type alpha: float

        :return edges: The number of edges in the generated graph.
        :rtype edges: int
        """

        # Generate the graph using the duplication-divergence model.
        graph = extended_duplication_divergence_graph(self.n, self.delta, alpha)

        # Return the number of edges in the graph.
        return graph.number_of_edges()


def calibrate_delta(n, target_edges, alpha=None, lo_delta=0.0, hi_delta=0.99, intervals=10,
                    repetitions=100, max_iterations=5, threads=-1, verbose=False):
    """
    This method calibrates the delta parameter for the duplication-divergence
    model given a target number of edges. The method uses a modified binary
    search algorithm to find the correct value for the parameter. The
    algorithm calibrates the value of delta using the following steps:

        1. Given the expected number of edges, the method creates partitions
        of the search space [lo_delta, hi_delta].

        2. For each partition, the method generates several graphs using the
        duplication-divergence model with the given parameters.

        3. The method computes the average number of edges for each partition.

        4. If the behaviour of the average number of edges is monotonic, the
        method narrows the search space to the partition including the target
        number of edges and repeats the procedure. Otherwise, the method stops
        and returns the average of the limits of the partition with the
        expected number of edges.

    :param n: The number of vertices in the graph.
    :type n: int
    :param target_edges: The target number of edges.
    :type target_edges: int
    :param alpha: The alpha parameter for the duplication-divergence model. If \
    set to None, the method will use the extended duplication-divergence model. \
    Otherwise, the method will use the duplication-divergence model.
    :type alpha: float
    :param lo_delta: The lower bound for the delta parameter.
    :type lo_delta: float
    :param hi_delta: The upper bound for the delta parameter.
    :type hi_delta: float
    :param intervals: The number of partitions for the search space.
    :type intervals: int
    :param repetitions: The number of graphs generated for each partition.
    :type repetitions: int
    :param max_iterations: The maximum number of iterations for the binary \
    search algorithm.
    :type max_iterations: int
    :param threads: The number of threads to use for parallel execution. If \
    set to -1, the method will use all available threads.
    :type threads: int
    :param verbose: If set to True, the method will display the progress of \
    the node aggregation process using the tqdm library.

    :return delta: The calibrated value for the delta parameter.
    :rtype delta: float
    """

    # Create the object for parallel execution using the _DeltaGivenAlphaGraph
    # or the _DeltaGraph class, according to the value of alpha.
    graphs = _DeltaGraph(n) if alpha is None else _DeltaGivenAlphaGraph(n, alpha)

    # Main loop
    if verbose:
        iterable = tqdm(range(max_iterations), desc="Calibrating delta", disable=not verbose)
    else:
        iterable = range(max_iterations)
    for _ in iterable:
        # Create partitions for the search space.
        partitions = np.linspace(lo_delta, hi_delta, intervals + 1)

        # create list of deltas (repetitions times for each partition)
        deltas = [part for part in partitions for _ in range(repetitions)]

        # Executing the __call__ method for each object in parallel using Pool class.
        # This computation returns a list of ints with the number of edges for
        # each delta in deltas. The list is reshaped to a matrix with intervals
        # rows and repetitions columns. The average number of edges for each
        # partition is computed and stored in avg_edges.
        with Pool(threads) as pool:
            edges = pool.map(graphs, deltas)
        edges = np.array(edges).reshape(intervals + 1, repetitions)
        avg_edges = np.mean(edges, axis=1)

        # Check if the behaviour of the average number of edges is monotonic.
        # If it is, narrow the search space to the partition including the
        # target number of edges. Otherwise, return the average of the limits
        # of the partition with the expected number of edges.
        if np.all(np.diff(avg_edges) > 0):
            # monotonic increasing
            hi_delta = partitions[avg_edges >= target_edges][0]
            lo_delta = partitions[avg_edges < target_edges][-1]
        elif np.all(np.diff(avg_edges) < 0):
            # monotonic decreasing
            hi_delta = partitions[avg_edges <= target_edges][0]
            lo_delta = partitions[avg_edges > target_edges][-1]
        else:
            # Counting the number of partitions with more edges than the previous
            # partition. If the number of partitions with more edges is greater
            # than the number of partitions with fewer edges than the previous
            # partition, the behaviour is monotonic increasing. Otherwise, it
            # is monotonic decreasing.
            increasing_count = np.sum(np.diff(avg_edges) > 0)
            decreasing_count = np.sum(np.diff(avg_edges) < 0)
            if increasing_count > decreasing_count:
                # Finding the index i such that partitions[i] is the last
                # partition with fewer edges than the target number of edges.
                # The target number of edges is located in the interval
                # [partitions[i], partitions[i+1]].
                i = np.where(avg_edges < target_edges)[0][-1]
                return np.mean(partitions[i:i + 2])
            else:
                # Finding the index i such that partitions[i] is the last
                # partition with more edges than the target number of edges.
                # The target number of edges is located in the interval
                # [partitions[i], partitions[i+1]].
                i = np.where(avg_edges > target_edges)[0][-1]
                return np.mean(partitions[i:i + 2])

    # If the maximum number of iterations is reached, return the average of
    # the limits of the search space.
    return np.mean([lo_delta, hi_delta])


def calibrate_alpha(n, target_edges, delta, lo_alpha=0.0, hi_alpha=0.99, intervals=10,
                    repetitions=100, max_iterations=5, threads=-1, verbose=False):
    """
    This method calibrates the alpha parameter for the extended
    duplication-divergence model given a target number of edges. The method
    uses a modified binary search algorithm to find the correct value for
    the parameter. The algorithm calibrates the value of alpha using the
    following steps:

        1. Given the expected number of edges, the method creates partitions
        of the search space [lo_alpha, hi_alpha].

        2. For each partition, the method generates several graphs using the
        extended duplication-divergence model with the given parameters.

        3. The method computes the average number of edges for each partition.

        4. If the behaviour of the average number of edges is monotonic, the
        method narrows the search space to the partition including the target
        number of edges and repeats the procedure. Otherwise, the method stops
        and returns the average of the limits of the partition with the
        expected number of edges.

    :param n: The number of vertices in the graph.
    :type n: int
    :param target_edges: The target number of edges.
    :type target_edges: int
    :param delta: The delta parameter for the extended duplication-divergence \
    model.
    :type delta: float
    :param lo_alpha: The lower bound for the alpha parameter.
    :type lo_alpha: float
    :param hi_alpha: The upper bound for the alpha parameter.
    :type hi_alpha: float
    :param intervals: The number of partitions for the search space.
    :type intervals: int
    :param repetitions: The number of graphs generated for each partition.
    :type repetitions: int
    :param max_iterations: The maximum number of iterations for the binary \
    search algorithm.
    :type max_iterations: int
    :param threads: The number of threads to use for parallel execution. If \
    set to -1, the method will use all available threads.
    :type threads: int
    :param verbose: If set to True, the method will display the progress of \
    the node aggregation process using the tqdm library.

    :return alpha: The calibrated value for the alpha parameter.
    :rtype alpha: float
    """

    # Create the object for parallel execution using the _AlphaGivenDeltaGraph
    # class.
    graphs = _AlphaGivenDeltaGraph(n, delta)

    # Main loop
    if verbose:
        iterable = tqdm(range(max_iterations), desc="Calibrating alpha", disable=not verbose)
    else:
        iterable = range(max_iterations)
    for _ in iterable:
        # Create partitions for the search space
        partitions = np.linspace(lo_alpha, hi_alpha, intervals + 1)

        # create list of alphas (repetitions times for each partition)
        alphas = [part for part in partitions for _ in range(repetitions)]

        # Executing the __call__ method for each object in parallel using Pool class.
        # This computation returns a list of ints with the number of edges for
        # each delta in deltas. The list is reshaped to a matrix with intervals
        # rows and repetitions columns. The average number of edges for each
        # partition is computed and stored in avg_edges.
        with Pool(threads) as pool:
            edges = pool.map(graphs, alphas)
        edges = np.array(edges).reshape(intervals + 1, repetitions)
        avg_edges = np.mean(edges, axis=1)

        # Check if the behaviour of the average number of edges is monotonic.
        # If it is, narrow the search space to the partition including the
        # target number of edges. Otherwise, return the average of the limits
        # of the partition with the expected number of edges.
        if np.all(np.diff(avg_edges) > 0):
            # monotonic increasing
            hi_alpha = partitions[avg_edges >= target_edges][0]
            lo_alpha = partitions[avg_edges < target_edges][-1]
        elif np.all(np.diff(avg_edges) < 0):
            # monotonic decreasing
            hi_alpha = partitions[avg_edges <= target_edges][0]
            lo_alpha = partitions[avg_edges > target_edges][-1]
        else:
            # Counting the number of partitions with more edges than the previous
            # partition. If the number of partitions with more edges is greater
            # than the number of partitions with fewer edges than the previous
            # partition, the behaviour is monotonic increasing. Otherwise, it
            # is monotonic decreasing.
            increasing_count = np.sum(np.diff(avg_edges) > 0)
            decreasing_count = np.sum(np.diff(avg_edges) < 0)
            if increasing_count > decreasing_count:
                # Finding the index i such that partitions[i] is the last
                # partition with fewer edges than the target number of edges.
                # The target number of edges is located in the interval
                # [partitions[i], partitions[i+1]].
                i = np.where(avg_edges < target_edges)[0][-1]
                return np.mean(partitions[i:i + 2])
            else:
                # Finding the index i such that partitions[i] is the last
                # partition with more edges than the target number of edges.
                # The target number of edges is located in the interval
                # [partitions[i], partitions[i+1]].
                i = np.where(avg_edges > target_edges)[0][-1]
                return np.mean(partitions[i:i + 2])

        # If the maximum number of iterations is reached, return the average of
        # the limits of the search space.
    return np.mean([lo_alpha, hi_alpha])


def synth_graph_from_network(g, alpha=None, lo_delta=0.0, hi_delta=0.99, intervals=10,
                             repetitions=100, max_iterations=5, threads=-1, verbose=False,
                             seed=None, track_evolution=False):
    """
    This function generates a synthetic graph from a given network using the
    extended duplication-divergence model. The parameters of the model are
    calibrated by considering the number of vertices as n and searching for
    the value of delta that generates a graph with the same number of edges
    as the given network. The alpha parameter should be given as input to
    the function.

    :param g: The network used as base to extract information to generate \
    the synthetic graph.
    :type g: networkx.Graph
    :param alpha: The alpha parameter for the extended duplication-divergence. \
    if set to None, the duplication-divergence model is used.
    :type alpha: float
    :param lo_delta: The lower bound for the delta parameter.
    :type lo_delta: float
    :param hi_delta: The upper bound for the delta parameter.
    :type hi_delta: float
    :param intervals: The number of partitions for the search space.
    :type intervals: int
    :param repetitions: The number of graphs generated for each partition.
    :type repetitions: int
    :param max_iterations: The maximum number of iterations for the binary \
    search algorithm.
    :type max_iterations: int
    :param threads: The number of threads to use for parallel execution. If \
    set to -1, the method will use all available threads.
    :type threads: int
    :param verbose: If set to True, the method will display the progress of \
    the node aggregation process using the tqdm library.
    :type verbose: bool
    :param seed: The seed for the random number generator.
    :type seed: int
    :param track_evolution: If set to True, the method will return the \
    evolution of the number of edges for each partition.
    :type track_evolution: bool

    :return g_synth: The synthetic graph generated using the extended \
    duplication-divergence model.
    :rtype g_synth: networkx.Graph
    """
    # Get the number of vertices and edges of the given network.
    n = g.number_of_nodes()
    target_edges = g.number_of_edges()

    # Calibrate the delta parameter.
    delta = calibrate_delta(n, target_edges, alpha, lo_delta, hi_delta, intervals,
                            repetitions, max_iterations, threads, verbose)

    # Generate the synthetic graph.
    if alpha is None:
        g_synth = duplication_divergence_graph(n, delta, seed=seed,
                                               track_evolution=track_evolution,
                                               verbose=verbose)
    else:
        g_synth = extended_duplication_divergence_graph(n, delta, alpha, seed=seed,
                                                        track_evolution=track_evolution,
                                                        verbose=verbose)

    # Return the synthetic graph.
    return g_synth


def synth_graph_from_intersection(g1, g2, alpha=None, lo_delta=0.0, hi_delta=0.99,
                                  intervals=10, repetitions=100, max_iterations=5,
                                  threads=-1, verbose=False, seed=None,
                                  track_evolution=False):
    """
    This function generates a synthetic graph from the intersection of two
    given networks using the extended duplication-divergence model. The
    parameters of the model are calibrated by considering the number of
    vertices as n and searching for the value of delta that generates a
    graph with the same number of edges as the given network. The alpha
    parameter should be given as input to the function.

    The number of nodes to use is the intersection between the two networks,
    g1 and g2. The number of edges to use is the maximum between the number
    of edges from g1 having both nodes in the node intersection and the number
    of edges from g2 having both nodes in the node intersection.

    :param g1: The first network used as base to extract information to \
    generate the synthetic graph.
    :type g1: networkx.Graph
    :param g2: The second network used as base to extract information to \
    generate the synthetic graph.
    :type g2: networkx.Graph
    :param alpha: The alpha parameter for the extended duplication-divergence. \
    if set to None, the duplication-divergence model is used.
    :type alpha: float
    :param lo_delta: The lower bound for the delta parameter.
    :type lo_delta: float
    :param hi_delta: The upper bound for the delta parameter.
    :type hi_delta: float
    :param intervals: The number of partitions for the search space.
    :type intervals: int
    :param repetitions: The number of graphs generated for each partition.
    :type repetitions: int
    :param max_iterations: The maximum number of iterations for the binary \
    search algorithm.
    :type max_iterations: int
    :param threads: The number of threads to use for parallel execution. If \
    set to -1, the method will use all available threads.
    :type threads: int
    :param verbose: If set to True, the method will display the progress of \
    the node aggregation process using the tqdm library.
    :type verbose: bool
    :param seed: The seed for the random number generator.
    :type seed: int
    :param track_evolution: If set to True, the method will return the \
    evolution of the number of edges for each partition.
    :type track_evolution: bool

    :return g_synth: The synthetic graph generated using the extended \
    duplication-divergence model.
    """
    # Generate the subset of nodes appearing in both networks.
    nodes = set(g1.nodes()).intersection(set(g2.nodes()))
    n = len(nodes)

    # Generate the subset of edges appearing in both networks using the
    # intersection of the nodes.
    g1_edges = nx.subgraph(g1, nodes).edges()
    g2_edges = nx.subgraph(g2, nodes).edges()
    target_edges = max(len(g1_edges), len(g2_edges))

    # Calibrate the delta parameter.
    delta = calibrate_delta(n, target_edges, alpha, lo_delta, hi_delta, intervals,
                            repetitions, max_iterations, threads, verbose)

    # Generate the synthetic graph.
    if alpha is None:
        g_synth = duplication_divergence_graph(n, delta, seed=seed,
                                               track_evolution=track_evolution,
                                               verbose=verbose)
    else:
        g_synth = extended_duplication_divergence_graph(n, delta, alpha, seed=seed,
                                                        track_evolution=track_evolution,
                                                        verbose=verbose)

    # Return the synthetic graph.
    return g_synth


if __name__ == "__main__":
    # Set the parameters for the extended duplication-divergence model.
    N = 100
    DELTAS = [0.1, 0.2, 0.3, 0.4, 0.5]
    ALPHAS = [0.01, 0.05, 0.1, 0.15]
    TARGET_EDGES = 1500
    INTERVALS = 10
    REPETITIONS = 100
    MAX_ITERATIONS = 10
    THREADS = 7
    VERBOSE = False

    timestamp("Calibration of delta values starts.")
    # Calibrate the delta parameter, without alpha.
    d = calibrate_delta(N, TARGET_EDGES, None, intervals=INTERVALS, repetitions=REPETITIONS,
                        max_iterations=MAX_ITERATIONS, threads=THREADS, verbose=VERBOSE)
    print(f"alpha = None --> delta = {d:0.5f}")

    # Calibrate the delta parameter, given alpha.
    for a in ALPHAS:
        d = calibrate_delta(N, TARGET_EDGES, a, intervals=INTERVALS, repetitions=REPETITIONS,
                            max_iterations=MAX_ITERATIONS, threads=THREADS, verbose=VERBOSE)
        print(f"alpha = {a:0.2f} --> delta = {d:0.5f}")

    timestamp("Calibration of alpha values starts.")
    # Calibrate the alpha parameter.
    for d in DELTAS:
        a = calibrate_alpha(N, TARGET_EDGES, d, intervals=INTERVALS, repetitions=REPETITIONS,
                            max_iterations=MAX_ITERATIONS, threads=THREADS, verbose=VERBOSE)
        print(f"delta = {d:0.2f} --> alpha = {a:0.5f}")
    timestamp("Calibration ends.")
