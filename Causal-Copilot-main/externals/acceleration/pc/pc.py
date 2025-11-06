import numpy as np
import cupy as cp
import dask.array as da
from gpucsl.pc.pc import GaussianPC, DiscretePC
import networkx as nx

# use the local causal-learn package
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
else:
    sys.path.insert(0, causal_learn_dir)

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from typing import Tuple, Dict
import itertools
sys.path.append(os.path.join(root_dir, 'externals', 'pc_adjacency_search'))
import gpucmiknn
import globals
from gpu_ci import gpu_single, gpu_row

def apply_rules(dag: nx.DiGraph):
    def non_adjacent(g, v_1, v_2):
        return not g.has_edge(v_1, v_2) and not g.has_edge(v_2, v_1)

    def existing_edge_is_directed_only(g, v_1, v_2):
        return not g.has_edge(v_2, v_1)

    def undirected(g, v_1, v_2):
        return g.has_edge(v_1, v_2) and g.has_edge(v_2, v_1)

    num_nodes = len(dag.nodes)

    def column_major_edge_ordering(edge):
        return edge[1] * num_nodes + edge[0]

    while True:
        graph_changed = False

        # Rule 1
        # v_1 -> v_2 - v_3 to v_1 -> v_2 -> v_3
        dag2 = dag.copy()
        for v_1, v_2 in sorted(dag2.edges, key=column_major_edge_ordering):
            if dag2.has_edge(v_2, v_1):
                continue
            for v_3 in sorted(dag2.successors(v_2)):
                if v_1 == v_3:
                    continue
                if dag2.has_edge(v_3, v_2) and non_adjacent(dag2, v_1, v_3):
                    # only no conflict solution
                    if undirected(dag, v_2, v_3):
                        dag.add_edge(v_2, v_3)
                        dag.remove_edges_from([(v_3, v_2)])
                        graph_changed = True

        # Rule 2
        # v_1 -> v_3 -> v_2 with v_1 - v_2: v_1 -> v_2
        dag2 = dag.copy()  # work on current dag after Rule 1
        for v_1, v_2 in sorted(dag2.edges, key=column_major_edge_ordering):
            if not dag2.has_edge(v_2, v_1):
                continue
            for v_3 in sorted(
                set(dag2.successors(v_1)).intersection(dag2.predecessors(v_2))
            ):
                if existing_edge_is_directed_only(
                    dag2, v_1, v_3
                ) and existing_edge_is_directed_only(dag2, v_3, v_2):
                    dag.add_edge(v_1, v_2)
                    dag.remove_edges_from([(v_2, v_1)])
                    graph_changed = True

        # Rule 3
        # ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
        # │v_3├───┤v_1├───┤v_4│   │v_3├───┤v_1├───┤v_4│
        # └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
        #   │       │       │  to   │       │       │
        #   │       │       │       │       │       │
        #   │     ┌─┴─┐     │       │     ┌─▼─┐     │
        #   └────►│v_2│◄────┘       └────►│v_2│◄────┘
        #         └───┘                   └───┘
        dag2 = dag.copy()  # work on current dag after Rule 2
        for v_1, v_2 in sorted(dag2.edges, key=column_major_edge_ordering):
            if not dag2.has_edge(v_2, v_1):
                continue
            neighbors_v1 = set(dag2.successors(v_1)).intersection(
                dag2.predecessors(v_1)
            )
            predecessors_v2 = set(dag2.predecessors(v_2)).difference(
                dag2.successors(v_2)
            )
            C = sorted(
                neighbors_v1.intersection(predecessors_v2),
            )
            for v_3, v_4 in itertools.combinations(C, 2):
                if non_adjacent(dag2, v_3, v_4):
                    dag.add_edge(v_1, v_2)
                    dag.remove_edges_from([(v_2, v_1)])
                    graph_changed = True

        if not graph_changed:
            return


def orient_v_structure(
    dag: nx.DiGraph,
    separation_sets: np.ndarray,
    skeleton: nx.Graph = None,
) -> None:
    def in_separation_set(v, v_1, v_2):
        return v in separation_sets[v_1][v_2]

    if skeleton is None:
        skeleton = dag.to_undirected()

    def non_adjacent(v_1, v_2):
        return not skeleton.has_edge(v_1, v_2)

    num_nodes = len(skeleton.nodes)

    for v_1, v_2 in sorted(
        skeleton.to_directed().edges, key=lambda x: x[1] * num_nodes + x[0]
    ):
        for v_3 in sorted(skeleton.neighbors(v_2), reverse=False):
            if v_1 == v_3:
                continue
            if non_adjacent(v_1, v_3) and not (
                in_separation_set(v_2, v_1, v_3) or in_separation_set(v_2, v_3, v_1)
            ):
                dag.add_edges_from([(v_1, v_2), (v_3, v_2)])
                dag.remove_edges_from([(v_2, v_1), (v_2, v_3)])

# GPU-accelerated KCI test using Dask + CuPy
def kci_dask_cupy(X: int, Y: int, Z: list, data: da.Array, alpha: float, gamma: float) -> bool:
    n_samples = data.shape[0]

    if len(Z) == 0:
        X_data = data[:, X].reshape(-1, 1)
        Y_data = data[:, Y].reshape(-1, 1)
    else:
        Z_data = data[:, Z]
        X_data = data[:, X]
        Y_data = data[:, Y]

        # Residualize X on Z
        beta_X = cp.linalg.lstsq(Z_data, X_data, rcond=None)[0]
        X_data = X_data - Z_data @ beta_X

        # Residualize Y on Z
        beta_Y = cp.linalg.lstsq(Z_data, Y_data, rcond=None)[0]
        Y_data = Y_data - Z_data @ beta_Y

    # Compute RBF kernels
    K_X = cp.exp(-gamma * cp.linalg.norm(X_data[:, None] - X_data, axis=2) ** 2)
    K_Y = cp.exp(-gamma * cp.linalg.norm(Y_data[:, None] - Y_data, axis=2) ** 2)

    # Center kernels
    H = cp.eye(n_samples) - cp.ones((n_samples, n_samples)) / n_samples
    Kc_X = H @ K_X @ H
    Kc_Y = H @ K_Y @ H

    # Compute HSIC statistic
    hsic_stat = cp.trace(Kc_X @ Kc_Y) / ((n_samples - 1) ** 2)

    return hsic_stat < alpha


# Fisher-Z with GPU acceleration using gpucsl
def fisherz_gpu_gpucsl(data: np.ndarray, alpha: float, depth: int, node_names: list) -> Tuple[np.ndarray, Dict, CausalGraph]:
    if depth < 0:
        depth = data.shape[1]
    pc_result = GaussianPC(data, depth, alpha).set_distribution_specific_options().execute()
    ((directed_graph, separation_sets, _, _, _, _), pc_runtime) = pc_result

    # Convert directed graph to adjacency matrix
    adj_matrix = np.zeros((directed_graph.number_of_nodes(), directed_graph.number_of_nodes()))
    for edge in directed_graph.edges():
        adj_matrix[edge[0], edge[1]] = 1
    
    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = -1
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = 1
    
    info = {
        'sepset': separation_sets,
        'PC_elapsed': pc_runtime,
    }
    return adj_matrix, info


# Chi-Square with GPU acceleration using gpucsl
def chi_square_gpu_gpucsl(data: np.ndarray, alpha: float, depth: int, node_names: list) -> Tuple[np.ndarray, Dict, CausalGraph]:
    if depth < 0:
        depth = data.shape[1]
    pc_result = DiscretePC(data, depth, alpha).set_distribution_specific_options().execute()
    ((directed_graph, separation_sets, _, _, _, _), pc_runtime) = pc_result
    # Convert directed graph to adjacency matrix
    adj_matrix = np.zeros((directed_graph.number_of_nodes(), directed_graph.number_of_nodes()))
    for edge in directed_graph.edges():
        adj_matrix[edge[0], edge[1]] = 1
    
    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = -1
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = 1
    info = {
        'sepset': separation_sets,
        'PC_elapsed': pc_result.PC_elapsed,
    }
    return directed_graph, info


# CMIknn with GPU acceleration
def cmiknn_gpu(data: np.ndarray, alpha: float, depth: int, node_names: list) -> Tuple[np.ndarray, Dict, CausalGraph]:
    if depth < 0:
        depth = data.shape[1]

    globals.init()
    globals.alpha = alpha
    globals.vertices = data.shape[1]
    globals.permutations = 100
    globals.max_sepset_size = data.shape[1] - 2 if depth is None else min(depth, data.shape[1] - 2)

    globals.start_level = 0

    globals.split_size = None
    globals.k_cmi = int(np.sqrt(data.shape[0]) / 5) # min(400, int(data.shape[0] * 0.2)) #'adaptive' # int(data.shape[0] * 0.2)

    print(f"CMI KNN hyperparameters: alpha={alpha}, depth={depth}, k_cmi={globals.k_cmi}, max_sepset_size={globals.max_sepset_size}, permutations={globals.permutations}, split_size={globals.split_size}, start_level={globals.start_level}")
    
    # Initialize GPU
    globals.gpu_free_mem = gpucmiknn.init_gpu()
        
    # Get skeleton and sepsets using GPU-accelerated CMIknn
    skeleton, sepsets = gpu_single(data)

    for i in range(skeleton.shape[0]):
        skeleton[i, i] = 0
    
    # Convert separation set from dict to matrix format
    n_nodes = data.shape[1]
    separation_sets = np.full((n_nodes, n_nodes, globals.max_sepset_size), -1, dtype=int)
    
    for (i, j), info in sepsets.items():
        sepset = info['sepset']
        for k, node in enumerate(sepset):
            if k < globals.max_sepset_size: 
                separation_sets[i, j, k] = node
                separation_sets[j, i, k] = node
    
    # Orient edges
    dag = nx.DiGraph(skeleton).to_directed()
    orient_v_structure(dag, separation_sets)
    apply_rules(dag)
    
    # Convert directed graph to adjacency matrix
    adj_matrix = np.zeros((data.shape[1], data.shape[1]))
    for edge in dag.edges():
        adj_matrix[edge[0], edge[1]] = 1
    
    # Format adjacency matrix according to causal-learn conventions
    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = -1
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = 1
    
    info = {
        'sepset': sepsets,
        'PC_elapsed': 0.0,  # We don't have timing info from gpu_single
    }
    
    return adj_matrix, info


# Unified PC algorithm function
def accelerated_pc(
    data: np.ndarray, alpha: float = 0.05, indep_test: str = 'fisherz', depth: int = -1, gamma: float = 1.0
) -> Tuple[np.ndarray, Dict, CausalGraph]:
    """
    Accelerated PC algorithm that chooses between GPU-accelerated implementations
    based on the independence test.
    
    Args:
        data: Input dataset as a NumPy array.
        alpha: Significance level.
        indep_test: Independence test to use ('fisherz', 'chi_square', 'kci', 'cmiknn').
        depth: Maximum conditioning set size.
        gamma: Bandwidth for KCI test.
        
    Returns:
        Tuple containing adjacency matrix, info dictionary, and CausalGraph object.
    """
    node_names = [str(i) for i in range(data.shape[1])]

    if indep_test == 'fisherz':
        return fisherz_gpu_gpucsl(data, alpha, depth, node_names)

    elif indep_test == 'chisq':
        return chi_square_gpu_gpucsl(data, alpha, depth, node_names)

    elif indep_test == 'cmiknn':
        return cmiknn_gpu(data, alpha, depth, node_names)

    # elif indep_test == 'kci':
    #     data_dask = da.from_array(cp.asarray(data), chunks=data.shape)
    #     pc_result = pc(
    #         data,
    #         alpha=alpha,
    #         indep_test=lambda X, Y, Z: kci_dask_cupy(X, Y, Z, data_dask, alpha, gamma),
    #         depth=depth,
    #     )
    #     directed_graph = pc_result.G.graph
    #     cg = CausalGraph(len(node_names), node_names)
    #     cg.G = directed_graph
    #     return directed_graph, {}, cg

    else:
        raise ValueError(f"Independence test '{indep_test}' not supported. Use 'fisherz', 'chisq', or 'cmiknn'.")
