import sys
import numpy as np
import time
# import psutil

# use the local causal-learn package
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
sys.path.insert(0, causal_learn_dir)

from gpucsl.pc.discover_skeleton_gaussian import discover_skeleton_gpu_gaussian
from gpucsl.pc.discover_skeleton_discrete import discover_skeleton_gpu_discrete
from gpucsl.pc.helpers import init_pc_graph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from typing import Tuple, Dict, Optional
import networkx as nx


import itertools
import logging

import networkx as nx
import numpy as np

import itertools
import logging

import networkx as nx

# Add path for GPU-accelerated CMIknn
sys.path.append(os.path.join(root_dir, 'externals', 'pc_adjacency_search'))
import gpucmiknn
import globals
from gpu_ci import gpu_single


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
                        logging.debug(f"R1: remove ({v_3, v_2})")
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
                    logging.debug(f"R2: remove ({v_2, v_1})")
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
                    logging.debug(f"R3: remove ({v_2, v_1})")
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
                logging.debug(f"v: {[(v_2, v_1), (v_2, v_3)]}")
                dag.add_edges_from([(v_1, v_2), (v_3, v_2)])
                dag.remove_edges_from([(v_2, v_1), (v_2, v_3)])


# **GPU-Accelerated Skeleton Discovery**
def skeleton_discovery_gpu(data: np.ndarray, alpha: float, indep_test: str = 'fisherz', max_level: int = -1) -> Tuple[nx.Graph, dict]:
    """
    Perform skeleton discovery using GPU-accelerated methods.
    
    Args:
        data: Input dataset as a NumPy array (samples, variables)
        alpha: Significance level
        indep_test: Independence test to use ('fisherz', 'discrete', or 'cmiknn')
        max_level: Maximum conditioning set size
        
    Returns:
        Tuple containing skeleton graph and separation sets
    """
    num_vars = data.shape[1]
    
    # Run GPU-accelerated skeleton discovery
    if indep_test == 'fisherz':
        if max_level < 0:
            max_level = num_vars
        from gpucsl.pc.pc import GaussianPC
        pc = GaussianPC(data=data, alpha=alpha, max_level=max_level, should_log=False)
        ((result, separation_sets, _, _), _) = pc.set_distribution_specific_options().discover_skeleton()
        result[:, -1] = 0
        skeleton = nx.DiGraph(result)
    elif indep_test == 'chisq':
        if max_level < 0:
            max_level = num_vars
        from gpucsl.pc.pc import DiscretePC
        pc = DiscretePC(data=data, alpha=alpha, max_level=max_level, should_log=False)
        ((result, separation_sets, _, _), _) = pc.set_distribution_specific_options().discover_skeleton()
        result[:, -1] = 0
        skeleton = nx.DiGraph(result)
    elif indep_test == 'cmiknn':
        # Initialize globals for CMIknn
        globals.init()
        globals.alpha = alpha
        globals.vertices = data.shape[1]
        globals.permutations = 100
        globals.max_sepset_size = data.shape[1] - 2 if max_level < 0 else min(max_level, data.shape[1] - 2)
        globals.start_level = 0
        globals.split_size = None
        globals.k_cmi = min(int(data.shape[0] * 0.07), 200)
        
        # Initialize GPU
        gpucmiknn.init_gpu()
        
        # Get skeleton and sepsets using GPU-accelerated CMIknn
        adj_matrix, sep_dict = gpu_single(data)

        for i in range(adj_matrix.shape[0]):
            adj_matrix[i, i] = 0
            adj_matrix[i, -1] = 0
        
        # Convert separation set from dict to matrix format
        n_nodes = data.shape[1]
        separation_sets = np.full((n_nodes, n_nodes, globals.max_sepset_size), -1, dtype=int)
        
        for (i, j), info in sep_dict.items():
            sepset = info['sepset']
            for k, node in enumerate(sepset):
                if k < globals.max_sepset_size: 
                    separation_sets[i, j, k] = node
                    separation_sets[j, i, k] = node

        # Convert to networkx graph
        skeleton = nx.DiGraph(adj_matrix)
    else:
        raise ValueError(f"Unsupported independence test: {indep_test}")
        
    return skeleton, separation_sets


def orient_edges(
    skeleton: nx.Graph,
    separation_sets: np.ndarray,
) -> np.ndarray:
    dag = skeleton.to_directed()

    # Then orient v-structures based on the separation sets
    orient_v_structure(dag, separation_sets)
    
    # Apply Meek rules to orient remaining edges
    apply_rules(dag)

    # Convert directed graph to adjacency matrix
    adj_matrix = np.zeros((dag.number_of_nodes(), dag.number_of_nodes()))
    for edge in dag.edges():
        adj_matrix[edge[0], edge[1]] = 1

    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = -1
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
            adj_matrix[i, j] = -1
            adj_matrix[j, i] = 1
    
    return adj_matrix

# **Accelerated CDNOD using GPuCSL**
def accelerated_cdnod(
    data: np.ndarray, c_indx: np.ndarray, indep_test: str = 'fisherz', alpha: float = 0.05, depth: int = -1) -> Tuple[np.ndarray, Dict, CausalGraph, float, float]:
    """
    Accelerated CDNOD algorithm that runs with GPuCSL acceleration.
    
    Args:
        data: Input dataset as a NumPy array.
        c_indx: Context index capturing domain shifts.
        alpha: Significance level.
        depth: Maximum conditioning set size.
        uc_rule: Rule for unshielded collider orientation.
        uc_priority: Rule for prioritizing collider conflicts.
        
    Returns:
        Tuple containing adjacency matrix, info dictionary, CausalGraph object, runtime, and memory usage.
    """

    # Append `c_indx` to data
    data_aug = np.concatenate((data, c_indx.reshape(-1, 1)), axis=1)

    # **Perform GPU-Accelerated Skeleton Discovery**
    skeleton, separation_sets = skeleton_discovery_gpu(data_aug, alpha, indep_test, depth)

    # **Orient edges using UCSepset and Meek**
    adj_matrix = orient_edges(skeleton, separation_sets)
    
    return adj_matrix
