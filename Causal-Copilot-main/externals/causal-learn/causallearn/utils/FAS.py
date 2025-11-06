from __future__ import annotations

from itertools import combinations
from functools import partial
from typing import List, Dict, Tuple, Set

import joblib
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Node import Node
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def _process_node_pair(
    x: int, 
    y: int, 
    depth: int, 
    cg: CausalGraph, 
    alpha: float, 
    stable: bool, 
    knowledge: BackgroundKnowledge | None, 
    verbose: bool
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int, Set[int]], float]]:
    """
    Process a single node pair for conditional independence testing
    
    Parameters
    ----------
    x : int, first node index
    y : int, second node index
    depth : int, current depth of conditioning set
    cg : CausalGraph, the current causal graph
    alpha : float, significance level
    stable : bool, whether to use stable PC
    knowledge : background knowledge
    verbose : bool, whether to print verbose output

    Returns
    -------
    edge_removal : list of tuples, edges to remove
    sepsets_dict : dict, separation sets for each edge
    test_results : dict, results of conditional independence tests
    """
    edge_removal = []
    sepsets_dict = {}
    test_results = {}
    
    knowledge_ban_edge = False
    sepsets = set()
    
    if knowledge is not None and (
            knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
            and knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
        knowledge_ban_edge = True
        
    if knowledge_ban_edge:
        if not stable:
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                edge_removal.append((x, y))
            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
            if edge2 is not None:
                edge_removal.append((y, x))
            sepsets_dict[(x, y)] = set()
            sepsets_dict[(y, x)] = set()
            return edge_removal, sepsets_dict, test_results
        else:
            edge_removal.append((x, y))
            edge_removal.append((y, x))
            
    Neigh_x = cg.neighbors(x)
    Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
    
    for S in combinations(Neigh_x_noy, depth):
        p = cg.ci_test(x, y, S)
        test_results[(x, y, S)] = p
        
        if p > alpha:
            if verbose:
                print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
            if not stable:
                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                if edge1 is not None:
                    edge_removal.append((x, y))
                edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                if edge2 is not None:
                    edge_removal.append((y, x))
                sepsets_dict[(x, y)] = set(S)
                sepsets_dict[(y, x)] = set(S)
                return edge_removal, sepsets_dict, test_results
            else:
                edge_removal.append((x, y))
                edge_removal.append((y, x))
                for s in S:
                    sepsets.add(s)
        else:
            if verbose:
                print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                
    sepsets_dict[(x, y)] = sepsets
    sepsets_dict[(y, x)] = sepsets
    
    return edge_removal, sepsets_dict, test_results


def fas(data: ndarray, nodes: List[Node], independence_test_method: CIT_Base, alpha: float = 0.05,
        knowledge: BackgroundKnowledge | None = None, depth: int = -1,
        verbose: bool = False, stable: bool = True, show_progress: bool = True, n_jobs: int = 4) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int, Set[int]], float]]:
    """
    Implements the "fast adjacency search" used in several causal algorithm in this file. In the fast adjacency
    search, at a given stage of the search, an edge X*-*Y is removed from the graph if X _||_ Y | S, where S is a subset
    of size d either of adj(X) or of adj(Y), where d is the depth of the search. The fast adjacency search performs this
    procedure for each pair of adjacent edges in the graph and for each depth d = 0, 1, 2, ..., d1, where d1 is either
    the maximum depth or else the first such depth at which no edges can be removed. The interpretation of this adjacency
    search is different for different algorithm, depending on the assumptions of the algorithm. A mapping from {x, y} to
    S({x, y}) is returned for edges x *-* y that have been removed.

    Parameters
    ----------
    data: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    nodes: The search nodes.
    independence_test_method: the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    knowledge: background background_knowledge
    depth: the depth for the fast adjacency search, or -1 if unlimited
    verbose: True is verbose output should be printed or logged
    stable: run stabilized skeleton discovery if True (default = True)
    show_progress: whether to use tqdm to show progress bar
    n_jobs: int, number of parallel jobs (default = 4)
    Returns
    -------
    graph: Causal graph skeleton, where graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j.
    sep_sets: Separated sets of graph
    test_results: Results of conditional independence tests
    """
    ## ------- check parameters ------------
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if not all(isinstance(node, Node) for node in nodes):
        raise TypeError("'nodes' must be 'List[Node]' type!")
    if not isinstance(independence_test_method, CIT_Base):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
    if knowledge is not None and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'knowledge' must be 'BackgroundKnowledge' type!")
    if type(depth) != int or depth < -1:
        raise TypeError("'depth' must be 'int' type >= -1!")
    ## ------- end check parameters ------------

    if depth == -1:
        depth = float('inf')

    no_of_var = data.shape[1]
    node_names = [node.get_name() for node in nodes]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(independence_test_method)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    test_results: Dict[Tuple[int, int, Set[int]], float] = {}

    def remove_if_exists(x: int, y: int) -> None:
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)

    pbar = tqdm(total=no_of_var) if show_progress else None
    current_depth: int = -1
    
    while cg.max_degree() - 1 > current_depth and current_depth < depth:
        current_depth += 1
        if show_progress:
            pbar.reset(total=no_of_var)
            pbar.set_description(f'Depth={current_depth}')
            
        # Collect all node pairs to test at this depth
        node_pairs = []
        for x in range(no_of_var):
            if show_progress:
                pbar.update(1)
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < current_depth - 1:
                continue
            for y in Neigh_x:
                if x < y:  # Process each pair only once
                    node_pairs.append((x, y))
        
        if not node_pairs:
            continue
            
        # Process node pairs in parallel
        process_func = partial(
            _process_node_pair, 
            depth=current_depth, 
            cg=cg, 
            alpha=alpha, 
            stable=stable, 
            knowledge=knowledge, 
            verbose=verbose
        )
        
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(process_func)(x, y) for x, y in node_pairs
        )
        
        # Collect all edges to remove, sepsets, and test results
        all_edge_removals = []
        all_sepsets = {}
        
        for edge_removal, sepsets_dict, test_result_dict in results:
            all_edge_removals.extend(edge_removal)
            all_sepsets.update(sepsets_dict)
            test_results.update(test_result_dict)
        
        # Update sepsets
        for (x, y), sepset in all_sepsets.items():
            if (x, y) in all_edge_removals or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                append_value(cg.sepset, x, y, tuple(sepset))
                sep_sets[(x, y)] = sepset
        
        # Remove edges
        for (x, y) in list(set(all_edge_removals)):
            edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge is not None:
                cg.G.remove_edge(edge)

    if show_progress:
        pbar.close()

    return cg.G, sep_sets, test_results
