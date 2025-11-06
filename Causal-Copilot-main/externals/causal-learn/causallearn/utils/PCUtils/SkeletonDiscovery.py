from __future__ import annotations

from itertools import combinations
from functools import partial

import numpy as np
from numpy import ndarray
from typing import List, Tuple, Set
from tqdm.auto import tqdm
import joblib

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import CIT


def _process_node_pair(
    x: int, 
    y: int, 
    depth: int, 
    cg: CausalGraph, 
    alpha: float, 
    stable: bool, 
    background_knowledge: BackgroundKnowledge | None, 
    verbose: bool,
    domain_index: int | None = None,
    non_linear_cit: CIT | None = None,
    backup_indep_test: CIT | None = None
) -> Tuple[List[Tuple[int, int]], dict]:
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
    background_knowledge : background knowledge
    verbose : bool, whether to print verbose output
    domain_index : int or None, the index of the domain column
    non_linear_cit : CIT or None, the non-linear CIT to use for tests involving domain_index
    backup_indep_test : CIT or None, the backup indep test to use for fastkci bug

    Returns
    -------
    edge_removal : list of tuples, edges to remove
    sepsets_dict : dict, separation sets for each edge
    """
    edge_removal = []
    sepsets_dict = {}
    
    knowledge_ban_edge = False
    sepsets = set()
    
    if background_knowledge is not None and (
            background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
            and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
        knowledge_ban_edge = True
        
    if knowledge_ban_edge:
        if not stable:
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                edge_removal.append((x, y))
            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
            if edge2 is not None:
                edge_removal.append((y, x))
            sepsets_dict[(x, y)] = ()
            sepsets_dict[(y, x)] = ()
            return edge_removal, sepsets_dict
        else:
            edge_removal.append((x, y))
            edge_removal.append((y, x))
            
    Neigh_x = cg.neighbors(x)
    Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
    
    for S in combinations(Neigh_x_noy, depth):
        # Use non_linear_cit if domain_index is in the test, otherwise use the default cit
        try:
            if domain_index is not None and (x == domain_index or y == domain_index or domain_index in S):
                p = non_linear_cit(x, y, S) if non_linear_cit is not None else cg.ci_test(x, y, S)
            else:
                p = cg.ci_test(x, y, S)
        except ValueError as e:
            # fastkci bug:
            # null_sample_spectrum error: ValueError: zero-size array to reduction operation maximum which has no identity
            # use backup_indep_test (kci) as a workaround
            p = backup_indep_test(x, y, S)

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
                sepsets_dict[(x, y)] = S
                sepsets_dict[(y, x)] = S
                return edge_removal, sepsets_dict
            else:
                edge_removal.append((x, y))
                edge_removal.append((y, x))
                for s in S:
                    sepsets.add(s)
        else:
            if verbose:
                print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                
    sepsets_dict[(x, y)] = tuple(sepsets)
    sepsets_dict[(y, x)] = tuple(sepsets)
    
    return edge_removal, sepsets_dict


def skeleton_discovery(
    data: ndarray, 
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    depth: int = -1,
    n_jobs: int = 4,
    domain_index: int | None = None,
    non_linear_cit: CIT | None = None
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    n_jobs: int, number of parallel jobs (default = 4)
    domain_index: int, the index of the domain column in the data, if it is not None, use KCI for independence test including it

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)
    backup_indep_test = CIT(data, method='kci')

    if depth == -1:
        depth = no_of_var
    current_depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    
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
            background_knowledge=background_knowledge, 
            verbose=verbose,
            domain_index=domain_index,
            non_linear_cit=non_linear_cit,
            backup_indep_test=backup_indep_test
        )
        
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(process_func)(x, y) for x, y in node_pairs
        )
        
        # Collect all edges to remove and sepsets
        all_edge_removals = []
        all_sepsets = {}
        
        for edge_removal, sepsets_dict in results:
            all_edge_removals.extend(edge_removal)
            all_sepsets.update(sepsets_dict)
        
        # Update sepsets
        for (x, y), sepset in all_sepsets.items():
            if (x, y) in all_edge_removals or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                append_value(cg.sepset, x, y, sepset)
        
        # Remove edges
        for (x, y) in list(set(all_edge_removals)):
            edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge is not None:
                cg.G.remove_edge(edge)

    if show_progress:
        pbar.close()

    return cg