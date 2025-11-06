import numpy as np
import pandas as pd
from typing import Set, Tuple, List, Callable, Dict, Optional
import networkx as nx
import matplotlib.pyplot as plt

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.insert(0, causal_learn_dir)

from causallearn.utils.cit import CIT, fisherz, kci, chisq, gsq


def MB2CPDAG(data: pd.DataFrame, mb_dict: Dict[int, List[int]], indep_test: str = 'fisherz', alpha: float = 0.05, n_jobs: int = 1) -> np.ndarray:
    """
    Convert Markov blanket results to a CPDAG by running PC algorithm on each target and its MB nodes.
    
    Args:
        data: pandas DataFrame containing the data
        mb_dict: Dictionary mapping each target node index to list of its MB node indices
        is_discrete: Whether the data is discrete (True) or continuous (False)
        alpha: Significance level for independence tests
        
    Returns:
        numpy array representing the merged CPDAG adjacency matrix
    """
    from causallearn.search.ConstraintBased.PC import pc
    
    n_vars = data.shape[1]
    merged_cpdag = np.zeros((n_vars, n_vars))
    
    def compute_local_cpdag(target, mb_nodes):
        local_indices = [target] + mb_nodes
        if len(local_indices) < 2:
            return None
        if len(local_indices) == 2:
            local_adj = np.zeros((2,2))
            local_adj[0,1] = 2
            return (local_indices, local_adj)
        local_data = data.iloc[:, local_indices].values
        node_names = [str(i) for i in local_indices]
        cg = pc(local_data,
                alpha=alpha,
                indep_test=indep_test,
                node_names=node_names,
                verbose=False)
        local_adj = cg.G.graph
        return (local_indices, local_adj)

    results = []
    for target, mb_nodes in mb_dict.items():
        result = compute_local_cpdag(target, mb_nodes)
        results.append(result)

    for res in results:
        if res is None:
            continue
        local_indices, local_adj = res
        for i in range(len(local_indices)):
            for j in range(len(local_indices)):
                if local_adj[i, j] == 1:
                    if local_adj[j, i] == -1:
                        merged_cpdag[local_indices[i], local_indices[j]] = 1
                    elif local_adj[j, i] == 1:
                        if merged_cpdag[local_indices[j], local_indices[i]] == 0:
                            merged_cpdag[local_indices[i], local_indices[j]] = 2
                elif local_adj[i, j] == -1 and local_adj[j, i] == -1:
                    if merged_cpdag[local_indices[j], local_indices[i]] == 0:
                        merged_cpdag[local_indices[i], local_indices[j]] = 2

    # Remove symmetries by keeping only one entry for undirected edges
    indices = np.where(merged_cpdag == 2)
    for i, j in zip(indices[0], indices[1]):
        if merged_cpdag[i,j] == merged_cpdag[j,i] == 2:
            merged_cpdag[i,j] = 2
            merged_cpdag[j,i] = 0
    indices = np.where(merged_cpdag == 1)
    for i, j in zip(indices[0], indices[1]):
        if merged_cpdag[i,j] == merged_cpdag[j,i] == 1:
            merged_cpdag[i,j] = 2
            merged_cpdag[j,i] = 0 

    return merged_cpdag


# class MB2CPDAG:
#     def __init__(self, data: pd.DataFrame, ci_test: str = 'fisherz'):
#         """
#         Initialize the classifier with a dataset and conditional independence test
        
#         Args:
#             data: DataFrame containing observations of all variables
#             ci_test: Conditional independence test to use ('fisherz', 'kci', 'chisq', 'gsq')
#         """
#         self.data = data.to_numpy()
#         self.column_names = data.columns.tolist()
#         self.ci_test = CIT(self.data, method=ci_test)
        
#     def conditional_independence_test(self, A: str, B: str, conditioning_set: List[str], alpha: float = 0.05) -> bool:
#         """
#         Test if A and B are conditionally independent given conditioning_set using the specified CI test
        
#         Args:
#             A: First variable name
#             B: Second variable name 
#             conditioning_set: List of conditioning variable names
#             alpha: Significance level for independence test
            
#         Returns:
#             bool: True if independent, False if dependent
#         """
#         A_idx = self.column_names.index(A)
#         B_idx = self.column_names.index(B)
#         conditioning_indices = [self.column_names.index(X) for X in conditioning_set]
        
#         print(f"A: {A_idx}, B: {B_idx}, conditioning_set: {conditioning_indices}")
#         pvalue = self.ci_test(A_idx, B_idx, conditioning_indices)
#         print(f"A: {A}, B: {B}, conditioning_set: {conditioning_set}, pvalue: {pvalue}, alpha: {alpha}")
#         return pvalue >= alpha
    
#     def classify_blanket_nodes(self, T: str, MB: List[str]) -> Dict[str, str]:
#         """
#         Classify Markov blanket nodes into spouses and parent/child nodes
        
#         Args:
#             T: Target node
#             MB: List of nodes in T's Markov blanket
            
#         Returns:
#             Dict[str, str]: Dictionary mapping each MB node to either 'spouse' or 'parent_or_child'
#         """
#         classification = {}
#         MB_set = set(MB)
        
#         for X in MB:
#             # Condition on all other MB nodes except X
#             conditioning_set = list(MB_set - {X})
            
#             # Test if T and X are independent given MB\{X}
#             # For parent or child, T and X are dependent whether conditioning on MB\{X} or not
#             # For spouse, T and X are independent before conditioning on MB\{X}, but dependent after conditioning on MB\{X}
#             is_independent_before = self.conditional_independence_test(T, X, [])
#             is_independent_after = self.conditional_independence_test(T, X, conditioning_set)
#             # If T ⟂ X | (MB(T)\{X}), then X is a spouse
#             # else X is a direct neighbor (parent or child)
#             if is_independent_before and not is_independent_after:
#                 classification[X] = 'spouse'
#             else:
#                 classification[X] = 'parent_or_child'
                
#         return classification
    
    

# def generate_test_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, Dict[str, str]]:
#     """
#     Generate synthetic data from a known Bayesian network structure
    
#     Returns:
#         Tuple[pd.DataFrame, Dict[str, str]]: 
#             - Generated data
#             - True classification of nodes
#     """
#     # Generate parent nodes
#     np.random.seed(42)
#     data = pd.DataFrame({
#         'P1': np.random.normal(0, 1, n_samples),
#         'P2': np.random.normal(0, 1, n_samples),
#         'S1': np.random.normal(0, 1, n_samples),
#         'S2': np.random.normal(0, 1, n_samples)
#     })
    
#     # Generate target node X based on parents
#     data['X'] = (0.9 * data['P1'] + 
#                  0.8 * data['P2'] + 
#                  np.random.normal(0, 0.5, n_samples))
    
#     # Generate child node based on X and spouse nodes
#     data['C1'] = (0.8 * data['X'] + 
#                   0.7 * data['S1'] + 
#                   0.6 * data['S2'] + 
#                   np.random.normal(0, 0.5, n_samples))
    
#     true_classification = {
#         'P1': 'parent_or_child',
#         'P2': 'parent_or_child', 
#         'S1': 'spouse',
#         'S2': 'spouse',
#         'C1': 'parent_or_child'
#     }
    
#     return data, true_classification

# def visualize_network(classification: Dict[str, str], T: str):
#     """
#     Visualize the discovered network structure
#     """
#     G = nx.DiGraph()
    
#     # Add nodes
#     G.add_node(T, color='lightblue')
#     for node, node_type in classification.items():
#         if node_type == 'spouse':
#             G.add_node(node, color='lightgreen')
#         else:
#             G.add_node(node, color='lightcoral')
    
#     # Add edges (simplified visualization)
#     for node, node_type in classification.items():
#         if node_type == 'parent_or_child':
#             if node.startswith('P'):
#                 G.add_edge(node, T)
#             else:
#                 G.add_edge(T, node)
    
#     # Draw the graph
#     pos = nx.spring_layout(G)
#     colors = [G.nodes[node]['color'] for node in G.nodes()]
#     plt.figure(figsize=(10, 8))
#     nx.draw(G, pos, node_color=colors, with_labels=True, 
#             node_size=1500, font_size=12, font_weight='bold')
#     plt.title('Discovered Network Structure')
#     plt.show()





def convert_names_to_indices(data: pd.DataFrame, name_mapping: Dict[str, List[str]]) -> Dict[int, List[int]]:
    """
    Convert a mapping of column names to a mapping of column indices.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataset with named columns.
    name_mapping : Dict[str, List[str]]
        A dictionary mapping column names to lists of their highly correlated column names.
    
    Returns
    -------
    Dict[int, List[int]]
        A dictionary mapping column indices to lists of their highly correlated column indices.
    """
    # Get column name to index mapping
    col_to_idx = {col: idx for idx, col in enumerate(data.columns)}
    
    # Convert name mapping to index mapping
    idx_mapping = {}
    for col_name, corr_cols in name_mapping.items():
        if col_name in col_to_idx:
            idx = col_to_idx[col_name]
            idx_mapping[idx] = [col_to_idx[c] for c in corr_cols if c in col_to_idx]
    
    return idx_mapping


def convert_indices_to_names(data: pd.DataFrame, idx_mapping: Dict[int, List[int]]) -> Dict[str, List[str]]:
    """
    Convert a mapping of column indices to a mapping of column names.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataset with named columns.
    idx_mapping : Dict[int, List[int]]
        A dictionary mapping column indices to lists of their highly correlated column indices.
    
    Returns
    -------
    Dict[str, List[str]]
        A dictionary mapping column names to lists of their highly correlated column names.
    """
    # Get index to column name mapping
    idx_to_col = {idx: col for idx, col in enumerate(data.columns)}
    
    # Convert index mapping to name mapping
    name_mapping = {}
    for idx, corr_idxs in idx_mapping.items():
        if idx in idx_to_col:
            col_name = idx_to_col[idx]
            name_mapping[col_name] = [idx_to_col[i] for i in corr_idxs if i in idx_to_col]
    
    return name_mapping


def find_correlated_components(data: pd.DataFrame, threshold: float = 0.95) -> List[List[int]]:
    """
    Find connected components of highly correlated features using correlation matrix.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    threshold : float, default=0.95
        Absolute correlation threshold to consider features as highly correlated.
    
    Returns
    -------
    List[List[int]]
        List of connected components where each component is a list of column indices.
    """
    n_features = data.shape[1]
    # Calculate correlation matrix
    corr_matrix = data.corr().abs().values
    
    # Create a graph for connected components
    G = nx.Graph()
    G.add_nodes_from(range(n_features))
    
    # Add edges for highly correlated pairs
    for i in range(n_features):
        for j in range(i+1, n_features):
            if corr_matrix[i, j] >= threshold:
                G.add_edge(i, j)
    
    # Find connected components (groups of correlated features)
    components = list(nx.connected_components(G))
    return [sorted(list(comp)) for comp in components if len(comp) > 1]

def select_representatives(components: List[List[int]]) -> Dict[int, List[int]]:
    """
    Select representative features for each connected component.
    
    Parameters
    ----------
    components : List[List[int]]
        List of connected components.
    
    Returns
    -------
    Dict[int, List[int]]
        Mapping from representative indices to lists of represented indices.
    """
    representatives = {}
    
    for component in components:
        # Select the first node as representative
        rep_idx = component[0]
        represented = [idx for idx in component if idx != rep_idx]
        representatives[rep_idx] = represented
    
    return representatives

def remove_highly_correlated_features(data: pd.DataFrame, high_corr_feature_groups: Dict[str, List[str]] = None, threshold: float = 0.95) -> Tuple[pd.DataFrame, Dict[int, List[int]], List[int]]:
    """
    Remove highly correlated features from the dataset using connected components.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataset with all features.
    high_corr_feature_groups : Dict[str, List[str]], optional
        A dictionary mapping column names to lists of their highly correlated column names.
        If provided, this will be used instead of computing correlations.
    threshold : float, default=0.95
        Correlation threshold to consider features as highly correlated.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, List[int]], List[int]]
        - The dataset with highly correlated features removed
        - The mapping with indices for the reduced dataset
        - List of original indices kept in the reduced dataset
    """
    if high_corr_feature_groups is not None:
        # Convert name-based mapping to index-based mapping
        idx_mapping = convert_names_to_indices(data, high_corr_feature_groups)
        
        # Identify all nodes to be removed
        nodes_to_remove = []
        for _, correlated_nodes in idx_mapping.items():
            nodes_to_remove.extend(correlated_nodes)
        
        # Create a list of nodes to keep
        all_nodes = list(range(data.shape[1]))
        nodes_to_keep = [node for node in all_nodes if node not in nodes_to_remove]
        
        # Create mapping from original indices to new indices in reduced dataset
        orig_to_reduced = {orig_idx: reduced_idx for reduced_idx, orig_idx in enumerate(nodes_to_keep)}
        
        # Create adjusted mapping for the reduced dataset
        adjusted_mapping = {}
        for source_node, correlated_nodes in idx_mapping.items():
            if source_node in nodes_to_keep:  # Only include if source node is kept
                adjusted_mapping[orig_to_reduced[source_node]] = correlated_nodes
        
        # Remove correlated features from the dataset
        reduced_data = data.iloc[:, nodes_to_keep]
        
        return reduced_data, adjusted_mapping, nodes_to_keep
    else:
        # Find connected components based on correlation
        components = find_correlated_components(data, threshold)
        
        # If no correlated components found, return original data
        if not components:
            return data, {}, list(range(data.shape[1]))
        
        # Select representatives for each component
        rep_dict = select_representatives(components)
        
        # Identify all nodes to be removed
        nodes_to_remove = []
        for _, represented_idxs in rep_dict.items():
            nodes_to_remove.extend(represented_idxs)
        
        # Create a list of nodes to keep
        all_nodes = list(range(data.shape[1]))
        nodes_to_keep = [node for node in all_nodes if node not in nodes_to_remove]
        
        # Create mapping from original indices to new indices in reduced dataset
        orig_to_reduced = {orig_idx: reduced_idx for reduced_idx, orig_idx in enumerate(nodes_to_keep)}
        
        # Create adjusted mapping for the reduced dataset
        adjusted_mapping = {}
        for rep_idx, represented_idxs in rep_dict.items():
            if rep_idx in nodes_to_keep:  # Only include if representative node is kept
                adjusted_mapping[orig_to_reduced[rep_idx]] = represented_idxs
        
        # Remove correlated features from the dataset
        reduced_data = data.iloc[:, nodes_to_keep]
        
        return reduced_data, adjusted_mapping, nodes_to_keep

def add_correlated_nodes_to_graph(graph: np.ndarray, original_indices: List[int], correlated_nodes_map: Dict[str, List[str]] = None, data: pd.DataFrame = None, threshold: float = 0.95) -> np.ndarray:
    """
    Add highly correlated nodes back to the graph after causal discovery.
    
    Parameters
    ----------
    graph : np.ndarray
        The adjacency matrix representing the causal graph.
        In this graph, a value of 1 at position [j, i] means that i causes j (i → j).
    correlated_nodes_map : Dict[str, List[str]], optional
        A dictionary mapping column names to lists of their highly correlated column names.
        If provided, this will be used instead of computing correlations.
    data : pd.DataFrame
        The original dataset with all features.
    threshold : float, default=0.95
        Correlation threshold to consider features as highly correlated.
    original_indices : List[int], optional
        List of original indices that were kept in the reduced dataset. If provided,
        this ensures the graph uses the correct original indices.
    
    Returns
    -------
    np.ndarray
        The expanded adjacency matrix including the correlated nodes.
    """
    if data is None:
        return graph
    
    # Check if we're dealing with a lagged graph (3D) or standard graph (2D)
    is_lagged_graph = len(graph.shape) == 3
    
    # Original dataset size
    orig_size = data.shape[1]
    
    # If using provided correlated nodes map, convert from names to indices
    idx_mapping = {}
    if correlated_nodes_map is not None:
        idx_mapping = convert_names_to_indices(data, correlated_nodes_map)
    else:
        # Find connected components based on correlation
        components = find_correlated_components(data, threshold)
        # Select representatives
        idx_mapping = select_representatives(components)
    
    if not idx_mapping:
        return graph
    
    if is_lagged_graph:
        # For lagged graphs (L, N, N)
        L, n, _ = graph.shape
        
        # Create expanded graph with space for all original nodes
        expanded_graph = np.zeros((L, orig_size, orig_size), dtype=graph.dtype)
        
        for i in range(n):
            for j in range(n):
                for lag in range(L):
                    if graph[lag, i, j] != 0:
                        expanded_graph[lag, original_indices[i], original_indices[j]] = graph[lag, i, j]
       
        # Add correlated nodes with the same relationships
        for rep_idx, corr_idxs in idx_mapping.items():
            # Skip if rep_idx is not in the graph (might happen with original_indices)
            rep_reduced_idx = original_indices.index(rep_idx)
                
            for corr_idx in corr_idxs:
                # For each lag layer
                for lag in range(L):
                    # Copy incoming edges (where rep_idx is the target/effect)
                    # In graph[j,i]=1, i is the cause and j is the effect
                    for i in range(n):
                        i_orig = original_indices[i] if original_indices is not None else i
                        if i_orig != rep_idx and graph[lag, rep_reduced_idx, i] != 0:
                            # If i causes rep_idx, then i should also cause corr_idx
                            expanded_graph[lag, corr_idx, i_orig] = graph[lag, rep_reduced_idx, i]
                    
                    # Copy outgoing edges (where rep_idx is the source/cause)
                    for j in range(n):
                        j_orig = original_indices[j] if original_indices is not None else j
                        if j_orig != rep_idx and graph[lag, j, rep_reduced_idx] != 0:
                            expanded_graph[lag, j_orig, corr_idx] = graph[lag, j, rep_reduced_idx]
                
                # Add correlation edge between rep and correlated node in lag 0 without symmetry
                expanded_graph[0, rep_idx, corr_idx] = 7  # Undirected edge (--) for correlation
    else:
        # For standard graphs (N, N)
        n = graph.shape[0]
        
        # Create expanded graph with space for all original nodes
        expanded_graph = np.zeros((orig_size, orig_size), dtype=graph.dtype)
        
        # Map from reduced indices to original indices
        for i in range(n):
            for j in range(n):
                if graph[i, j] != 0:
                    expanded_graph[original_indices[i], original_indices[j]] = graph[i, j]
        
        # Add correlated nodes with the same relationships
        for rep_idx, corr_idxs in idx_mapping.items():
            # print(f"Rep idx: {rep_idx}, Corr idxs: {corr_idxs}")
            # Skip if rep_idx is not in the graph (might happen with original_indices)
            rep_reduced_idx = None
            try:
                rep_reduced_idx = original_indices.index(rep_idx)
            except ValueError:
                continue
                
            for corr_idx in corr_idxs:
                # Copy incoming edges (where rep_idx is the target/effect)
                # In graph[j,i]=1, i is the cause and j is the effect
                for i in range(n):
                    i_orig = original_indices[i]
                    if i_orig != rep_idx and graph[rep_reduced_idx, i] != 0:
                        # If i causes rep_idx, then i should also cause corr_idx
                        expanded_graph[corr_idx, i_orig] = graph[rep_reduced_idx, i]
                
                # Copy outgoing edges (where rep_idx is the source/cause)
                for j in range(n):
                    j_orig = original_indices[j]
                    if j_orig != rep_idx and graph[j, rep_reduced_idx] != 0:
                        # If rep_idx causes j, then corr_idx should also cause j
                        expanded_graph[j_orig, corr_idx] = graph[j, rep_reduced_idx]
                
                # Add correlation edge between rep and correlated node without symmetry
                expanded_graph[rep_idx, corr_idx] = 7  # Undirected edge (--) for correlation
    
    return expanded_graph

def restore_original_node_indices(reduced_graph: np.ndarray, original_indices: List[int], correlated_nodes_map: Dict[int, List[int]]) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Restore the original node indices in the graph and correlated nodes map.
    
    Parameters
    ----------
    reduced_graph : np.ndarray
        The graph adjacency matrix with reduced indices.
    original_indices : List[int]
        The list of original indices that were kept in the reduced dataset.
    correlated_nodes_map : Dict[int, List[int]]
        A dictionary mapping reduced node indices to lists of original indices of their highly correlated nodes.
    
    Returns
    -------
    Tuple[np.ndarray, Dict[int, List[int]]]
        - The graph with original node indices
        - The correlated nodes map with original indices
    """
    if not original_indices:
        return reduced_graph, correlated_nodes_map
    
    # Create mapping from reduced indices to original indices
    reduced_to_orig = {i: original_indices[i] for i in range(len(original_indices))}
    
    # Create corrected mapping with original indices
    corrected_map = {}
    for reduced_idx, corr_nodes in correlated_nodes_map.items():
        if reduced_idx < len(original_indices):
            orig_idx = reduced_to_orig[reduced_idx]
            corrected_map[orig_idx] = corr_nodes
    
    return reduced_graph, corrected_map

def evaluate_graph_accuracy(predicted_graph: np.ndarray, true_graph: np.ndarray, skip_correlated_edges: bool = True) -> dict:
    """
    Calculate precision, recall, and F1 score between predicted and true graphs.
    
    Parameters
    ----------
    predicted_graph : np.ndarray
        The predicted adjacency matrix.
    true_graph : np.ndarray
        The true adjacency matrix.
    skip_correlated_edges : bool, default=True
        Whether to skip correlation edges (type 2) in evaluation.
    
    Returns
    -------
    dict
        Dictionary with precision, recall, and F1 score.
    """
    # Count true positives, false positives, and false negatives
    tp = 0  # Correctly predicted edges
    fp = 0  # Predicted edges that don't exist in true graph
    fn = 0  # True edges that weren't predicted
    
    for i in range(true_graph.shape[0]):
        for j in range(true_graph.shape[1]):
            if i == j:
                continue  # Skip self-loops
            
            # Skip correlation edges if specified
            if skip_correlated_edges and (true_graph[i, j] == 2 or predicted_graph[i, j] == 2):
                continue
                
            # Count based on causal edges (value 1)
            if predicted_graph[i, j] == 1 and true_graph[i, j] == 1:
                tp += 1
            elif predicted_graph[i, j] == 1 and true_graph[i, j] != 1:
                fp += 1
            elif predicted_graph[i, j] != 1 and true_graph[i, j] == 1:
                fn += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
def test_correlated_components_pipeline():
    """
    Test the connected components approach for handling correlated features.
    """
    print("\n=== TESTING CONNECTED COMPONENTS PIPELINE ===")
    
    # Create synthetic dataset with transitive correlations
    np.random.seed(42)
    n_samples = 200
    
    # Independent variables
    X1 = np.random.normal(0, 1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)
    X6 = np.random.normal(0, 1, n_samples)
    
    # Create transitively correlated groups:
    # Group 1: X1, X2, X7 (X1 correlates with X2, X2 correlates with X7)
    X2 = 0.3 * X1 + np.random.normal(0, 1, n_samples)  # Correlated with X1
    X7 = 0.9 * X2 + np.random.normal(0, 1, n_samples)  # Correlated with X2, and transitively with X1
    
    # Group 2: X3, X4, X5 
    X4 = X3 * -0.4 + np.random.normal(0, 1, n_samples)  # Correlated with X3
    X5 = X4 * -1.5 + np.random.normal(0, 1, n_samples)  # Correlated with X4, and transitively with X3
    
    # Create causal relationships
    Y = 0.7 * X1 + 0.5 * X3 + 0.3 * X6 + 0.1 * np.random.normal(0, 1, n_samples)
    
    # Create dataframe
    data = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6, 'X7': X7, 'Y': Y
    })
    
    # Define the true graph structure
    true_graph = np.zeros((8, 8))
    
    # Add true causal edges: X1->Y, X3->Y, X6->Y
    col_idx = {name: i for i, name in enumerate(data.columns)}
    true_graph[col_idx['X2'], col_idx['X1']] = 1
    true_graph[col_idx['X7'], col_idx['X2']] = 1
    true_graph[col_idx['X4'], col_idx['X3']] = 1
    true_graph[col_idx['X5'], col_idx['X4']] = 1
    true_graph[col_idx['Y'], col_idx['X1']] = 1
    true_graph[col_idx['Y'], col_idx['X3']] = 1
    true_graph[col_idx['Y'], col_idx['X6']] = 1

    print(f"True graph: {true_graph}")
    
    # Step 1: Find connected components
    threshold = 0.95
    components = find_correlated_components(data, threshold)
    print(f"\nConnected Components (threshold={threshold}):")
    for i, comp in enumerate(components):
        print(f"Component {i+1}: {[data.columns[idx] for idx in comp]}")
    
    # Step 2: Remove highly correlated features
    reduced_data, adjusted_mapping, original_indices = remove_highly_correlated_features(data, threshold=threshold)
    print("\nReduced Dataset Columns:")
    print(reduced_data.columns.tolist())
    
    # Step 3: Run PC algorithm on the reduced dataset
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pc import PC
    from evaluation.evaluator import GraphEvaluator
    
    # Initialize PC algorithm with default parameters
    pc_algo = PC({
        'alpha': 0.05,
        'indep_test': 'fisherz',
        'depth': 4,
        'stable': True,
        'show_progress': False
    })
    
    # Run PC algorithm on the reduced dataset
    reduced_graph, info, _ = pc_algo.fit(reduced_data)
    
    print("\nPC Algorithm Results on Reduced Dataset:")
    print(f"Graph shape: {reduced_graph.shape}")
    print(f"Graph: {reduced_graph}")
    
    # Display the discovered edges
    reduced_cols = reduced_data.columns.tolist()
    print(f"Reduced cols: {reduced_cols}")
    for i in range(reduced_graph.shape[0]):
        for j in range(reduced_graph.shape[1]):
            if reduced_graph[i, j] == 1:  # Directed edge
                print(f"Discovered edge: {reduced_cols[j]} -> {reduced_cols[i]}")
    
    # Step 4: Add correlated nodes back to the graph
    expanded_graph = add_correlated_nodes_to_graph(reduced_graph, original_indices=original_indices, data=data, threshold=threshold)

    print(f"Expanded graph: {expanded_graph}")
    
    # Evaluate accuracy against the true graph
    evaluator = GraphEvaluator()
    metrics = evaluator.compute_metrics(true_graph, expanded_graph)
    
    print("\n=== Graph Accuracy Metrics (excluding correlation edges) ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    test_correlated_components_pipeline()

