from typing import Dict, Tuple
import numpy as np
import pandas as pd

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')

sys.path.append(root_dir)
sys.path.append(algorithm_dir)
sys.path.insert(0, causal_learn_dir)

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.wrappers.pc import PC
from causal_discovery.wrappers.ges import GES
from causallearn.graph.GraphClass import CausalGraph, GeneralGraph
from causallearn.utils.PCUtils.Meek import meek
from causal_discovery.evaluation.evaluator import GraphEvaluator
from causallearn.search.FCMBased.ANM.ANM import ANM
from causallearn.search.FCMBased.PNL.PNL import PNL

class Hybrid(CausalDiscoveryAlgorithm):
    """
    A hybrid approach that combines constraint-based methods (e.g., PC, GES) with
    functional model-based methods (e.g., ANM, PNL). The final output is a
    maximally oriented DAG/CPDAG by applying a second-stage pairwise orientation
    procedure on top of the initial Markov equivalence class.
    """

    def __init__(self, params: Dict = {}):
        """
        Parameters
        ----------
        params : dict
            'first_stage_algo': the name of the first stage algorithm to generate a CPDAG (e.g., 'pc', 'ges').
            'second_stage_method': the name of the functional model-based method (e.g., 'anm', 'pnl').
            'alpha': significance level for the functional test.
            'm_max': max number of potential confounders to consider.
            Additional algorithm-specific parameters may be included.
        """
        super().__init__(params)
        self._params = {
            'first_stage_algo': 'pc',
            'second_stage_method': 'pnl',
            'alpha': 0.05,
            'm_max': 3,
        }
        self._params.update(params)

    @property
    def name(self):
        return "Hybrid"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        # define your primary parameter keys as needed
        self._primary_param_keys = ['first_stage_algo', 'second_stage_method', 'alpha', 'm_max']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        # define any additional parameter keys if needed
        self._secondary_param_keys = []
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        """
        Fit the Hybrid method on the data.

        Step 1: Use a constraint-based method (PC/GES) to obtain a CPDAG.
        Step 2: For each undirected edge in the CPDAG, attempt to orient the edge
                using the functional model-based method (e.g., ANM).
        Step 3: Apply Meek's rule to further propagate orientations.

        Returns
        -------
        adj_matrix : np.ndarray
            The adjacency matrix of the final oriented graph.
        info : Dict
            Additional info as needed (e.g., the CPDAG before second-stage orientation).
        cg : CausalGraph
            The final oriented CausalGraph object.
        """
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values
        first_stage_algo = self._params.get('first_stage_algo', 'pc')
        second_stage_method = self._params.get('second_stage_method', 'pnl')
        alpha = self._params.get('alpha', 0.05)
        m_max = self._params.get('m_max', 1)

        # 1) Obtain an initial CPDAG using PC or GES wrapper
        if first_stage_algo.lower() == 'pc':
            stage1_algo = PC({'alpha': alpha})
            adj_matrix_stage1, info_stage1, cg_stage1 = stage1_algo.fit(data)
        elif first_stage_algo.lower() == 'ges':
            stage1_algo = GES({'alpha': alpha})
            adj_matrix_stage1, info_stage1, _cg_stage1 = stage1_algo.fit(data)
            cg_stage1 = CausalGraph(len(node_names), node_names)
            cg_stage1.G = _cg_stage1['G']
        else:
            raise ValueError(f"Unknown first stage algorithm: {first_stage_algo}")

        # Skip second stage if second_stage_method is None
        if second_stage_method is None:
            final_cg = cg_stage1.G
            adj_matrix = adj_matrix_stage1
            info = {
                'initial_cpdag': None,
                'adj_matrix_final': adj_matrix
            }
            return adj_matrix, info, final_cg

        # 2) For each undirected edge in the CPDAG, we attempt to orient using a
        # functional model-based method. We'll assume there's a function cause_or_effect
        # that returns (pval_for, pval_back).

        cg_stage1.to_nx_skeleton()
        G_undirected = cg_stage1.nx_skel

        def get_pi_qi(cg_stage1, i, j):
            neigh_i = cg_stage1.neighbors(i)
            pi = []  # confirmed confounders that are also i's neighbors
            qi = []  # potential confounders that are also i's neighbors
            for nn in neigh_i:
                if nn == j:
                    continue
                # Check if nn is a confirmed confounder by checking if it has directed edges to both i and j
                if (cg_stage1.G.is_directed_from_to(cg_stage1.G.nodes[nn], cg_stage1.G.nodes[i]) and 
                    cg_stage1.G.is_directed_from_to(cg_stage1.G.nodes[nn], cg_stage1.G.nodes[j])):
                    pi.append(nn)
                else:
                    # Check if nn is a confirmed collider (both i and j point to nn)
                    is_collider = (cg_stage1.G.is_directed_from_to(cg_stage1.G.nodes[i], cg_stage1.G.nodes[nn]) and
                                 cg_stage1.G.is_directed_from_to(cg_stage1.G.nodes[j], cg_stage1.G.nodes[nn]))
                    
                    # Only add as potential confounder if not a collider
                    if not is_collider:
                        qi.append(nn)
            return pi, qi

        def cause_or_effect(i, j, confounders, data_arr):
            # Extract data for variables i and j
            data_i = data_arr[:, i].reshape(-1, 1)
            data_j = data_arr[:, j].reshape(-1, 1)
            
            # Initialize the functional model based method
            if second_stage_method.lower() == 'anm':
                model = ANM()
            elif second_stage_method.lower() == 'pnl':
                model = PNL()
            else:
                raise ValueError(f"Unknown second stage method: {second_stage_method}")
                
            # Get p-values for both directions
            pval_for, pval_back = model.cause_or_effect(data_i, data_j, data_arr[:, confounders])
            return pval_for, pval_back

        from itertools import combinations

        has_change = True
        while has_change:
            has_change = False
            edges_to_check = []
            for i in range(len(node_names)):
                for j in range(i+1, len(node_names)):
                    if cg_stage1.G.is_undirected_from_to(cg_stage1.G.nodes[i], cg_stage1.G.nodes[j]):
                        edges_to_check.append((i, j))

            for (i, j) in edges_to_check:
                pi, qi = get_pi_qi(cg_stage1, i, j)
                oriented = False
                for m in range(m_max + 1):
                    if oriented:
                        break
                    for cand_conf in combinations(qi, m):
                        full_conf = list(set(pi).union(set(cand_conf)))
                        pval_for, pval_back = cause_or_effect(i, j, full_conf, data_values)
                        if pval_for > alpha and pval_for > pval_back:
                            edge1 = cg_stage1.G.get_edge(cg_stage1.G.nodes[i], cg_stage1.G.nodes[j])
                            if edge1 is not None:
                                cg_stage1.G.remove_edge(edge1)
                            print(f"Adding directed edge {i} -> {j}")
                            cg_stage1.G.add_directed_edge(cg_stage1.G.nodes[i], cg_stage1.G.nodes[j])
                            meek(cg_stage1)
                            oriented = True
                            has_change = True
                            break
                        elif pval_back > alpha and pval_back > pval_for:
                            edge1 = cg_stage1.G.get_edge(cg_stage1.G.nodes[i], cg_stage1.G.nodes[j])
                            if edge1 is not None:
                                cg_stage1.G.remove_edge(edge1)
                            print(f"Adding directed edge {j} -> {i}")
                            cg_stage1.G.add_directed_edge(cg_stage1.G.nodes[j], cg_stage1.G.nodes[i])
                            meek(cg_stage1)
                            oriented = True
                            has_change = True
                            break

        cg_stage1.to_nx_skeleton()
        final_cg = cg_stage1
        adj_matrix = self.convert_final_to_adjacency_matrix(final_cg)

        info = {
            'initial_cpdag': None,
            'adj_matrix_final': adj_matrix
        }
        return adj_matrix, info, final_cg

    def convert_final_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = cg.G.graph
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # directed edge: j -> i
                inferred_flat[i, j] = 1
            elif adj_matrix[j, i] == 1:
                # bidirected edge: j <-> i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 3

        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # undirected edge: j -- i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 2
        return inferred_flat

    def test_algorithm(self):
        """
        Test the Hybrid algorithm using randomly generated data (linear relationships
        in this example), then compute metrics to demonstrate functionality.
        """
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)

        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})
        print("Testing Hybrid algorithm with pandas DataFrame:")

        # First run with only first stage algorithm
        params = {
            'first_stage_algo': 'pc',
            'alpha': 0.05,
            'm_max': 1,
            'second_stage_method': None  # Disable second stage
        }
        self._params.update(params)
        adj_matrix_first, info_first, _ = self.fit(df)

        print("First Stage Only Adjacency Matrix:")
        print(adj_matrix_first)

        # Then run with both stages
        params['second_stage_method'] = 'pnl'
        self._params.update(params)
        adj_matrix, info, _ = self.fit(df)

        print("\nFull Hybrid Adjacency Matrix:")
        print(adj_matrix)

        # Define a simple ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Evaluate both results
        evaluator = GraphEvaluator()
        
        print("\nFirst Stage Only Metrics:")
        metrics_first = evaluator.compute_metrics(gt_graph, adj_matrix_first)
        print(f"F1 Score: {metrics_first['f1']:.4f}")
        print(f"Precision: {metrics_first['precision']:.4f}")
        print(f"Recall: {metrics_first['recall']:.4f}")
        print(f"SHD: {metrics_first['shd']:.4f}")

        print("\nFull Hybrid Metrics:")
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")



if __name__ == "__main__":
    hybrid = Hybrid({})
    hybrid.test_algorithm()
