import numpy as np
import pandas as pd
from typing import Dict, Tuple
import torch
# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)
sys.path.insert(0, causal_learn_dir)

from causallearn.search.ScoreBased.CALM import calm
from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm

class CALM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'lambda1': 0.005,  # Coefficient for L0 penalty
            'alpha': 0.01,     # Significance level for moral graph estimation
            'tau': 0.5,        # Temperature for Gumbel-Sigmoid
            'rho_init': 1e-5,  # Initial penalty parameter
            'rho_mult': 3,     # Multiplication factor for rho
            'htol': 1e-8,      # Tolerance for acyclicity
            'subproblem_iter': 40000,  # Number of iterations for subproblem
            'standardize': True,  # Whether to standardize data
            'device': 'auto'     # Device for computation
        }
        self._params.update(params)
        # Automatically decide device_type if set to 'auto'
        if self._params.get('device', 'cpu') == 'auto':
            self._params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def name(self):
        return "CALM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['lambda1', 'alpha', 'subproblem_iter']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = [ 'tau', 'rho_init', 'rho_mult', 'htol', 'subproblem_iter', 
                                    'standardize', 'device']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        # Convert data to numpy array
        X = data.values

        # Run CALM algorithm
        record = calm(X, **self._params)

        # Extract adjacency matrix from the GeneralGraph
        adj_matrix = self.convert_to_adjacency_matrix(record['G'])

        # Prepare additional information
        info = {
            'B_weighted': record['B_weighted']
        }

        return adj_matrix, info, record['G']

    def convert_to_adjacency_matrix(self, G) -> np.ndarray:
        """Convert GeneralGraph to adjacency matrix format."""
        d = len(G.get_nodes())
        adj_matrix = np.zeros((d, d))
        
        for i in range(d):
            for j in range(d):
                if G.graph[i, j] == 1 and G.graph[j, i] == -1:  
                    # If there's a directed edge j -> i
                    adj_matrix[i, j] = 1

        return adj_matrix

    def test_algorithm(self):
        """Test the CALM algorithm with synthetic data."""
        # Generate sample data with linear relationships
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing CALM algorithm with pandas DataFrame:")
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("\nWeighted Adjacency Matrix:")
        print(info['B_weighted'])

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Use GraphEvaluator to compute metrics
        from causal_discovery.evaluation.evaluator import GraphEvaluator
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")

if __name__ == "__main__":
    calm_algo = CALM({})
    calm_algo.test_algorithm() 