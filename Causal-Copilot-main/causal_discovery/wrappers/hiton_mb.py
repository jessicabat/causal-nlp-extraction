import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Use local pyCausalFS package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pycausalfs_dir = os.path.join(root_dir, 'externals', 'pyCausalFS')
if not os.path.exists(pycausalfs_dir):
    raise FileNotFoundError(f"Local pyCausalFS directory not found: {pycausalfs_dir}")
sys.path.append(pycausalfs_dir)
sys.path.append(root_dir)


from CBD.MBs.HITON.HITON_MB import HITON_MB
from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator
from causal_discovery.wrappers.utils.tab_utils import MB2CPDAG

class HITONMB(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'n_jobs': 4,
            'n_subjobs': 1,
        }
        self._params.update(params)

    @property
    def name(self):
        return "HITONMB"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['n_jobs', 'n_subjobs']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Fit the HITON_MB algorithm to data.
        
        Args:
            data: pandas DataFrame containing the data
            
        Returns:
            Tuple containing:
            - adj_matrix: numpy array representing the adjacency matrix
            - info: dictionary with additional information
        """
        from joblib import Parallel, delayed
        n_vars = data.shape[1]
        mb_dict = {}  # Dictionary to store MB results
        
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        params = {**self.get_primary_params(), **self.get_secondary_params()}
        total_ci_tests = 0
        
        results = Parallel(n_jobs=params['n_jobs'])(
            delayed(HITON_MB)(
                data=data.values,
                target=target,
                alpha=params['alpha'],
                indep_test=params['indep_test'],
                n_jobs=params['n_jobs']
            ) for target in range(n_vars)
        )
        for target, (mb, ci_num) in enumerate(results):
            mb_dict[target] = list(mb)
            total_ci_tests += ci_num

        # Convert MB results to CPDAG
        adj_matrix = MB2CPDAG(
            data=data,
            mb_dict=mb_dict,
            indep_test=params['indep_test'],
            alpha=params['alpha'],
            n_jobs=params['n_subjobs']
        )

        info = {
            'total_ci_tests': total_ci_tests,
            'mb_dict': mb_dict
        }

        return adj_matrix, info, mb_dict

    def test_algorithm(self):
        """Run a simple test of the algorithm on synthetic data."""
        # Generate sample data with linear relationships
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing HITON-MB algorithm with synthetic data:")
        adj_matrix, info, mb_dict = self.fit(df)
        
        print("\nAdjacency Matrix:")
        print(adj_matrix)
        print("\nAdditional Info:")
        print(f"Total CI tests: {info['total_ci_tests']}")

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")

if __name__ == "__main__":
    hiton_mb = HITONMB()
    hiton_mb.test_algorithm() 