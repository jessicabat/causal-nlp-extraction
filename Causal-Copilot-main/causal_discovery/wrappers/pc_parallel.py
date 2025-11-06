import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator
from castle.algorithms import PC as pc_parallel


class PCParallel(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'cores': 8,
            'memory_efficient': False,
            'background_knowledge': None,
            'batch': None
        }
        self._params.update(params)

    @property
    def name(self):
        return "PCParallel"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'cores']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['memory_efficient', 'background_knowledge', 'batch']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data = np.array(data)
        
        # Initialize the parallel PC
        model = pc_parallel(
            variant='parallel',
            alpha=self._params['alpha'],
            ci_test=self._params['indep_test'],
            priori_knowledge=self._params['background_knowledge']
        )
        
        # Initialize the number of cores to be used
        # No significant speedup past 16 cores
        cores = self._params['cores'] if self._params['cores'] < 16 else min(os.cpu_count(), 14)
        
        # Fit the model
        model.learn(
            data,
            columns=node_names,
            p_cores=cores,
            s=self._params['memory_efficient'],
            batch=self._params['batch']
        )

        # Get the adjacency matrix
        adj_matrix = model.causal_matrix
        if isinstance(adj_matrix, pd.DataFrame):
            adj_matrix = adj_matrix.values
            
        # Use the transpose st if i->j adj[j,i] = 1
        adj_matrix = self.convert_to_cpdag(adj_matrix.T)
        
        # Prepare additional information
        info = {
            'nodes': node_names,
            'cores': cores
        }
        
        return adj_matrix, info, model
    
    def convert_to_cpdag(self, adj: np.ndarray) -> np.ndarray:
        adj_matrix = adj
        
        for i in range(len(adj_matrix)):
            if adj_matrix[i, i] == 1:
                adj_matrix[i, i] = 0
                
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i,j] == 1 and adj_matrix[j,i] == 1:
                    adj_matrix[i,j] = 2
                    adj_matrix[j,i] = 0
        
        return adj_matrix

    def test_algorithm(self):
        # Generate sample data with linear relationships
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing PC-Parallel algorithm with pandas DataFrame:")
       
        adj_matrix, info, _ = self.fit(df)
        
        print("Adjacency Matrix:")
        print(adj_matrix)

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
    params = {
        'alpha': 0.05,
        'indep_test': 'fisherz',
        'cores': 8,
        'memory_efficient': True
    }
    pc_algo = PCParallel(params)
    pc_algo.test_algorithm() 