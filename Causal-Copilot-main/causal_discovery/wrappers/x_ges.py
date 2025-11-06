import numpy as np
import pandas as pd
from typing import Dict, Tuple
import time

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(algorithm_dir)
sys.path.append(root_dir)

from xges import XGES as xges

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator
from data.simulator.dummy import DataSimulator

class XGES(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 2.0
        }
        self._params.update(params)

    @property
    def name(self):
        return "XGES"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values

        # Run XGES algorithm
        model = xges(**self.get_primary_params())
        result = model.fit(data_values)

        # Convert the graph to adjacency matrix
        adj_pdag = result.to_adjacency_matrix()
        adj_matrix = self.convert_to_adjacency_matrix(adj_pdag)

        # Prepare additional information
        info = {
            # 'score': result['score']
        }

        return adj_matrix, info, result

    def convert_to_adjacency_matrix(self, G: np.ndarray) -> np.ndarray:
        n = G.shape[0]
        custom_matrix = np.zeros_like(G, dtype=int)

        indices = np.where(G == 1)

        for i, j in zip(indices[0], indices[1]):
            if G[j, i] == 1:
                if i > j:
                    custom_matrix[i, j] = 2
            else:
                custom_matrix[j, i] = 1

        return custom_matrix

    def test_algorithm(self):
        # Import the DataSimulator from the benchmark module
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from data.simulator.dummy import DataSimulator
        
        # Create a simulator instance
        simulator = DataSimulator()
        
        # Generate sample data with MLP (non-linear) relationships
        np.random.seed(42)
        n_nodes = 20
        n_samples = 3000
        
        # Generate synthetic dataset with MLP relationships similar to benchmark.py
        gt_graph, df = simulator.generate_dataset(
            n_nodes=n_nodes,
            n_samples=n_samples,
            edge_probability=0.5,
            function_type="mlp",  # Using MLP for non-linear relationships
            noise_type="gaussian",
            noise_scale=1.0,
            discrete_ratio=0.0,  # All continuous variables
            n_domains=1
        )
        
        # Extract the data and ground truth graph
        df = pd.DataFrame(df)
            
        print(f"Testing XGES algorithm with MLP data ({n_nodes} nodes, {n_samples} samples):")
        params = {
            'alpha': 2.0
        }
        
        # Set parameters and fit the model
        self._params = params
        adj_matrix, info, _ = self.fit(df)
        
        # print("Adjacency Matrix (subset):")
        # print(gt_graph)  # Print just a subset for readability
        
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
        
        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']}")
        print(f"Best Graph (subset):\n{adj_matrix}")

if __name__ == "__main__":
    xges_algo = XGES({})
    xges_algo.test_algorithm()