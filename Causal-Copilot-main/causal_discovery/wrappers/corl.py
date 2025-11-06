import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import torch
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator
from castle.algorithms.gradient.corl.torch import CORL as TrustAI_CORL


class CORL(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'batch_size': 64,
            'input_dim': 100,
            'embed_dim': 256,
            'normalize': False,
            'encoder_name': 'transformer',
            'encoder_heads': 2,  # default: 8
            'encoder_blocks': 3,  # default: 3
            'encoder_dropout_rate': 0.1,
            'decoder_name': 'lstm',
            'reward_mode': 'episodic',
            'reward_score_type': 'BIC',
            'reward_regression_type': 'LR',
            'reward_gpr_alpha': 1.0,
            'iteration': 500, # default: 5000
            'lambda_iter_num': 500,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'alpha': 0.99,
            'init_baseline': -1.0,
            'random_seed': 0,
            'device_type': 'auto',
            'device_ids': 0
        }
        self._params.update(params)
        # Automatically decide device_type if set to 'auto'
        if self._params.get('device_type', 'cpu') == 'auto':
            self._params['device_type'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    @property
    def name(self):
        return "CORL"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['iteration']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['batch_size', 'embed_dim', 'reward_mode', 'reward_score_type', 
                                      'actor_lr', 'critic_lr', 'input_dim', 'normalize', 
                                      'encoder_name', 'encoder_heads', 'encoder_blocks', 
                                      'encoder_dropout_rate', 'decoder_name', 
                                      'reward_regression_type', 'reward_gpr_alpha',
                                      'lambda_iter_num', 'alpha', 'init_baseline', 
                                      'random_seed', 'device_type', 'device_ids']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        # Check and remove domain_index if it exists
        if isinstance(data, pd.DataFrame) and 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
        
        # Initialize CORL algorithm
        corl = TrustAI_CORL(**self._params)
        
        # Fit the model
        corl.learn(data)
        
        # Get the causal matrix
        causal_matrix = np.array(corl.causal_matrix).T
        
        # Prepare additional information
        info = {
            'device_type': self._params['device_type'],
            'iterations': self._params['iteration'],
            'reward_mode': self._params['reward_mode']
        }
        
        return causal_matrix, info

    def test_algorithm(self):
        """Test the CORL algorithm with synthetic data"""
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        # Generate DAG structure
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        data = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5
        })

        print("Testing CORL algorithm with synthetic data:")
        
        # Create instance with test parameters
        corl_test = CORL()
        
        # Fit the model
        adj_matrix, info = corl_test.fit(data)
        
        print("\nInferred Adjacency Matrix:")
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
    corl = CORL()
    corl.test_algorithm() 