import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import sys
import os
from sklearn import preprocessing
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator
from causal_discovery.wrappers.utils.ts_utils import generate_stationary_linear
from externals.notears.model import MODEL_NTS_NOTEARS, train_NTS_NOTEARS
class NTSNOTEARS(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'p': 10,  # number of lags
            'lambda1': 0.005,  # lambda for convolutional parameters in each time step. Can be float or list [lag2, lag1, instantaneous]
            'lambda2': 0.01,  # lambda for all parameters
            'w_threshold': 0.3,  # threshold for edge weights. Can be float or list [lag2, lag1, instantaneous]
            'max_iter': 100,
            'h_tol': 1e-8,
            'dims_conv': 10,  # hidden dimension size
            'device': 'auto',  # Device type ('cpu', 'cuda', or 'auto')
        }
        self._params.update(params)
        # Automatically decide device if set to 'auto'
        if self._params.get('device', 'cpu') == 'auto':
            try:
                import torch
                self._params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self._params['device'] = 'cpu'

    @property
    def name(self):
        return "NTSNOTEARS"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['lambda1', 'lambda2', 'w_threshold']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['h_tol', 'max_iter', 'device']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.to_numpy()
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data = np.array(data)
        print("p =", self._params['p'], "type =", type(self._params['p']))
        scaler = preprocessing.StandardScaler().fit(data)
        data_normalized = scaler.transform(data)
        max_lag = self._params['p']
        n_nodes = data.shape[1]
        # init NTS_NOTEARS model
        model = MODEL_NTS_NOTEARS(
            [n_nodes, self._params['dims_conv'], 1], 
            bias=True, 
            number_of_lags=max_lag)

        output = train_NTS_NOTEARS(model, 
            data_normalized,  
            **self.get_primary_params(), **self.get_secondary_params(),
            verbose=0)

        prediction = output[:,-n_nodes:]
        # prediction is defined as: row variable -> column variable
        # row variables and column variables are the same, in the order of:
        # ..., x1_{t-2}, x2_{t-2}, ..., x1_{t-1}, x2_{t-1}, ..., x1_{t}, x2_{t}, ...
        
        # convert prediction to adj matrix
        lag_matrix = np.zeros((max_lag + 1, n_nodes, n_nodes))
        for cause in range(n_nodes * (max_lag + 1)):
            cause_d = cause % n_nodes
            cause_t = max_lag - cause // n_nodes
            for effect in range(n_nodes):
                if prediction[cause, effect]:
                    lag_matrix[cause_t, effect, cause_d] = 1 #prediction[cause, effect]

        summary_matrix = np.any(lag_matrix, axis=0).astype(int)
        
        info = {
            'lag_matrix': lag_matrix,
            'lag': max_lag,
            'nodes': node_names,
            'model': model
        }

        return summary_matrix, info, output

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        n_nodes = 3
        lag = 2
        
        df, gt_graph, gt_summary, graph_net = generate_stationary_linear(
            n_nodes,
            n_samples,
            lag,
            degree_intra=1,
            degree_inter=2,
        )
        print("Testing NTSNOTEARS algorithm with pandas DataFrame:")
        print("Ground truth graph\n", gt_graph)
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        # Run the algorithm
        for _ in range(1):
            adj_matrix, _, _ = self.fit(df)
            print("Prediction\n", adj_matrix)
            evaluator = GraphEvaluator()
            metrics = evaluator._compute_single_metrics(gt_summary, adj_matrix)
            f1_scores.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            shds.append(metrics['shd'])

        # Calculate average and standard deviation
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        avg_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        avg_shd = np.mean(shds)
        std_shd = np.std(shds)

        print("\nAverage Metrics:")
        print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"SHD: {avg_shd:.4f} ± {std_shd:.4f}")

if __name__ == "__main__":
    params = {
        'p': 2,
        'lambda1': 1e-5,
        'lambda2': 1e-5,
        'w_threshold':8,
        'max_iter': 100,
    }
    ntsnotears_algo = NTSNOTEARS(params)
    ntsnotears_algo.test_algorithm() 