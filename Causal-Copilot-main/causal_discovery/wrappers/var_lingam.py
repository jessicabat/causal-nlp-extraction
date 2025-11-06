import numpy as np
import pandas as pd
from typing import Dict, Tuple

# use the local lingam package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)
sys.path.append(causal_learn_dir)

from causallearn.search.FCMBased.lingam import VARLiNGAM as VARLiNGAM_model

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator
from causal_discovery.wrappers.utils.ts_utils import generate_stationary_linear

import torch
cuda_available = torch.cuda.is_available()
try:
    from culingam.varlingam import VARLiNGAM as AcVarLiNGAM
    from culingam.utils import check_array
    from culingam.directlingam import DirectLiNGAM as AcDirectLiNGAM
except ImportError:
    if not cuda_available:
        from utils.logger import logger
        logger.debug("CUDA not available, using CPU acceleration", "VARLiNGAM")

if cuda_available:
    class AcVarLiNGAMWrapper(AcVarLiNGAM):
        """
        A wrapper for AcVarLiNGAM that fixes compatibility issues with the fit method.
        This class overrides the fit method to handle the 'original' parameter issue.
        """
        
        def __init__(self, lags=1, criterion='bic', prune=True, ar_coefs=None, lingam_model=None, random_state=None):
            """
            Initialize the AcVarLiNGAM wrapper with the same parameters as the original class.
            
            Parameters
            ----------
            lags : int, optional (default=1)
                Number of lags.
            criterion : {'aic', 'fpe', 'hqic', 'bic', None}, optional (default='bic')
                Criterion to decide the best lags within ``lags``.
            prune : boolean, optional (default=True)
                Whether to prune the adjacency matrix of lags or not.
            ar_coefs : array-like, optional (default=None)
                Coefficients of AR model.
            lingam_model : lingam object, optional (default=None)
                LiNGAM model for causal discovery.
            random_state : int, optional (default=None)
                Seed for random number generator.
            """
            super().__init__(lags=lags, criterion=criterion, prune=prune, 
                            ar_coefs=ar_coefs, lingam_model=lingam_model, 
                            random_state=random_state)
        
        def fit(self, X, disable_tqdm=False):
            """
            Fit the model to X, removing the 'original' parameter that causes issues.
            
            Parameters
            ----------
            X: array-like, shape (n_samples, n_features)
                Training data, where ``n_samples`` is the number of samples
                and ``n_features`` is the number of features.
            disable_tqdm: bool, optional (default=False)
                Whether to disable the tqdm progress bar.
                
            Returns
            -------
            self : object
                Returns the instance itself.
            """
            self._causal_order = None
            self._adjacency_matrices = None
            
            X = check_array(X)
            
            lingam_model = self._lingam_model
            if lingam_model is None:
                lingam_model = AcDirectLiNGAM()
                
            M_taus = self._ar_coefs
            
            if M_taus is None:
                M_taus, lags, residuals = self._estimate_var_coefs(X)
            else:
                lags = M_taus.shape[0]
                residuals = self._calc_residuals(X, M_taus, lags)
                
            model = lingam_model
            # Call fit without the 'original' parameter
            model.fit(residuals, disable_tqdm=disable_tqdm)
            
            B_taus = self._calc_b(X, model._adjacency_matrix, M_taus)
            
            if self._prune:
                B_taus = self._pruning(X, B_taus, model._causal_order)
                
            self._ar_coefs = M_taus
            self._lags = lags
            self._residuals = residuals
            
            self._causal_order = model._causal_order
            self._adjacency_matrix = B_taus
            return self


# {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}
class VARLiNGAM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'lags': 10,
            'criterion': "bic",
            'prune': True,
            'ar_coefs': None,
            'lingam_model': None,
            'random_state': None,
            'gpu': True
        }
        self._params.update(params)

    @property
    def name(self):
        return "VARLiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['lags', 'criterion', 'prune', 'gpu']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['ar_coefs', 'lingam_model', 'random_state']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}


    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, VARLiNGAM_model]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}
        all_params.pop('gpu')

        if cuda_available and self._params['gpu']:
            model = AcVarLiNGAM(**all_params)
            model = model.fit(data_values)
            lag_matrix = model._adjacency_matrix
        else:
            model = VARLiNGAM_model(**all_params)
            model.fit(data_values)
            lag_matrix = model._adjacency_matrices

        # Prepare additional information
        info = {
            'model': model,
            'lag': model._lags,
            'residuals': model._residuals,
            'causal_order': model.causal_order_,
            'lag_matrix': lag_matrix
        }
        summary_matrix = np.any(lag_matrix, axis=0).astype(int)
        
        return summary_matrix, info, model
    
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
        print("Testing VARLINGAM algorithm with pandas DataFrame:")
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
        'lags': 2,
        'criterion': "bic",
        'prune': False,
        'ar_coefs': None,
        'lingam_model': None,
        'random_state': None,
    }
    algo = VARLiNGAM(params)
    algo.test_algorithm() 

