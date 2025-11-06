import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import time

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

# Add path for GPU-accelerated CDNOD
acceleration_dir = os.path.join(root_dir, 'externals', 'acceleration')
sys.path.append(acceleration_dir)

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.CDNOD import cdnod
from externals.acceleration.cdnod.cdnod import accelerated_cdnod

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator

class TSCDNOD(CausalDiscoveryAlgorithm):
    """
    Time-Series CDNOD (TS-CDNOD) algorithm for causal discovery in time series data.
    
    This algorithm constructs time-delayed variables up to tau_max and then runs CDNOD
    with time index as the context index.
    """
    
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': 4,
            'tau_max': 3,
            'use_gpu': False,
            'verbose': False
        }
        self._params.update(params)
        
    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['tau_max', 'use_gpu', 'verbose']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}
    
    def _create_time_delayed_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create time-delayed variables up to tau_max.
        
        Args:
            data: Input time series data as a pandas DataFrame
            
        Returns:
            Tuple containing:
            - time-delayed data as numpy array
            - time index as numpy array
            - list of variable names with time lags
        """
        tau_max = self._params['tau_max']
        verbose = self._params['verbose']
        
        if verbose:
            print(f"Creating time-delayed variables with tau_max={tau_max}")
        
        # Original variable names
        orig_var_names = list(data.columns)
        n_vars = len(orig_var_names)
        n_samples = len(data)
        
        # Create time-delayed variable names
        var_names = []
        for t in range(tau_max + 1):
            for var in orig_var_names:
                var_names.append(f"{var}(t-{t})")
        
        # Create time-delayed data matrix
        # We'll lose the first tau_max samples due to lagging
        effective_samples = n_samples - tau_max
        delayed_data = np.zeros((effective_samples, n_vars * (tau_max + 1)))
        
        # Fill the delayed data matrix
        for t in range(tau_max + 1):
            for i, var in enumerate(orig_var_names):
                col_idx = t * n_vars + i
                delayed_data[:, col_idx] = data.iloc[tau_max - t:n_samples - t, i].values
        
        # Create time index as context variable
        time_index = np.arange(effective_samples).reshape(-1, 1)
        
        if verbose:
            print(f"Created delayed data with shape {delayed_data.shape}")
            print(f"Time index shape: {time_index.shape}")
        
        return delayed_data, time_index, var_names
    
    def _process_causal_graph(self, adj_matrix: np.ndarray, var_names: List[str]) -> np.ndarray:
        """
        Process the causal graph to create a summary graph showing relationships between original variables.
        
        Args:
            adj_matrix: Adjacency matrix from CDNOD
            var_names: List of variable names with time lags
            
        Returns:
            Summary adjacency matrix for original variables
        """
        tau_max = self._params['tau_max']
        n_vars_with_lags = len(var_names)
        n_orig_vars = n_vars_with_lags // (tau_max + 1)
        
        # Remove the time index column/row
        adj_matrix = adj_matrix[:-1, :-1]
        
        # Create a summary adjacency matrix for original variables
        summary_adj = np.zeros((n_orig_vars, n_orig_vars))
        
        # For each pair of original variables, check if there's a causal link in any time lag
        for i in range(n_orig_vars):
            for j in range(n_orig_vars):
                # Check if variable i at any lag causes variable j at t=0
                j_at_t0 = j
                for t in range(tau_max + 1):
                    i_with_lag = i + t * n_orig_vars
                    if adj_matrix[i_with_lag, j_at_t0] == 1:
                        summary_adj[i, j] = 1
                        break
        
        return summary_adj
    
    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Optional[CausalGraph]]:
        """
        Fit the TS-CDNOD algorithm to time series data.
        
        Args:
            data: Input time series data as a pandas DataFrame
            
        Returns:
            Tuple containing:
            - adjacency matrix
            - info dictionary
            - CausalGraph object (if available)
        """
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        # Get parameters
        alpha = self._params['alpha']
        indep_test = self._params['indep_test']
        depth = self._params['depth']
        use_gpu = self._params['use_gpu']
        verbose = self._params['verbose']
        
        # Create time-delayed variables
        delayed_data, time_index, var_names = self._create_time_delayed_data(data)
        
        # Original variable names (without time lags)
        orig_var_names = list(data.columns)
        
        start_time = time.time()
        
        # Run CDNOD with time index as context
        if use_gpu:
            if verbose:
                print("Using GPU-accelerated CDNOD")
            adj_matrix = accelerated_cdnod(
                data=delayed_data,
                c_indx=time_index,
                indep_test=indep_test,
                alpha=alpha,
                depth=depth
            )
            cg = None  # GPU version doesn't return CausalGraph object
        else:
            if verbose:
                print("Using CPU version of CDNOD")
            cg = cdnod(
                data=delayed_data,
                c_indx=time_index,
                indep_test=indep_test,
                alpha=alpha,
                depth=depth
            )
            adj_matrix = cg.G.graph
        
        end_time = time.time()
        
        # Process the causal graph to create a summary graph
        summary_adj = self._process_causal_graph(adj_matrix, var_names)
        
        # Prepare additional information
        info = {
            'var_names': orig_var_names,
            'var_names_with_lags': var_names,
            'tau_max': self._params['tau_max'],
            'runtime': end_time - start_time,
            'full_adj_matrix': adj_matrix
        }
        
        return summary_adj, info, cg
