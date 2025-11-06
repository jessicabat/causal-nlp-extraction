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
from castle.algorithms import GOLEM as golem

class GOLEM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'lambda_1': 1e-2,  # L1 penalty coefficient
            'lambda_2': 5.0,  # DAG penalty coefficient
            'equal_variances': True,  # Whether to assume equal noise variances
            'learning_rate': 1e-3,  # Learning rate for Adam optimizer
            'num_iter': 1e4,  # Number of training iterations (default: 1e5)
            'checkpoint_iter': 5000,  # Iterations between checkpoints
            'seed': 1,  # Random seed
            'graph_thres': 0.3,  # Threshold for weighted matrix
            'device_type': 'auto',  # Device type ('cpu' or 'gpu' or 'auto')
            'device_ids': 0  # GPU device IDs to use
        }
        self._params.update(params)
        # Automatically decide device_type if set to 'auto'
        if self._params.get('device_type', 'cpu') == 'auto':
            try:
                import torch
                self._params['device_type'] = 'gpu' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self._params['device_type'] = 'cpu'

    @property
    def name(self):
        return "GOLEM"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['lambda_1', 'num_iter', 'graph_thres']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['lambda_2', 'learning_rate', 'equal_variances',
                                      'checkpoint_iter', 'seed', 'device_type', 'device_ids']
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

        all_params = {**self.get_primary_params(), **self.get_secondary_params()}
        model = golem(**all_params)

        model.learn(data)
        
        # GOLEM returns transposed matrix compared to our convention
        adj_matrix = model.causal_matrix.T

        info = {
            'adj_matrix': adj_matrix,
            'node_names': node_names
        }

        return adj_matrix, info, adj_matrix
    def test_algorithm(self):
        import time
        import numpy as np
        import pandas as pd
        from causal_discovery.evaluation.evaluator import GraphEvaluator
        from data.simulator.dummy import DataSimulator

        # Fix all random seeds for reproducibility
        np.random.seed(42)
        
        # Set random seeds for other libraries if they're being used
        import random
        random.seed(42)
        
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
            
        # Set TensorFlow seed if it's being used
        try:
            import tensorflow as tf
            tf.random.set_seed(42)
        except ImportError:
            pass
        
        start_time = time.time()
        
        def degree2prob(degree, node_size):
            return degree / (node_size-1)
        
        node_sizes = [20]  # [5, 10, 15, 20, 25]
        sample_sizes = [5000]  # [500, 1000, 1500, 2000]
        num_runs = 1  # Number of runs to average results
        edge_probability = degree2prob(7, node_sizes[0])
        
        # Define different parameter configurations to compare
        configurations = [
            {"name": "Default", "lambda_1": 0.01, "num_iter": 10000, "graph_thres": 0.3},
            {"name": "High Sparsity", "lambda_1": 0.1, "num_iter": 10000, "graph_thres": 0.3},
            {"name": "Low Sparsity", "lambda_1": 0.001, "num_iter": 10000, "graph_thres": 0.3},
        ]
        
        results = {}
        
        print("Testing GOLEM algorithm with different configurations")
        print("=" * 80)
        print(f"Running {num_runs} iterations for each configuration")
        print("=" * 80)
        
        for config in configurations:
            config_name = config["name"]
            results[config_name] = {}
            
            print(f"\nTesting configuration: {config_name}")
            print("-" * 60)
            
            for n_nodes in node_sizes:
                results[config_name][n_nodes] = {}
                for n_samples in sample_sizes:
                    metrics_list = []
                    time_list = []
                    
                    print(f"\nTesting with {n_nodes} nodes and {n_samples} samples:")
                    
                    for run in range(num_runs):
                        print(f"  Run {run+1}/{num_runs}...")
                        
                        # Create a DataSimulator instance with new random seed for each run
                        seed = 42 + run
                        np.random.seed(seed)
                        simulator = DataSimulator()
                        
                        # Generate data
                        gt_graph, df = simulator.generate_dataset(
                            n_samples=n_samples, 
                            n_nodes=n_nodes, 
                            noise_type='gaussian',
                            function_type='linear', 
                            edge_probability=edge_probability,
                            n_domains=1
                        )
                        
                        # Configure GOLEM algorithm based on current configuration
                        self._params['lambda_1'] = config["lambda_1"]
                        self._params['num_iter'] = config["num_iter"]
                        self._params['graph_thres'] = config["graph_thres"]
                        
                        run_start_time = time.time()
                        adj_matrix, info, _ = self.fit(df)
                        run_time = time.time() - run_start_time
                        
                        # Evaluate results
                        evaluator = GraphEvaluator()
                        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
                        
                        # Store results
                        metrics_list.append(metrics)
                        time_list.append(run_time)
                    
                    # Calculate average metrics
                    avg_metrics = {
                        'f1': np.mean([m['f1'] for m in metrics_list]),
                        'precision': np.mean([m['precision'] for m in metrics_list]),
                        'recall': np.mean([m['recall'] for m in metrics_list]),
                        'shd': np.mean([m['shd'] for m in metrics_list]),
                        'time': np.mean(time_list)
                    }
                    
                    results[config_name][n_nodes][n_samples] = avg_metrics
                    
                    # Print average results for this configuration
                    print(f"  Results for {n_nodes} nodes, {n_samples} samples (averaged over {num_runs} runs):")
                    print(f"    F1 Score: {avg_metrics['f1']:.4f}")
                    print(f"    Precision: {avg_metrics['precision']:.4f}")
                    print(f"    Recall: {avg_metrics['recall']:.4f}")
                    print(f"    SHD: {avg_metrics['shd']:.4f}")
                    print(f"    Time: {avg_metrics['time']:.4f} seconds")
        
        # Print summary of results for each configuration
        print("\n" + "=" * 80)
        print("SUMMARY OF RESULTS")
        print("=" * 80)
        
        for config_name in results:
            print(f"\n{config_name}:")
            print("-" * 60)
            
            print("F1 Scores:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['f1']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nPrecision:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['precision']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nRecall:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['recall']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nSHD:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['shd']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
        
        total_time = time.time() - start_time
        print(f"\nTotal experiment time: {total_time:.2f} seconds")
        
        # Analyze and print conclusions
        print("\n" + "=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)
        
        # Compare configurations
        print("\nConfiguration Comparison:")
        # Calculate average F1 across all node/sample combinations for each config
        config_avg_f1 = {}
        for config_name in results:
            all_f1 = []
            for n_nodes in node_sizes:
                for n_samples in sample_sizes:
                    all_f1.append(results[config_name][n_nodes][n_samples]['f1'])
            config_avg_f1[config_name] = np.mean(all_f1)
        
        # Sort configs by average F1
        sorted_configs = sorted(config_avg_f1.items(), key=lambda x: x[1], reverse=True)
        print("  Configurations ranked by average F1 score:")
        for i, (config_name, avg_f1) in enumerate(sorted_configs):
            print(f"    {i+1}. {config_name}: {avg_f1:.4f}")
        
        # Final recommendations
        print("\nRecommendations:")
        print("  • For optimal performance, the lambda_1 parameter should be tuned based on expected graph sparsity")
        print("  • Higher lambda_1 values (0.1) promote sparser graphs, while lower values (0.001) allow more connections")
        print("  • The default lambda_1 value (0.01) provides a good balance for most datasets")
        print("  • Increasing num_iter may improve results for complex graphs but increases computation time")

if __name__ == "__main__":
    golem_algo = GOLEM({})
    golem_algo.test_algorithm()
