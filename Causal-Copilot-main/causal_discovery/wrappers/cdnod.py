import numpy as np
import pandas as pd
from typing import Dict, Tuple
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

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.CDNOD import cdnod as cl_cdnod
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from data.simulator.dummy import DataSimulator
from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.wrappers.pc import PC
from causal_discovery.evaluation.evaluator import GraphEvaluator

import torch
cuda_available = torch.cuda.is_available()
try:
    from externals.acceleration.cdnod.cdnod import accelerated_cdnod
except ImportError:
    if not cuda_available:
        from utils.logger import logger
        logger.debug("CUDA not available, using CPU acceleration", "CDNOD")


class CDNOD(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz_cpu',
            'stable': True,
            'uc_rule': 0,
            'uc_priority': 2,
            'depth': 5, #-1,
            'mvcdnod': False,
            'correction_name': 'MV_Crtn_Fisher_Z',
            'background_knowledge': None,
            'verbose': False,
            'show_progress': False,
        }
        self._params.update(params)

    def _create_background_knowledge(self, background_knowledge_spec, node_names):
        """
        Create BackgroundKnowledge object from JSON specification.
        
        Args:
            background_knowledge_spec: Dictionary or string specifying background knowledge
            node_names: List of variable names
            
        Returns:
            BackgroundKnowledge object or None
        """
        if background_knowledge_spec is None:
            return None
            
        # Create nodes
        nodes = [GraphNode(name) for name in node_names]
        node_dict = {name: node for name, node in zip(node_names, nodes)}
        
        bk = BackgroundKnowledge()
        
        if isinstance(background_knowledge_spec, str):
            # Handle string specifications like "forbidden_edges", "required_edges", etc.
            return None
        elif isinstance(background_knowledge_spec, dict):
            # Handle dictionary specification
            if 'forbidden_edges' in background_knowledge_spec:
                for edge in background_knowledge_spec['forbidden_edges']:
                    var1, var2 = edge
                    if var1 in node_dict and var2 in node_dict:
                        bk.add_forbidden_by_node(node_dict[var1], node_dict[var2])
                        
            if 'required_edges' in background_knowledge_spec:
                for edge in background_knowledge_spec['required_edges']:
                    var1, var2 = edge
                    if var1 in node_dict and var2 in node_dict:
                        bk.add_required_by_node(node_dict[var1], node_dict[var2])
                        
            if 'tiers' in background_knowledge_spec:
                for var_name, tier_level in background_knowledge_spec['tiers'].items():
                    if var_name in node_dict:
                        bk.add_node_to_tier(node_dict[var_name], tier_level)
                        
            return bk
        
        return None

    @property
    def name(self):
        return "CDNOD"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvcdnod', 'correction_name',
                                'background_knowledge', 'verbose', 'show_progress']
        return {k: v for k, v in self._params.items() if k in secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        node_names = list(data.columns)
        
        # Extract c_indx (assuming it's the last column)
        if 'domain_index' not in data.columns:
            print("Dataset must contain a 'domain_index' column for CDNOD, simulating one for testing")
            data['domain_index'] = np.ones(data.shape[0])
        c_indx = data['domain_index'].values.reshape(-1, 1)
        data = data.drop(columns=['domain_index'])
        data_values = data.values
        
        # Get original node names (without domain_index)
        original_node_names = list(data.columns)

        # Check if GPU implementation should be used
        if cuda_available and 'gpu' in self._params['indep_test']:
            use_gpu = True
            self._params['indep_test'] = self._params['indep_test'].replace('_gpu', '')
        else:
            use_gpu = False
            self._params['indep_test'] = self._params['indep_test'].replace('_cpu', '')
        
        start_time = time.time()
        
        if use_gpu:
            # Use GPU implementation (doesn't support background knowledge)
            all_params = {
                'alpha': self._params['alpha'],
                'indep_test': self._params['indep_test'],
                'depth': self._params['depth'],
            }
            cg = accelerated_cdnod(data_values, c_indx, **all_params)
            # Convert the graph to adjacency matrix
            adj_matrix = self.convert_to_adjacency_matrix(cg)
            
            # Prepare additional information
            info = {
                'graph': cg,
                'PC_elapsed': time.time() - start_time,
            }
        else:
            # Use CPU implementation with background knowledge support
            secondary_params = self.get_secondary_params().copy()
            
            # Process background knowledge if provided
            if secondary_params.get('background_knowledge') is not None:
                bk = self._create_background_knowledge(
                    secondary_params['background_knowledge'], 
                    original_node_names
                )
                secondary_params['background_knowledge'] = bk
            
            # Combine primary and secondary parameters
            all_params = {**self.get_primary_params(), **secondary_params, 'node_names': original_node_names}
            
            # Run CD-NOD algorithm
            cg = cl_cdnod(data_values, c_indx, **all_params)
            
            # Convert the graph to adjacency matrix
            adj_matrix = self.convert_to_adjacency_matrix(cg.G.graph)
            
            # Prepare additional information
            info = {
                'graph': cg,
                'PC_elapsed': cg.PC_elapsed if hasattr(cg, 'PC_elapsed') else time.time() - start_time,
            }

        return adj_matrix, info, cg

    def convert_to_adjacency_matrix(self, adj_matrix) -> np.ndarray:
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # directed edge: j -> i
                inferred_flat[i, j] = 1
            elif adj_matrix[j, i] == 1:
                # bidirected edge: j <-> i
                if inferred_flat[j, i] == 0:
                    # keep asymmetric that only one entry is recorded
                    inferred_flat[i, j] = 3

        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # undirected edge: j -- i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 2

        # remove the domain index column
        print(inferred_flat)

        inferred_flat = inferred_flat[:-1, :-1]
        return inferred_flat

    def test_algorithm(self):
        # Generate sample data with linear relationships and multiple domains
        np.random.seed(42)
        n_samples = 1000
        n_nodes = 10
        n_domains = 10
        # 0.5 prob = 2 / 4
        # 0.095 prob = 2 / 9
        edge_probability = 2/9 # 2/(n_nodes - 1)
        # print(f"Edge probability: {edge_probability}")
        
        # Create a DataSimulator instance
        simulator = DataSimulator()
                
        # Generate multi-domain data with the same graph structure but domain-specific parameters
        gt_graph, df = simulator.generate_dataset(n_samples=n_samples, n_nodes=n_nodes, noise_type='gaussian',
                               function_type='mlp', edge_probability=edge_probability, n_domains=n_domains)

        # Fill missing values with 0 (as done in the benchmarking code)
        # df = df.fillna(0)
        
        # Ensure domain_index column exists
        if 'domain_index' not in df.columns:
            print("Warning: domain_index column not found in the dataset. Adding a default domain index.")
            df['domain_index'] = np.ones(df.shape[0])
                
        print(f"Testing CDNOD algorithm with {n_domains} domains:")
        print(f"Domain distribution: {df['domain_index'].value_counts().sort_index().values}")
        print(f"Ground truth graph structure:")
        print(gt_graph)

        pc = PC()
        print("Indep test: ", pc._params['indep_test'])
        # Test CPU implementation
        print("\nRunning CPU implementation:")
        start_time = time.time()
        adj_matrix_pc, info_pc, _ = pc.fit(df)
        time_elapsed = time.time() - start_time
        print(f"Time elapsed: {time_elapsed:.4f} seconds")

        cdnod = CDNOD()
        print("Indep test: ", cdnod._params['indep_test'])
        # Test CPU implementation
        print("\nRunning CPU implementation:")
        start_time = time.time()
        adj_matrix, info, _ = cdnod.fit(df)
        time_elapsed = time.time() - start_time
        print(f"Time elapsed: {time_elapsed:.4f} seconds")

        # cdnod_nl = CDNOD({'indep_test': 'kci_cpu'})
        # print("Indep test: ", cdnod_nl._params['indep_test'])
        # # Test CPU implementation
        # print("\nRunning CPU implementation:")
        # start_time = time.time()
        # adj_matrix_nl, info_nl, _ = cdnod_nl.fit(df)
        # time_elapsed = time.time() - start_time
        # print(f"Time elapsed: {time_elapsed:.4f} seconds")

        cdnod_cmi = CDNOD({'indep_test': 'rcit_cpu'})
        print("Indep test: ", cdnod_cmi._params['indep_test'])
        # Test CPU implementation
        print("\nRunning CPU implementation:")
        start_time = time.time()
        adj_matrix_cmi, info_cmi, _ = cdnod_cmi.fit(df)
        time_elapsed = time.time() - start_time
        print(f"Time elapsed: {time_elapsed:.4f} seconds")

        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
        # metrics_nl = evaluator.compute_metrics(gt_graph, adj_matrix_nl)
        metrics_cmi = evaluator.compute_metrics(gt_graph, adj_matrix_cmi)
        metrics_pc = evaluator.compute_metrics(gt_graph, adj_matrix_pc)

        print(f"\nMetrics: {'|'.join([f'{k}: {v:.3f}' for k, v in metrics.items() if k != 'best_graph'])}")
        # print(f"NL Metrics: {'|'.join([f'{k}: {v:.3f}' for k, v in metrics_nl.items() if k != 'best_graph'])}")
        print(f"CMI Metrics: {'|'.join([f'{k}: {v:.3f}' for k, v in metrics_cmi.items() if k != 'best_graph'])}")
        print(f"PC Metrics: {'|'.join([f'{k}: {v:.3f}' for k, v in metrics_pc.items() if k != 'best_graph'])}")

if __name__ == "__main__":
    cdnod = CDNOD()
    cdnod.test_algorithm()