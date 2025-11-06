import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
import importlib
import inspect
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from causal_discovery.evaluation.evaluator import GraphEvaluator
from causal_discovery.tests.benchmarking_config import get_config

class BenchmarkRunner:
    def __init__(self, algorithm, hyperparams=None, debug_mode: bool = False, resume_dir: str = None):
        self.config = get_config()
        self.algorithm = algorithm
        self.hyperparams = hyperparams or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = self.config['data_dir']
        self.resume_dir = resume_dir 
        self.output_dir = self.resume_dir if resume_dir else os.path.join(self.config['output_dir'], self.timestamp)
        self.evaluator = GraphEvaluator()
        self.debug_mode = debug_mode
        self.processed_datasets = set()  # Track already processed datasets
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Benchmarking data directory: {self.data_dir}")
        print(f"Benchmarking output directory: {self.output_dir}")
        
        # Create algorithm identifier with hyperparameters
        self.algo_id = self._create_algo_id()
        
        # Load all algorithm classes from wrappers
        self.algorithm_candidates = [self.algorithm]
        self.algorithms = self._load_algorithms()
        
        # If resuming, load existing results to avoid reprocessing
        if self.resume_dir:
            self._load_existing_results()
        
        # Index all available datasets
        self.dataset_index = self._index_datasets()
        
    def _create_algo_id(self) -> str:
        """Create a unique identifier for algorithm with hyperparameters"""
        if not self.hyperparams:
            return self.algorithm
            
        # Create a string representation of the hyperparameters
        param_str = "_".join([f"{k}={v}" for k, v in sorted(self.hyperparams.items())])
        return f"{self.algorithm}_{param_str}"

    def _load_algorithms(self) -> Dict:
        """Dynamically load all algorithm classes from wrappers directory"""
        algorithms = {}
        wrappers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wrappers')
        
        # Import all algorithm classes from __init__.py
        import causal_discovery.wrappers as wrappers
        
        # Get all available algorithm classes
        available_algorithms = {}
        for name in self.algorithm_candidates: # wrappers.__all__:
            available_algorithms[name] = getattr(wrappers, name)
            
        # Only keep algorithms specified in config
        for algo_name in self.algorithm_candidates:
            if algo_name in available_algorithms:
                algorithms[algo_name] = available_algorithms[algo_name]
                # print(f"Added algorithm: {algo_name}")
        
        print(f"Loaded {len(algorithms)} algorithms, {' | '.join(list(algorithms.keys()))}")
                        
        return algorithms

    
    def _index_datasets(self) -> List[Tuple[str, str, Dict]]:
        """Index all available datasets, including absolute paths and configs"""
        dataset_index = []
        
        # Check if the data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return dataset_index
            
        # Recursively walk through all directories
        for root, dirs, files in os.walk(self.data_dir):
            # Check if this directory contains dataset files (config.json)
            config_files = [f for f in files if f.endswith('config.json') and f != 'simulation_config.json' and f != 'dataset_configs.json']
            if not config_files:
                continue
                
            # Process each config file in the directory
            for config_file in config_files:
                config_path = os.path.join(root, config_file)
                dataset_dir_abs = os.path.dirname(config_path)
                
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Extract seed number using regex
                    import re
                    seed_match = re.search(r'seed_(\d+)', config_path)
                    seed_num = int(seed_match.group(1)) if seed_match else 0
                    config['seed'] = seed_num
                    
                    # Determine if this is a time series dataset or regular dataset
                    is_ts_dataset = 'lag' in config and 'degree_inter' in config and 'degree_intra' in config
                    
                    # Create dataset name with key configuration parameters
                    if is_ts_dataset:
                        # Time series dataset naming
                        dataset_name = f"nodes{config.get('n_nodes', config.get('num_nodes', 'unknown'))}_lag{config.get('lag', 'unknown')}_samples{config.get('n_samples', 'unknown')}_noise{config.get('noise_type', 'unknown')}"
                        
                        if 'degree_inter' in config:
                            dataset_name += f"_degreeInter{config['degree_inter']}"
                        if 'degree_intra' in config:
                            dataset_name += f"_degreeIntra{config['degree_intra']}"
                        if 'function_type' in config:
                            dataset_name += f"_func{config['function_type']}"
                    else:
                        # Regular dataset naming
                        dataset_name = f"nodes{config.get('n_nodes', 'unknown')}_samples{config.get('n_samples', 'unknown')}_domains{config.get('n_domains', 'unknown')}_noise{config.get('noise_type', 'unknown')}_func{config.get('function_type', 'unknown')}"
                        
                        if 'edge_probability' in config and config['edge_probability'] is not None:
                            dataset_name += f"_edgeprob{config['edge_probability']}"
                        if 'discrete_ratio' in config and config['discrete_ratio'] is not None:
                            dataset_name += f"_discrete{config['discrete_ratio']}"
                        if 'measurement_error' in config and config['measurement_error'] is not None:
                            measurement_error = config['measurement_error_value']
                            dataset_name += f"_measurement{measurement_error}"
                        if 'missing_rate' in config and config['missing_rate'] is not None:
                            missing_rate = config['missing_rate_value']
                            dataset_name += f"_missing{missing_rate}"
                    
                    # Add experiment type if available
                    if 'experiment_type' in config:
                        dataset_name = f"{config['experiment_type']}_{dataset_name}"
                    
                    # Add seed to all dataset names
                    dataset_name += f"_seed{seed_num}"
                    
                    # Add scenario/category information from directory structure
                    rel_path = os.path.relpath(dataset_dir_abs, self.data_dir)
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 1 and path_parts[0] != '.':
                        scenario = path_parts[0]
                        if len(path_parts) > 2:
                            scenario += f"/{path_parts[1]}"
                        dataset_name = f"{scenario}_{dataset_name}"
                    
                    # Standardize config keys for time series datasets
                    if is_ts_dataset:
                        # Map time series specific keys to standard keys
                        if 'num_nodes' in config and 'n_nodes' not in config:
                            config['n_nodes'] = config['num_nodes']
                        if 'n_samples' not in config and 'sample_size' in config:
                            config['n_samples'] = config['sample_size']
                    
                    dataset_index.append((dataset_name, dataset_dir_abs, config))
                    
                except Exception as e:
                    print(f"Error processing config file {config_path}: {e}")
        
        print(f"Found {len(dataset_index)} datasets")
        
        # Sort by n_nodes first, then by n_samples
        dataset_index.sort(key=lambda x: (
            x[2].get('n_nodes', 0), 
            x[2].get('n_samples', 0)
        ))
            
        return dataset_index
    
    def _load_dataset(self, dataset_dir: str) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Load data, true graph, and config from a dataset directory"""
        data_file = [file for file in os.listdir(dataset_dir) if 'data.csv' in file][0]
        data = pd.read_csv(os.path.join(dataset_dir, data_file))
        true_graph = [file for file in os.listdir(dataset_dir) if 'graph.npy' in file][0]
        true_graph = np.load(os.path.join(dataset_dir, true_graph))
        config_file = [file for file in os.listdir(dataset_dir) if 'config.json' in file][0]
        config = json.load(open(os.path.join(dataset_dir, config_file)))

        return data, true_graph, config

    def _run_single_experiment(self, algo_instance: Any, data: pd.DataFrame, 
                             true_graph: np.ndarray, config: Dict) -> Dict:
        """Run a single experiment with given algorithm and configuration"""
        start_time = time.time()
        
        # Set a timeout limit (20 minutes)
        timeout_seconds = 1200  # 20 minutes
        
        try:
            # fill the missing value if it is existed
            data = data.fillna(0)

            # Skip if the algorithm is not designed for heterogeneous data
            if 'domain_index' in data.columns and "CDNOD" not in algo_instance.name:
                data = data.drop(columns=['domain_index'])
            if 'domain_index' not in data.columns and "CDNOD" in algo_instance.name:
                data["domain_index"] = np.ones(data.shape[0])

            # Use threading with a timeout instead of multiprocessing
            # This avoids issues with GPU processes not working properly with mp.Process
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def run_algorithm():
                try:
                    adj_matrix, info, _ = algo_instance.fit(data)
                    print(f"adj_matrix: {adj_matrix}")
                    print(f"info: {info}")
                    result_queue.put((adj_matrix, info))
                except Exception as e:
                    result_queue.put(e)
            
            # Start the algorithm in a thread
            thread = threading.Thread(target=run_algorithm)
            thread.daemon = True
            thread.start()
            
            # Wait for the thread to complete or timeout
            thread.join(timeout_seconds)
            
            # Check if the thread is still running after timeout
            if thread.is_alive():
                # We can't forcefully terminate a thread in Python,
                # but we can indicate a timeout occurred
                raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
            
            # Get the result from the queue
            if result_queue.empty():
                raise Exception("Process completed but no result was returned")
            
            result = result_queue.get()
            
            # Check if the result is an exception
            if isinstance(result, Exception):
                raise result
            
            adj_matrix, info = result
            
            runtime = time.time() - start_time

            is_time_series = 'lag' in config and 'degree_inter' in config and 'degree_intra' in config
            if is_time_series:
                metrics = self.evaluator._compute_single_metrics(true_graph, adj_matrix)
            else:
                metrics = self.evaluator.compute_metrics(true_graph, adj_matrix)
            
            result = {
                "success": True,
                "runtime": runtime,
                "metrics": metrics,
                "algorithm_params": algo_instance.get_params(),
                "data_config": config
            }
            
        except TimeoutError as e:
            result = {
                "success": False,
                "error": str(e),
                "data_config": config
            }
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "data_config": config
            }
            
        return result
    
    
    def _save_result(self, algo_id: str, result: Dict) -> None:
        """Save the result of a single experiment to a JSON file"""
        output_file = os.path.join(self.output_dir, f"{algo_id}_results.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        else:
            existing_results = []
        
        existing_results.append(result)
        
        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=2, default=str)
            
    def _load_existing_results(self) -> None:
        """Load existing results to avoid reprocessing datasets"""
        result_file = os.path.join(self.output_dir, f"{self.algo_id}_results.json")
        if os.path.exists(result_file):
            print(f"Found existing results file: {result_file}")
            try:
                with open(result_file, 'r') as f:
                    existing_results = json.load(f)
                
                # Extract dataset names from successful runs
                for result in existing_results:
                    if result.get('success', False) and 'dataset' in result:
                        self.processed_datasets.add(result['dataset'])
                
                print(f"Loaded {len(self.processed_datasets)} previously processed datasets")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                self.processed_datasets = set()

    def run_all_experiments(self) -> None:
        """Run all benchmark experiments"""
        all_results = {}
        
        # Group datasets by type and characteristics for smarter timeout handling
        tabular_groups = {}  # For regular tabular datasets
        timeseries_groups = {}  # For time series datasets
        
        for dataset_name, dataset_dir, config in self.dataset_index:
            # Skip already processed datasets if we're resuming
            if dataset_name in self.processed_datasets:
                print(f"Skipping already processed dataset: {dataset_name}")
                continue
                
            node_size = config['n_nodes']
            sample_size = config['n_samples']
            function_type = config.get('function_type', 'unknown')
            n_domains = config.get('n_domains', 1)
            
            # Determine if this is a time series dataset
            is_time_series = 'lag' in config and 'degree_inter' in config and 'degree_intra' in config
            
            if is_time_series:
                # Time series dataset grouping
                lag = config.get('lag', 0)
                degree_inter = config.get('degree_inter', 0)
                degree_intra = config.get('degree_intra', 0)
                
                # Create hierarchical grouping
                if node_size not in timeseries_groups:
                    timeseries_groups[node_size] = {}
                if lag not in timeseries_groups[node_size]:
                    timeseries_groups[node_size][lag] = {}
                if degree_inter not in timeseries_groups[node_size][lag]:
                    timeseries_groups[node_size][lag][degree_inter] = {}
                if degree_intra not in timeseries_groups[node_size][lag][degree_inter]:
                    timeseries_groups[node_size][lag][degree_inter][degree_intra] = []
                
                config_signature = f"ts_nodes{node_size}_lag{lag}_func{function_type}_degreeInter{degree_inter}_degreeIntra{degree_intra}"
                timeseries_groups[node_size][lag][degree_inter][degree_intra].append(
                    (dataset_name, dataset_dir, config, config_signature, sample_size)
                )
            else:
                # Tabular dataset grouping
                edge_probability = config.get('edge_probability', 0)
                
                # Create hierarchical grouping with additional n_domains and function_type
                if node_size not in tabular_groups:
                    tabular_groups[node_size] = {}
                if sample_size not in tabular_groups[node_size]:
                    tabular_groups[node_size][sample_size] = {}
                if edge_probability not in tabular_groups[node_size][sample_size]:
                    tabular_groups[node_size][sample_size][edge_probability] = {}
                if n_domains not in tabular_groups[node_size][sample_size][edge_probability]:
                    tabular_groups[node_size][sample_size][edge_probability][n_domains] = {}
                if function_type not in tabular_groups[node_size][sample_size][edge_probability][n_domains]:
                    tabular_groups[node_size][sample_size][edge_probability][n_domains][function_type] = []
                
                config_signature = f"tab_nodes{node_size}_samples{sample_size}_edgeprob{edge_probability}_domains{n_domains}_func{function_type}"
                tabular_groups[node_size][sample_size][edge_probability][n_domains][function_type].append(
                    (dataset_name, dataset_dir, config, config_signature)
                )
        
        # Track timeouts
        timeout_config = {}  # Store configurations where timeout occurred
        
        # Process tabular datasets
        if tabular_groups:
            print("Processing tabular datasets...")
            sorted_node_sizes = sorted(tabular_groups.keys())
            
            for node_size in sorted_node_sizes:
                sorted_sample_sizes = sorted(tabular_groups[node_size].keys())
                
                for sample_size in sorted_sample_sizes:
                    sorted_edge_probs = sorted(tabular_groups[node_size][sample_size].keys())
                    
                    for edge_prob in sorted_edge_probs:
                        sorted_domains = sorted(tabular_groups[node_size][sample_size][edge_prob].keys())
                        
                        for n_domains in sorted_domains:
                            sorted_functions = sorted(tabular_groups[node_size][sample_size][edge_prob][n_domains].keys())
                            
                            for function_type in sorted_functions:
                                datasets = tabular_groups[node_size][sample_size][edge_prob][n_domains][function_type]
                                
                                for dataset_name, dataset_dir, config, config_signature in datasets:
                                    # Skip if we've already had a timeout for this specific configuration
                                    if config_signature in timeout_config:
                                        print(f"Skipping {dataset_name} as timeout occurred for similar configuration: {config_signature}")
                                        skip_result = {
                                            "success": False,
                                            "error": f"Skipped due to previous timeout with similar configuration: {config_signature}",
                                            "data_config": config,
                                            "algorithm": next(iter(self.algorithms.keys())),
                                            "hyperparams": self.hyperparams,
                                            "algorithm_id": self.algo_id,
                                            "dataset": dataset_name
                                        }
                                        if self.algo_id not in all_results:
                                            all_results[self.algo_id] = []
                                        all_results[self.algo_id].append(skip_result)
                                        self._save_result(self.algo_id, skip_result)
                                        continue
                                    
                                    try:
                                        data, true_graph, _ = self._load_dataset(dataset_dir)
                                        
                                        for algo_name, algo_class in self.algorithms.items():
                                            algo_instance = algo_class(self.hyperparams)
                                            print(f"Running {self.algo_id} on {dataset_name}")
                                            result = self._run_single_experiment(algo_instance, data, true_graph, config)
                                            result['algorithm'] = algo_name
                                            result['hyperparams'] = self.hyperparams
                                            result['algorithm_id'] = self.algo_id
                                            result['dataset'] = dataset_name
                                            
                                            if self.algo_id not in all_results:
                                                all_results[self.algo_id] = []
                                            all_results[self.algo_id].append(result)
                                            self._save_result(self.algo_id, result)
                                            
                                            # Check if timeout occurred
                                            if not result['success'] and 'timed out' in str(result.get('error', '')).lower():
                                                timeout_config[config_signature] = True
                                                break
                                    
                                    except Exception as e:
                                        print(f"Error processing {dataset_name}: {str(e)}")
                                        continue
                                
                                if self.debug_mode:
                                    break  # Only test the first function type in debug mode
                            
                            if self.debug_mode:
                                break  # Only test the first n_domains in debug mode
                        
                        if self.debug_mode:
                            break  # Only test the first edge probability in debug mode
                    
                    if self.debug_mode:
                        break  # Only test the first sample size in debug mode
                
                if self.debug_mode:
                    break  # Only test the first node size in debug mode
        # Process time series datasets
        if timeseries_groups:
            print("Processing time series datasets...")
            sorted_node_sizes = sorted(timeseries_groups.keys())
            
            for node_size in sorted_node_sizes:
                sorted_lags = sorted(timeseries_groups[node_size].keys())
                
                for lag in sorted_lags:
                    sorted_degree_inters = sorted(timeseries_groups[node_size][lag].keys())
                    
                    for degree_inter in sorted_degree_inters:
                        sorted_degree_intras = sorted(timeseries_groups[node_size][lag][degree_inter].keys())
                        
                        for degree_intra in sorted_degree_intras:
                            datasets = timeseries_groups[node_size][lag][degree_inter][degree_intra]
                            
                            # Sort datasets by sample size
                            datasets.sort(key=lambda x: x[4])  # Sort by sample_size (5th element)
                            
                            for dataset_name, dataset_dir, config, config_signature, _ in datasets:
                                # Skip if we've already had a timeout for this specific configuration
                                if config_signature in timeout_config:
                                    print(f"Skipping {dataset_name} as timeout occurred for similar configuration: {config_signature}")
                                    skip_result = {
                                        "success": False,
                                        "error": f"Skipped due to previous timeout with similar configuration: {config_signature}",
                                        "data_config": config,
                                        "algorithm": next(iter(self.algorithms.keys())),
                                        "hyperparams": self.hyperparams,
                                        "algorithm_id": self.algo_id,
                                        "dataset": dataset_name
                                    }
                                    if self.algo_id not in all_results:
                                        all_results[self.algo_id] = []
                                    all_results[self.algo_id].append(skip_result)
                                    self._save_result(self.algo_id, skip_result)
                                    continue
                                
                                try:
                                    data, true_graph, _ = self._load_dataset(dataset_dir)
                                    
                                    for algo_name, algo_class in self.algorithms.items():
                                        algo_instance = algo_class(self.hyperparams)
                                        print(f"Running {self.algo_id} on {dataset_name}")
                                        result = self._run_single_experiment(algo_instance, data, true_graph, config)
                                        result['algorithm'] = algo_name
                                        result['hyperparams'] = self.hyperparams
                                        result['algorithm_id'] = self.algo_id
                                        result['dataset'] = dataset_name
                                        
                                        if self.algo_id not in all_results:
                                            all_results[self.algo_id] = []
                                        all_results[self.algo_id].append(result)
                                        self._save_result(self.algo_id, result)
                                        
                                        # Check if timeout occurred
                                        if not result['success'] and 'timed out' in str(result.get('error', '')).lower():
                                            timeout_config[config_signature] = True
                                            break
                                
                                except Exception as e:
                                    print(f"Error processing {dataset_name}: {str(e)}")
                                    continue
                            
                            if self.debug_mode:
                                break  # Only test the first degree_intra in debug mode
                        
                        if self.debug_mode:
                            break  # Only test the first degree_inter in debug mode
                    
                    if self.debug_mode:
                        break  # Only test the first lag in debug mode
                
                if self.debug_mode:
                    break  # Only test the first node size in debug mode
        
        print(f"Benchmark results saved to: {self.output_dir}")
        
def test_timeout_mechanism():
    """Test case to verify the timeout kill mechanism works properly"""
    import signal
    import time
    import multiprocessing
    
    # Define a function that will run for longer than the timeout
    def long_running_function():
        print("Starting long-running function...")
        time.sleep(10)  # This should be killed by the timeout
        print("This should not be printed!")
        return "Completed"
    
    # Function to test timeout in a separate process
    def test_in_process():
        # Set timeout to 2 seconds
        timeout_seconds = 2
        
        # Set up the timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
        
        # Set the timeout handler
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            print(f"Setting timeout to {timeout_seconds} seconds...")
            signal.alarm(timeout_seconds)
            
            # Run the function that should be killed
            result = long_running_function()
            
            # If we get here, the timeout didn't work
            print("ERROR: Timeout mechanism failed!")
            return False
            
        except TimeoutError as e:
            print(f"SUCCESS: Timeout mechanism worked correctly: {e}")
            return True
        finally:
            # Disable the alarm and restore original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    
    # Test in single process first
    print("Testing timeout in single process mode...")
    single_process_result = test_in_process()
    
    # Now test in multiprocessing environment
    print("\nTesting timeout in multiprocessing environment...")
    process = multiprocessing.Process(target=test_in_process)
    process.start()
    
    # Wait for the process to complete with a timeout
    process.join(timeout=5)
    
    # Check if the process is still alive (it shouldn't be)
    if process.is_alive():
        print("ERROR: Multiprocessing timeout test failed - process still running")
        process.terminate()
        process.join()
        multiprocessing_result = False
    else:
        print("SUCCESS: Multiprocessing timeout test completed")
        multiprocessing_result = True
    
    return single_process_result and multiprocessing_result

def test_timeout_series_handling():
    """Test that the new timeout handling mechanism correctly skips to next series after timeout."""
    try:
        print("\n=== Testing Series-based Timeout Handling ===")
        
        # Create a mock algorithm class for testing timeouts
        class MockTimeoutAlgorithm:
            def __init__(self, hyperparams=None):
                self.name = "MockTimeoutAlgo"
                self.params = hyperparams or {}
                # Configure which conditions should trigger timeouts
                self.timeout_edge_prob_threshold = self.params.get("timeout_edge_prob", 0.5)
                self.timeout_nodes_threshold = self.params.get("timeout_nodes", 50)
                self.default_runtime = self.params.get("default_runtime", 0.5)  # seconds
                
            def fit(self, data):
                """Mock algorithm that will timeout under specific conditions."""
                # Get number of nodes from data dimensions
                n_nodes = data.shape[1]
                
                # Check if domain_index is in the data
                if 'domain_index' in data.columns:
                    n_nodes -= 1  # Adjust for domain_index column
                    
                # Extract edge_probability from data's attributes if available
                edge_probability = getattr(data, 'edge_probability', 0.0)
                
                print(f"  MockAlgo running on {n_nodes} nodes with edge_prob {edge_probability}")
                
                # Simulate timeout based on edge probability
                if edge_probability > self.timeout_edge_prob_threshold:
                    print(f"  Simulating timeout due to dense graph (edge_prob={edge_probability})")
                    time.sleep(5)  # Sleep long enough to trigger timeout
                    return None, None, None
                    
                # Simulate timeout based on node count
                if n_nodes > self.timeout_nodes_threshold:
                    print(f"  Simulating timeout due to large graph (nodes={n_nodes})")
                    time.sleep(5)  # Sleep long enough to trigger timeout
                    return None, None, None
                    
                # For successful runs, simulate some work
                time.sleep(self.default_runtime)
                
                # Return mock results (random adjacency matrix)
                adj_matrix = np.random.randint(0, 2, size=(n_nodes, n_nodes))
                np.fill_diagonal(adj_matrix, 0)  # No self-loops
                
                return adj_matrix, {"info": "Mock algorithm result"}, None
                
            def get_params(self):
                return self.params
        
        # Override the algorithm loading mechanism to use our mock
        def mock_load_algorithms(self):
            print("  Using mock algorithm loader")
            return {"MockTimeoutAlgo": MockTimeoutAlgorithm}
        
        # Patch the BenchmarkRunner to use a very short timeout for testing
        original_run_single_experiment = BenchmarkRunner._run_single_experiment
        
        def mock_run_single_experiment(self, algo_instance, data, true_graph, config):
            # Add edge_probability as an attribute to the data for the mock algorithm to access
            data.edge_probability = config.get('edge_probability', 0.0)
            # Use a much shorter timeout for testing (2 seconds)
            timeout_seconds = 2
            
            start_time = time.time()
            
            try:
                # fill the missing value if it is existed
                data = data.fillna(0)

                # Skip if the algorithm is not designed for heterogeneous data
                if 'domain_index' in data.columns and "CDNOD" not in algo_instance.name:
                    data = data.drop(columns=['domain_index'])
                if 'domain_index' not in data.columns and "CDNOD" in algo_instance.name:
                    data["domain_index"] = np.ones(data.shape[0])

                # Use threading with a timeout instead of multiprocessing
                import threading
                import queue
                
                result_queue = queue.Queue()
                
                def run_algorithm():
                    try:
                        adj_matrix, info, _ = algo_instance.fit(data)
                        result_queue.put((adj_matrix, info))
                    except Exception as e:
                        result_queue.put(e)
                
                # Start the algorithm in a thread
                thread = threading.Thread(target=run_algorithm)
                thread.daemon = True
                thread.start()
                
                # Wait for the thread to complete or timeout
                thread.join(timeout_seconds)
                
                # Check if the thread is still running after timeout
                if thread.is_alive():
                    # We can't forcefully terminate a thread in Python,
                    # but we can indicate a timeout occurred
                    raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
                
                # Get the result from the queue
                if result_queue.empty():
                    raise Exception("Process completed but no result was returned")
                
                result = result_queue.get()
                
                # Check if the result is an exception
                if isinstance(result, Exception):
                    raise result
                
                adj_matrix, info = result
                
                runtime = time.time() - start_time
                metrics = self.evaluator.compute_metrics(true_graph, adj_matrix)
                
                result = {
                    "success": True,
                    "runtime": runtime,
                    "metrics": metrics,
                    "algorithm_params": algo_instance.get_params(),
                    "data_config": config
                }
                
            except TimeoutError as e:
                result = {
                    "success": False,
                    "error": str(e),
                    "data_config": config
                }
            except Exception as e:
                result = {
                    "success": False,
                    "error": str(e),
                    "data_config": config
                }
                
            return result
        
        # Apply the monkey patches
        print("  Applying method patches")
        original_load_algorithms = BenchmarkRunner._load_algorithms
        BenchmarkRunner._load_algorithms = mock_load_algorithms
        BenchmarkRunner._run_single_experiment = mock_run_single_experiment
        
        try:
            # Run the benchmark with our mock algorithm
            hyperparams = {
                "timeout_edge_prob": 0.5,  # Timeout on edge probability > 0.5
                "timeout_nodes": 100,      # Timeout on node count > 100
                "default_runtime": 0.1     # Fast runtime for successful runs
            }
            
            # Create a temporary output directory for the test
            import tempfile
            print("  Creating temporary directory for test output")
            with tempfile.TemporaryDirectory() as temp_dir:
                # Set up configuration for the test
                print("  Setting up test configuration")
                
                # Make sure we have the benchmarking config
                try:
                    from causal_discovery.tests.benchmarking_config import CONFIG, set_config
                    print("  Successfully imported benchmarking_config")
                except ImportError as e:
                    print(f"  Error importing benchmarking_config: {e}")
                    # Try to create a minimal configuration
                    CONFIG = {
                        "data_dir": "simulated_data/heavy_benchmarking_v6",
                        "output_dir": "benchmark_results",
                        "algorithms": ["MockTimeoutAlgo"]
                    }
                    def set_config(config):
                        global CONFIG
                        CONFIG = config
                
                # Save original config and create a test config
                original_config = CONFIG.copy() if 'CONFIG' in locals() else {}
                
                # Modify config for testing
                test_config = original_config.copy()
                test_config["output_dir"] = temp_dir
                test_config["algorithms"] = ["MockTimeoutAlgo"]
                print(f"  Test config: {test_config}")
                
                if 'set_config' in locals():
                    set_config(test_config)
                
                # Create and run the benchmark
                print("  Creating benchmark runner")
                runner = BenchmarkRunner(algorithm="MockTimeoutAlgo", hyperparams=hyperparams, debug_mode=True)
                
                # Print info about the datasets found
                print(f"  Found {len(runner.dataset_index)} datasets for testing")
                
                # Extract a small subset for testing if there are a lot
                if len(runner.dataset_index) > 20:
                    runner.dataset_index = runner.dataset_index[:20]
                    print(f"  Using subset of {len(runner.dataset_index)} datasets for testing")
                
                # Track experiment series completion
                completed_series = []
                skipped_series = []
                
                # Get list of varied parameters
                varied_parameters = {}
                for dataset_name, dataset_dir, config in runner.dataset_index:
                    # Extract parameters for series identification
                    if "n_nodes" in config:
                        if "nodes" not in varied_parameters:
                            varied_parameters["nodes"] = set()
                        varied_parameters["nodes"].add(config["n_nodes"])
                    if "n_samples" in config:
                        if "samples" not in varied_parameters:
                            varied_parameters["samples"] = set()
                        varied_parameters["samples"].add(config["n_samples"])
                    if "edge_probability" in config:
                        if "edge_prob" not in varied_parameters:
                            varied_parameters["edge_prob"] = set()
                        varied_parameters["edge_prob"].add(config.get("edge_probability", 0))
                
                print(f"  Found experiment series with parameters: {list(varied_parameters.keys())}")
                for param, values in varied_parameters.items():
                    print(f"    {param}: {sorted(values)}")
                    
                # Run the experiments
                print("  Running experiments...")
                runner.run_all_experiments()
                
                # Check the results file to see what completed
                result_file = os.path.join(temp_dir, f"MockTimeoutAlgo_timeout_edge_prob=0.5_timeout_nodes=100_default_runtime=0.1_results.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    # Count successful and failed experiments
                    success_count = sum(1 for r in results if r.get('success', False))
                    timeout_count = sum(1 for r in results if not r.get('success', False) and 'timed out' in str(r.get('error', '')).lower())
                    
                    print(f"\nTest Results:")
                    print(f"  Total experiments: {len(results)}")
                    print(f"  Successful runs: {success_count}")
                    print(f"  Timeouts: {timeout_count}")
                    
                    # Check for experiments from different parameter series
                    datasets_run = [r.get('dataset', '') for r in results]
                    
                    # Check if we have results from multiple series despite timeouts
                    has_edge_prob_results = any('edgeprob0.1' in ds for ds in datasets_run)
                    has_large_node_results = any('nodes100' in ds or 'nodes200' in ds for ds in datasets_run)
                    
                    print("\nSeries coverage:")
                    print(f"  Has edge probability series results: {has_edge_prob_results}")
                    print(f"  Has large node size series results: {has_large_node_results}")
                    
                    # Determine if the test passed
                    if has_edge_prob_results and has_large_node_results:
                        print("\n✅ TEST PASSED: Multiple series were processed despite timeouts")
                    else:
                        print("\n❌ TEST FAILED: Some series were not processed")
                else:
                    print(f"\n❌ TEST FAILED: No results file was created at {result_file}")
                
                # Restore original config
                if 'set_config' in locals():
                    set_config(original_config)
        
        finally:
            # Restore the original methods
            print("  Restoring original methods")
            BenchmarkRunner._load_algorithms = original_load_algorithms
            BenchmarkRunner._run_single_experiment = original_run_single_experiment
        
        print("=== Timeout Series Handling Test Complete ===\n")
    except Exception as e:
        import traceback
        print(f"ERROR in test_timeout_series_handling: {e}")
        traceback.print_exc()

def main():
    """Parse command line arguments and run benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run algorithm benchmarks')
    parser.add_argument('--algorithm', type=str, help='Algorithm name to benchmark')
    parser.add_argument('--params', type=str, help='JSON string of algorithm hyperparameters')
    parser.add_argument('--param_file', type=str, help='Path to JSON file containing algorithm hyperparameters')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (only first dataset)')
    parser.add_argument('--test', action='store_true', help='Run tests instead of benchmark')
    parser.add_argument('--resume_dir', type=str, help='Directory containing previous results to resume from', default="none")
    
    args = parser.parse_args()

    if args.resume_dir == "none":
        args.resume_dir = None
    
    # Run tests if requested
    if args.test:
        print("Running timeout mechanism tests...")
        if test_timeout_mechanism():
            print("Timeout mechanism tests passed!")
        else:
            print("Timeout mechanism tests failed!")
            
        print("\nRunning series timeout handling tests...")
        test_timeout_series_handling()
        return
    
    if not args.algorithm:
        parser.error("Algorithm name is required")
        
    # Load parameters from either command line or file
    hyperparams = None
    if args.param_file:
        try:
            with open(args.param_file, 'r') as f:
                hyperparams = json.load(f)
            print(f"Loaded hyperparameters from file: {hyperparams}")
        except Exception as e:
            print(f"Error loading hyperparameters from file: {e}")
            return
    elif args.params:
        try:
            hyperparams = json.loads(args.params)
            print(f"Loaded hyperparameters from command line: {hyperparams}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON hyperparameters: {e}")
            return
    
    # Run benchmark
    print(f"Running benchmark for algorithm: {args.algorithm}")
    benchmark = BenchmarkRunner(args.algorithm, hyperparams, debug_mode=args.debug, resume_dir=args.resume_dir)
    benchmark.run_all_experiments()

if __name__ == '__main__':
    main()
