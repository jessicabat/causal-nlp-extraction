import sys
import os
import time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from causallearn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
from causallearn.utils.PCUtils.SkeletonDiscovery import skeleton_discovery


def are_graphs_equal(cg1, cg2):
    """Compare two causal graphs for equality"""
    # Compare adjacency matrices
    adj1 = cg1.G.graph
    adj2 = cg2.G.graph
    return np.array_equal(adj1, adj2)


def generate_test_data(n_samples=1000, n_features=10, seed=42):
    """Generate a synthetic dataset for testing"""
    np.random.seed(seed)
    
    # Create a matrix with all 1s on the diagonal
    cov = np.eye(n_features)
    
    # Add some off-diagonal elements
    for i in range(n_features):
        for j in range(i+1, n_features):
            if np.random.random() < 0.3:  # 30% chance of correlation
                val = np.random.uniform(0.1, 0.5)  # correlation between 0.1 and 0.5
                cov[i, j] = val
                cov[j, i] = val  # ensure symmetry
    
    # Generate multivariate normal data
    data = np.random.multivariate_normal(mean=np.zeros(n_features), cov=cov, size=n_samples)
    return data


def compare_performance():
    """Compare performance between original and optimized versions"""
    # Test parameters
    alpha = 0.05
    # Use smaller feature sizes for KCI as it's much more computationally intensive
    feature_sizes = [5, 10, 15, 20]
    n_samples = 1000
    
    # Store the execution times
    orig_times = []
    opt_times_2cores = []
    opt_times_4cores = []
    opt_times_all = []
    
    for n_features in feature_sizes:
        print(f"Testing with {n_features} features...")
        data = generate_test_data(n_samples=n_samples, n_features=n_features)
        
        # Run with original function (single core)
        start_time = time.time()
        cg_orig = pc(data, alpha, kci, stable=True, uc_rule=0, uc_priority=2, mvpc=False, correction_name=None, background_knowledge=None, verbose=False, show_progress=False, n_jobs=1)
        orig_time = time.time() - start_time
        orig_times.append(orig_time)
        print(f"Original implementation: {orig_time:.2f} seconds")
        
        # Run with optimized function - 2 cores
        start_time = time.time()
        cg_opt_2 = pc(data, alpha, kci, stable=True, uc_rule=0, uc_priority=2, mvpc=False, correction_name=None, background_knowledge=None, verbose=False, show_progress=False, n_jobs=2)
        opt_time_2 = time.time() - start_time
        opt_times_2cores.append(opt_time_2)
        print(f"Optimized implementation (2 cores): {opt_time_2:.2f} seconds")
        
        # Run with optimized function - 4 cores
        start_time = time.time()
        cg_opt_4 = pc(data, alpha, kci, stable=True, uc_rule=0, uc_priority=2, mvpc=False, correction_name=None, background_knowledge=None, verbose=False, show_progress=False, n_jobs=4)
        opt_time_4 = time.time() - start_time
        opt_times_4cores.append(opt_time_4)
        print(f"Optimized implementation (4 cores): {opt_time_4:.2f} seconds")
        
        # Run with optimized function - all cores
        start_time = time.time()
        cg_opt_all = pc(data, alpha, kci, stable=True, uc_rule=0, uc_priority=2, mvpc=False, correction_name=None, background_knowledge=None, verbose=False, show_progress=False, n_jobs=-1)
        opt_time_all = time.time() - start_time
        opt_times_all.append(opt_time_all)
        print(f"Optimized implementation (all cores): {opt_time_all:.2f} seconds")
        
        # Verify results
        if not are_graphs_equal(cg_orig, cg_opt_all):
            print("WARNING: Results differ between original and optimized versions!")
            
            # Calculate difference percentage in edges
            adj_orig = cg_orig.G.graph
            adj_opt = cg_opt_all.G.graph
            total_edges_orig = np.sum(adj_orig != 0)
            total_edges_opt = np.sum(adj_opt != 0)
            diff_edges = np.sum(adj_orig != adj_opt)
            
            print(f"Original edges: {total_edges_orig}")
            print(f"Optimized edges: {total_edges_opt}")
            print(f"Different edges: {diff_edges}")
            
            if total_edges_orig > 0:
                percent_diff = (diff_edges / total_edges_orig) * 100
                print(f"Percentage difference: {percent_diff:.2f}%")
        else:
            print("Results match between original and optimized versions.")
        
        print("-" * 50)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(feature_sizes, orig_times, 'o-', label='Original')
    plt.plot(feature_sizes, opt_times_2cores, 'o-', label='Optimized (2 cores)')
    plt.plot(feature_sizes, opt_times_4cores, 'o-', label='Optimized (4 cores)')
    plt.plot(feature_sizes, opt_times_all, 'o-', label='Optimized (all cores)')
    plt.xlabel('Number of Features')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison of Skeleton Discovery with KCI Test')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison_kci.png')
    plt.show()
    
    # Calculate speedup
    speedups_2cores = [orig / opt2 for orig, opt2 in zip(orig_times, opt_times_2cores)]
    speedups_4cores = [orig / opt4 for orig, opt4 in zip(orig_times, opt_times_4cores)]
    speedups_all = [orig / opt_all for orig, opt_all in zip(orig_times, opt_times_all)]
    
    print("\nSpeedups with KCI test:")
    for i, n_features in enumerate(feature_sizes):
        print(f"{n_features} features:")
        print(f"  2 cores: {speedups_2cores[i]:.2f}x")
        print(f"  4 cores: {speedups_4cores[i]:.2f}x")
        print(f"  All cores: {speedups_all[i]:.2f}x")


if __name__ == "__main__":
    compare_performance() 