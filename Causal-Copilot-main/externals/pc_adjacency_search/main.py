import argparse
import numpy as np
import os
from gpu_ci import gpu_single, gpu_row
from cpu_ci import cpu_adj
from tigramite.independence_tests import CMIknn
import globals

import gpucmiknn

import sys

# packages for shd
import cdt
import networkx

parser = argparse.ArgumentParser("Parallel PC Algorithm on Mixed Data")
parser.add_argument('-a', '--alpha', help='Signficance Level used for CI Test', default=0.05, type=float)
parser.add_argument('-l', '--level', help='Maximum Level of the Run', default=None, type=int)
parser.add_argument('--process_count', help='Number of Processes used in the Run - Main process excluded', default=2, type=int)
parser.add_argument('--permutations', help='Number of Permutations used for a CI Test', default=50, type=int)
parser.add_argument('--par_strategy', help='Specify Parallelization Strategy: 1 -> CPU-Based | 2 -> Single GPU | 3 Rowwise GPU', type=int)
parser.add_argument('-s', '--start_level', help='Execute starting with level s for benchmarking tests with certain sepset size ', default=0, type=int)
parser.add_argument('-k', '--kcmi', help='KNN during cmi estimation. Default is adaptive, which is fixed to 7 for mesner, samples*0.2 for Runge or sqrt(samples)/5 for GKOV', type=int)
parser.add_argument('-b', '--block_size', help='Number of separation sets that are blocked together during rowwise processing of lvl > 0. Default is None and calculates the factor on encountering memory pressure due to large numbers of separation set candidates', default=None, type=int)
required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--input_file', help='Input File Name', required=True)
if __name__ == '__main__':
    # Call Parallel PC Algorithm Skeleton Estimation
    args = parser.parse_args()

    # Generate sample data with linear relationships
    np.random.seed(42)
    n_samples = 1000
    X1 = np.random.normal(0, 1, n_samples)
    X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
    X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
    X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
    X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
    X6 = 0.3 * X4 + 0.2 * X5 + np.random.normal(0, 0.3, n_samples)
    X7 = 0.5 * X1 + 0.3 * X6 + np.random.normal(0, 0.4, n_samples)
    X8 = 0.6 * X3 + 0.4 * X7 + np.random.normal(0, 0.3, n_samples)
    X9 = 0.2 * X6 + 0.7 * X8 + np.random.normal(0, 0.25, n_samples)
    X10 = 0.8 * X9 + 0.1 * X5 + np.random.normal(0, 0.2, n_samples)

    # Ground truth graph (10×10 adjacency matrix)
    gt_graph = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X1
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X2
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # X3
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # X4
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # X5
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # X6
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # X7
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # X8
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # X9
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]   # X10
    ])

    # Combine the data into a single array
    input_data = np.column_stack([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10])
    print(input_data.shape)
    
    np.set_printoptions(threshold=sys.maxsize)
    globals.init()
    globals.alpha = args.alpha
    globals.vertices = input_data.shape[1]
    globals.permutations = args.permutations
    globals.process_count = args.process_count
    globals.max_sepset_size = input_data.shape[1] - 2 if args.level is None else min(args.level, input_data.shape[1] - 2)

    globals.start_level = args.start_level

    globals.split_size = args.block_size
    if args.kcmi:
        globals.k_cmi = args.kcmi
    else:
        globals.k_cmi = 'adaptive'

    # Exception for Runge test for paper
    if (args.par_strategy == 1):
        globals.ci_test = CMIknn(knn=int(input_data.shape[0] * 0.2) if globals.k_cmi == 'adaptive' else globals.k_cmi,
                 shuffle_neighbors=globals.k_perm,
                 significance='shuffle_test',
                 sig_samples = globals.permutations,
                 workers=-1)
    else:
        globals.ci_test = None

    print("Starting level:", globals.start_level)

    skeleton = None
    sepsets = None
    if args.par_strategy == 2 or args.par_strategy == 3:
        globals.gpu_free_mem = gpucmiknn.init_gpu()
        print("init cuda")

    if (args.par_strategy == 3):
        print("Execution Rowwise on GPU")
        skeleton, sepsets = gpu_row(input_data)
    if (args.par_strategy == 2):
        print("Execution Single on GPU")
        skeleton, sepsets = gpu_single(input_data)
    if (args.par_strategy == 1):
        print("CPU only")
        skeleton, sepsets = cpu_adj(input_data)

    print(skeleton, sepsets)
    
