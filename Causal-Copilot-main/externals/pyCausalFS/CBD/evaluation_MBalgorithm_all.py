# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/18 15:38
desc:
"""
import numpy as np
import pandas as pd
import time
import os
import sys

# Add parent directory to Python path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from MBs.common.realMB import realMB
from MBs.MMMB.MMMB import MMMB
from MBs.HITON.HITON_MB import HITON_MB
from MBs.PCMB.PCMB import PCMB
from MBs.IPCMB.IPCMB import IPC_MB
from MBs.GSMB import GSMB
from MBs.IAMB import IAMB
from MBs.fast_IAMB import fast_IAMB
from MBs.inter_IAMB import inter_IAMB
from MBs.IAMBnPC import IAMBnPC
from MBs.interIAMBnPC import interIAMBnPC
from MBs.KIAMB import KIAMB
from MBs.STMB import STMB
from MBs.BAMB import BAMB
from MBs.FBEDk import FBED
from MBs.MBOR import MBOR
from MBs.LCMB import LRH

# Import SSD algorithms
from SSD.MBs.SLLMB import SLL
from SSD.MBs.S2TMB import S2TMB, S2TMB_p

# Import LSL algorithms
# from LSL.MBs.PCDbyPCD import PCDbyPCD  # too slow
# from LSL.MBs.MBbyMB import MBbyMB  # too slow
# from LSL.MBs.CMB.CMB import CMB  # too slow

def evaluation(
        method,
        path,
        all_number_Para,
        target_list,
        real_graph_path,
        is_discrete,
        alpha=0.01,
        k=1):

    # pre_set variables is zero
    Precision = 0
    Recall = 0
    F1 = 0
    Distance = 0
    use_time = 0
    ci_number = 0
    realmb, realpc = realMB(all_number_Para, real_graph_path)
    length_targets = len(target_list)
    completePath = path
    data = pd.read_csv(completePath)
    number, kVar = np.shape(data)
    ResMB = [[]] * length_targets
    # print("\ndata set is: " + str(m+1) + ".csv")
    for i, target in enumerate(target_list):
        # print("target is: " + str(target))
        if method == "MMMB":
            start_time = time.process_time()
            MB, ci_num = MMMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "IAMB":
            start_time = time.process_time()
            MB, ci_num = IAMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "KIAMB":
            start_time = time.process_time()
            MB, ci_num = KIAMB(data, target, alpha, k, is_discrete)
            print(MB)
            end_time = time.process_time()
        elif method == "IAMBnPC":
            start_time = time.process_time()
            MB, ci_num = IAMBnPC(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "inter_IAMB":
            start_time = time.process_time()
            MB, ci_num = inter_IAMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "interIAMBnPC":
            start_time = time.process_time()
            MB, ci_num = interIAMBnPC(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "fast_IAMB":
            start_time = time.process_time()
            MB, ci_num = fast_IAMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "GSMB":
            start_time = time.process_time()
            MB, ci_num = GSMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "HITON_MB":
            start_time = time.process_time()
            MB, ci_num = HITON_MB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "PCMB":
            start_time = time.process_time()
            MB, ci_num = PCMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "IPCMB":
            start_time = time.process_time()
            MB, ci_num = IPC_MB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "STMB":
            start_time = time.process_time()
            MB, ci_num = STMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "IAMBnPC":
            start_time = time.process_time()
            MB, ci_num = IAMBnPC(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "BAMB":
            start_time = time.process_time()
            MB, ci_num = BAMB(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "FBEDk":
            start_time = time.process_time()
            MB, ci_num = FBED(data, target, k, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "MBOR":
            start_time = time.process_time()
            MB, ci_num = MBOR(data, target, alpha, is_discrete)
            end_time = time.process_time()
        elif method == "LRH":
            start_time = time.process_time()
            MB, ci_num = LRH(data, target, alpha, is_discrete)
            end_time = time.process_time()
        # Add SSD algorithms
        elif method == "SLL":
            start_time = time.process_time()
            _, MB = SLL(data, target)
            ci_num = 0  # SSD algorithms don't return CI test count
            end_time = time.process_time()
        elif method == "S2TMB":
            start_time = time.process_time()
            _, MB = S2TMB(data, target)
            ci_num = 0
            end_time = time.process_time()
        elif method == "S2TMB_p":
            start_time = time.process_time()
            _, MB = S2TMB_p(data, target)
            ci_num = 0
            end_time = time.process_time()
        # Add LSL algorithms
        elif method == "PCDbyPCD":
            start_time = time.process_time()
            parents, children, undirected = PCDbyPCD(data, target, alpha, is_discrete)
            MB = list(set(parents + children + undirected))
            ci_num = 0
            end_time = time.process_time()
        elif method == "MBbyMB":
            start_time = time.process_time()
            parents, children, undirected = MBbyMB(data, target, alpha, is_discrete)
            MB = list(set(parents + children + undirected))
            ci_num = 0
            end_time = time.process_time()
        elif method == "CMB":
            start_time = time.process_time()
            parents, children, undirected = CMB(data, target, alpha, is_discrete)
            MB = list(set(parents + children + undirected))
            ci_num = 0
            end_time = time.process_time()
        else:
            raise Exception("method input error!")

        use_time += (end_time - start_time)
        ResMB[i] = MB
        ci_number += ci_num

    for n, target in enumerate(target_list):
        # print("target is: " + str(target) + " , n is: " + str(n))
        true_positive = list(
            set(realmb[target]).intersection(set(ResMB[n])))
        length_true_positive = len(true_positive)
        length_RealMB = len(realmb[target])
        length_ResMB = len(ResMB[n])
        if length_RealMB == 0:
            if length_ResMB == 0:
                precision = 1
                recall = 1
                distance = 0
                F1 += 1
            else:
                F1 += 0
                precision = 0
                distance = 2 ** 0.5
                recall = 0
        else:
            if length_ResMB != 0:
                precision = length_true_positive / length_ResMB
                recall = length_true_positive / length_RealMB
                distance = ((1 - precision) ** 2 + (1 - recall) ** 2) ** 0.5
                if precision + recall != 0:
                    F1 += 2 * precision * recall / (precision + recall)
            else:
                F1 += 0
                precision = 0
                recall = 0
                distance = 2 ** 0.5
        Distance += distance
        Precision += precision
        Recall += recall

        # print("current average Precision is: " + str(Precision / ((m+1) * (numberPara))))
        # print("current average Recall is: " + str(Recall / ((m+1) * (numberPara))))

    commonDivisor = length_targets * 1

    # 标准差

    return F1 / commonDivisor, Precision / commonDivisor, Recall / commonDivisor, Distance / \
        commonDivisor, ci_number / commonDivisor, use_time / commonDivisor

def evaluate_algorithm(method_info, data_path, num_para, list_target, real_graph_path, isdiscrete, alpha):
    method, uses_k, k_value = method_info
    results = {}
    try:
        print(f"\nEvaluating {method}...")
        
        # Run evaluation
        if uses_k:
            F1, Precision, Recall, Distance, ci_number, time = evaluation(
                method, data_path, num_para, list_target, real_graph_path, 
                isdiscrete, alpha, k_value)
        else:
            F1, Precision, Recall, Distance, ci_number, time = evaluation(
                method, data_path, num_para, list_target, real_graph_path,
                isdiscrete, alpha)

        results = {
            'method': method,
            'F1': F1,
            'Precision': Precision,
            'Recall': Recall,
            'Distance': Distance,
            'CI Tests': ci_number,
            'Time': time,
            'error': None
        }
        print(f"Finished {method} evaluation")

    except Exception as e:
        results = {
            'method': method,
            'error': str(e)
        }

    return results

# test main
if __name__ == '__main__':
    import multiprocessing as mp
    from functools import partial

    # Initialize common parameters
    real_graph_path = "./data/child_graph.txt"
    data_path = "./data/Child_s500_v1.csv"
    alpha = 0.05
    isdiscrete = True

    # Get data dimensions
    _, num_para = np.shape(pd.read_csv(data_path))
    list_target = [i for i in range(num_para)]

    # List of all algorithms to evaluate
    algorithms = [
        ("MMMB", False, 0),
        ("HITON_MB", False, 0), 
        ("PCMB", False, 0),
        ("IPC_MB", False, 0),
        ("GSMB", False, 0),
        ("IAMB", False, 0),
        ("fast_IAMB", False, 0),
        ("inter_IAMB", False, 0),
        ("IAMBnPC", False, 0),
        ("interIAMBnPC", False, 0),
        ("KIAMB", True, 0.8),
        ("STMB", False, 0),
        ("BAMB", False, 0),
        ("FBED", False, 0),
        ("MBOR", False, 0),
        ("LRH", False, 0),
        # Add SSD algorithms
        ("SLL", False, 0),
        ("S2TMB", False, 0),
        ("S2TMB_p", False, 0),
        # Add LSL algorithms
        # ("PCDbyPCD", False, 0),
        # ("MBbyMB", False, 0),
        # ("CMB", False, 0)
    ]

    # Create a pool of workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Create partial function with fixed arguments
    eval_func = partial(evaluate_algorithm,
                       data_path=data_path,
                       num_para=num_para,
                       list_target=list_target,
                       real_graph_path=real_graph_path,
                       isdiscrete=isdiscrete,
                       alpha=alpha)

    # Run evaluations in parallel
    results = pool.map(eval_func, algorithms)
    pool.close()
    pool.join()

    # Calculate combined scores for all methods
    scored_results = []
    max_time = max(r['Time'] for r in results if r['error'] is None)
    
    for result in results:
        if result['error'] is None:
            # Normalize time score (lower is better)
            time_score = 1 - (result['Time'] / max_time)
            # Calculate weighted score
            combined_score = 0.7 * result['F1'] + 0.3 * time_score
            scored_results.append({
                'method': result['method'],
                'score': combined_score,
                'F1': result['F1'],
                'Time': result['Time']
            })
    
    # Sort by combined score and get top 5
    top_5 = sorted(scored_results, key=lambda x: x['score'], reverse=True)[:5]

    # Write results to file
    with open(r".\output\all_results.txt", "w") as results_file:
        for result in results:
            method = result['method']
            if result.get('error') is None:
                results_file.write(f"\n{method} Results:\n")
                results_file.write(f"F1: {result['F1']:.2f}\n")
                results_file.write(f"Precision: {result['Precision']:.2f}\n")
                results_file.write(f"Recall: {result['Recall']:.2f}\n")
                results_file.write(f"Distance: {result['Distance']:.2f}\n")
                results_file.write(f"CI Tests: {result['CI Tests']:.2f}\n")
                results_file.write(f"Time: {result['Time']:.2f}s\n")

                # Also print to console
                print(f"\n{method} Results:")
                print(f"F1: {result['F1']:.2f}")
                print(f"Precision: {result['Precision']:.2f}")
                print(f"Recall: {result['Recall']:.2f}")
                print(f"Distance: {result['Distance']:.2f}")
                print(f"CI Tests: {result['CI Tests']:.2f}")
                print(f"Time: {result['Time']:.2f}s")
            else:
                error_msg = f"Error evaluating {method}: {result['error']}"
                results_file.write(f"\n{error_msg}\n")
                print(error_msg)
            
            results_file.write("-"*50 + "\n")

        # Write top 5 to file
        results_file.write("\nTOP 5 METHODS (weighted by F1 70%, Time 30%):\n")
        results_file.write("-"*50 + "\n")
        for i, result in enumerate(top_5, 1):
            results_file.write(f"{i}. {result['method']}\n")
            results_file.write(f"   Combined Score: {result['score']:.3f}\n")
            results_file.write(f"   F1 Score: {result['F1']:.2f}\n")
            results_file.write(f"   Time: {result['Time']:.2f}s\n")
            results_file.write("-"*50 + "\n")
        
        # Also print to console
        print("\nTOP 5 METHODS (weighted by F1 70%, Time 30%):")
        print("-"*50)
        for i, result in enumerate(top_5, 1):
            print(f"{i}. {result['method']}")
            print(f"   Combined Score: {result['score']:.3f}")
            print(f"   F1 Score: {result['F1']:.2f}")
            print(f"   Time: {result['Time']:.2f}s")
            print("-"*50)
