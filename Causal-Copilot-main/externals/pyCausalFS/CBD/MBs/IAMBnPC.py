# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/26 14:31
desc:
"""

import numpy as np
import os
import sys

causal_learn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'causal-learn')
sys.path.append(causal_learn_dir)

import causallearn.utils.cit as cit
from CBD.MBs.common.subsets import subsets

def IAMBnPC(data, target, alpha, indep_test='fisherz', n_jobs=1):
    CMB = []
    ci_number = 0
    number, kVar = np.shape(data)
    
    # Initialize conditional independence test
    cond_indep_test = cit.CIT(data, indep_test)

    while True:
        variDepSet = []
        Svariables = [i for i in range(kVar) if i != target and i not in CMB]
        if n_jobs > 1 and Svariables:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(cond_indep_test)(target, x, CMB) for x in Svariables
            )
            for x, pval in zip(Svariables, results):
                ci_number += 1
                if pval <= alpha:
                    variDepSet.append([x, -pval])
        else:
            for x in Svariables:
                ci_number += 1
                pval = cond_indep_test(target, x, CMB)
                if pval <= alpha:
                    variDepSet.append([x, -pval])

        variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
        if not variDepSet:
            break
        else:
            CMB.append(variDepSet[0][0])

    """shrinking phase"""
    TestMB = sorted(CMB)
    p = len(TestMB)
    DAG = np.ones((1, p))
    size = 0
    continueFlag = True
    max_k = 3
    while continueFlag:
        for y in range(p):
            if DAG[0, y] == 0:
                continue
            conditionAllSet = [i for i in range(p) if i != y and DAG[0, i] == 1]
            conditionSet = subsets(conditionAllSet, size)
            if n_jobs > 1 and conditionSet:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs)(
                    delayed(cond_indep_test)(target, TestMB[y], [TestMB[i] for i in S])
                    for S in conditionSet
                )
                ci_number += len(results)
                if any(r >= alpha for r in results):
                    DAG[0, y] = 0
                    continue
            else:
                for S in conditionSet:
                    ci_number += 1
                    pval_sp = cond_indep_test(target, TestMB[y], [TestMB[i] for i in S])
                    if pval_sp >= alpha:
                        DAG[0, y] = 0
                        break
        size += 1
        continueFlag = False
        if np.sum(DAG[0, :] == 1) >= size and size <= max_k:
            continueFlag = True

    MB = [TestMB[i] for i in range(p) if DAG[0, i] == 1]

    return MB, ci_number


# data = pd.read_csv(
# )
# print("the file read")
#
# target = 11
# alpha = 0.05
#
# MBs = interIAMBnPC(data,target,alpha)
# print("MBs is: "+str(MBs))


# F1 is: 0.8206423576423579
# Precision is: 0.9254166666666666
# Recall is: 0.7850833333333331
# time is: 21.96171875


# 5000

# F1 is: 0.93
# Precision is: 0.99
# Recall is: 0.88
# Distance is: 0.12
# ci_number is: 125.915
# time is: 73.69
