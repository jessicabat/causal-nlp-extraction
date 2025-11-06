# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/9 20:10
desc: 
"""

import numpy as np
import os
import sys

causal_learn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'causal-learn')
sys.path.append(causal_learn_dir)

import causallearn.utils.cit as cit
from joblib import Parallel, delayed

# Change signature: remove is_discrete, add indep_test and n_jobs
def inter_IAMB(data, target, alpha, indep_test='fisherz', n_jobs=1):
    number, kVar = np.shape(data)
    ci_number = 0
    MB = []
    circulateFlag = True
    removeSet = []
    rmNumberSet = [0 for i in range(kVar)]
    
    # Initialize CIT
    cond_indep_test = cit.CIT(data, indep_test)
    
    while circulateFlag:
        circulateFlag = False
        dep_temp = -float("inf")
        pval_temp = 1
        max_s = None

        variables = [i for i in range(kVar) if i != target and i not in MB and i not in removeSet]

        # Growing phase: if n_jobs>1 run the tests in parallel.
        if n_jobs > 1 and variables:
            results = Parallel(n_jobs=n_jobs)(
                delayed(cond_indep_test)(target, s, MB) for s in variables
            )
            for s, pval_gp in zip(variables, results):
                ci_number += 1
                dep_gp = -pval_gp
                if dep_gp > dep_temp:
                    dep_temp = dep_gp
                    max_s = s
                    pval_temp = pval_gp
        else:
            for s in variables:
                ci_number += 1
                pval_gp = cond_indep_test(target, s, MB)
                dep_gp = -pval_gp
                if dep_gp > dep_temp:
                    dep_temp = dep_gp
                    max_s = s
                    pval_temp = pval_gp

        if pval_temp <= alpha:
            circulateFlag = True
            MB.append(max_s)

        if not circulateFlag:
            break

        # Shrinking phase
        mb_index = len(MB)
        while mb_index > 0:
            mb_index -= 1
            x = MB[mb_index]
            ci_number += 1
            subsets_Variables = [i for i in MB if i != x]
            pval_sp = cond_indep_test(target, x, subsets_Variables)
            dep_sp = -pval_sp
            if pval_sp > alpha:
                MB.remove(x)
                if x == max_s:
                    break
                rmNumberSet[x] += 1
                if rmNumberSet[x] > 10:
                    removeSet.append(x)

    return list(set(MB)), ci_number


# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 4
# alpha = 0.05
#
# CMB=inter_IAMB(data,target,alpha)
#
# print(CMB)


# F1 is: 0.7603808691308696
# Precision is: 0.7835000000000002
# Recall is: 0.8212083333333333
# time is: 21.37359375


#5000

# F1 is: 0.91
# Precision is: 0.89
# Recall is: 0.95
# Distance is: 0.14
# ci_number is: 120.48
# time is: 68.37

