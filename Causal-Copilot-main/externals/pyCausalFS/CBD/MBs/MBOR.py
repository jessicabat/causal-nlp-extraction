#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/8/21 21:19
 @File    : MBOR.py
 """

import numpy as np
from CBD.MBs.common.subsets import subsets
import os
import sys

causal_learn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'causal-learn')
sys.path.append(causal_learn_dir)

import causallearn.utils.cit as cit
from joblib import Parallel, delayed


def IAMB(data, target, alpha, attribute, indep_test='fisherz'):
    CMB = []
    ci_number = 0
    circulate_Flag = True


    # Initialize CIT
    cond_indep_test = cit.CIT(data, indep_test)

    # Forward phase
    while circulate_Flag:
        circulate_Flag = False
        temp_dep = -float("inf")
        y = None
        variables = [i for i in attribute if i != target and i not in CMB]

        for x in variables:
            ci_number += 1
            pival = cond_indep_test(target, x, CMB)
            dep = -pival
            if pival <= alpha and dep > temp_dep:
                temp_dep = dep
                y = x

        if y is not None:
            CMB.append(y)
            circulate_Flag = True

    # Backward phase
    CMB_temp = CMB.copy()
    for x in CMB_temp:
        condition_Variables = [i for i in CMB if i != x]
        ci_number += 1
        pval = cond_indep_test(target, x, condition_Variables)
        if pval > alpha:
            CMB.remove(x)

    return CMB, ci_number

# Algorithm 2. PCSuperSet


def PCSuperSet(data, target, alpha, indep_test='fisherz'):
    ci_number = 0
    d_sep = dict()
    _, kVar = np.shape(data)
    PCS = [i for i in range(kVar) if i != target]
    PCS_temp = PCS.copy()

    cond_indep_test = cit.CIT(data, indep_test)

    for x in PCS_temp:
        ci_number += 1
        pval = cond_indep_test(target, x, [])
        if pval > alpha:
            PCS.remove(x)
            d_sep.setdefault(x, [])

    PCS_temp = PCS.copy()
    for x in PCS_temp:
        PCS_rmX = [i for i in PCS if i != x]
        for y in PCS_rmX:
            ci_number += 1
            pval = cond_indep_test(target, x, [y])
            if pval > alpha:
                PCS.remove(x)
                d_sep.setdefault(x, [y])
                break

    return PCS, d_sep, ci_number


# Algorithm 3. SPSuperSet

def SPSuperSet(data, target, PCS, d_sep, alpha, indep_test='fisherz'):
    ci_number = 0
    _, kVar = np.shape(data)
    SPS = []

    cond_indep_test = cit.CIT(data, indep_test)

    for x in PCS:
        SPS_x = []
        vari_set = [i for i in range(kVar) if i != target and i not in PCS]
        for y in vari_set:
            conditon_set = list(d_sep[y])
            conditon_set.append(x)
            conditon_set = list(set(conditon_set))
            ci_number += 1
            pval = cond_indep_test(target, y, conditon_set)
            if pval <= alpha:
                SPS_x.append(y)

        SPS_x_temp = SPS_x.copy()
        for y in SPS_x_temp:
            SPS_x_rmy = [i for i in SPS_x if i != y]
            for z in SPS_x_rmy:
                ci_number += 1
                pval = cond_indep_test(target, y, [x, z])
                if pval > alpha:
                    SPS_x.remove(y)
                    break

        SPS = list(set(SPS).union(set(SPS_x)))

    return SPS, ci_number


# Algorithm 4. MBtoPC

def MBtoPC(data, target, alpha, attribute, indep_test='fisherz', n_jobs=1):
    max_k = 3
    ci_number = 0

    # Pass indep_test to IAMB
    MB, ci_num = IAMB(data, target, alpha, attribute, indep_test)
    ci_number += ci_num
    PC = MB.copy()
    cond_indep_test = cit.CIT(data, indep_test)
    for x in MB:
        break_flag = False
        condtion_sets_all = [i for i in MB if i != x]
        c_length = len(condtion_sets_all)
        if c_length > max_k:
            c_length = max_k
        for j in range(c_length + 1):
            condtion_sets = subsets(condtion_sets_all, j)
            if n_jobs > 1 and len(condtion_sets) > 0:
                pvals = Parallel(n_jobs=n_jobs)(
                    delayed(cond_indep_test)(target, x, list(Z)) for Z in condtion_sets
                )
                ci_number += len(pvals)
                if any(p > alpha for p in pvals):
                    PC.remove(x)
                    break_flag = True
                    break
            else:
                for Z in condtion_sets:
                    ci_number += 1
                    pval = cond_indep_test(target, x, list(Z))
                    if pval > alpha:
                        PC.remove(x)
                        break_flag = True
                        break
            if break_flag:
                break
    return PC, ci_number


# Algorithm 1. MBOR

def MBOR(data, target, alpha, indep_test='fisherz', n_jobs=1):
    _, kVar = np.shape(data)
    max_k = 3
    ci_number = 0

    cond_indep_test = cit.CIT(data, indep_test)
    
    # Pass indep_test to PCSuperSet and SPSuperSet
    PCS, d_sep, ci_num = PCSuperSet(data, target, alpha, indep_test)
    ci_number += ci_num
    SPS, ci_num = SPSuperSet(data, target, PCS, d_sep, alpha, indep_test)
    ci_number += ci_num
    MBS = list(set(PCS).union(set(SPS)))

    data_attribute = [i for i in range(kVar) if i == target or i in MBS]

    PC, ci_num = MBtoPC(data, target, alpha, data_attribute, indep_test, n_jobs)
    ci_number += ci_num
    PCS_rmPC = [i for i in PCS if i not in PC]
    for x in PCS_rmPC:
        x_pcset, ci_num = MBtoPC(data, x, alpha, data_attribute, indep_test, n_jobs)
        ci_number += ci_num
        if target in x_pcset:
            PC.append(x)

    SP = []
    for x in PC:
        data_attribute = [i for i in range(kVar) if i != target]
        x_pcset, ci_num = MBtoPC(data, x, alpha, data_attribute, indep_test, n_jobs)
        ci_number += ci_num
        vari_set = [i for i in x_pcset if i != target and i not in PC]
        for y in vari_set:
            break_flag = False
            condition_all_set = [i for i in MBS if i != target and i != y]
            clength = len(condition_all_set)
            if clength > max_k:
                clength = max_k
            for j in range(clength + 1):
                from joblib import Parallel, delayed
                if n_jobs > 1:
                    pvals = Parallel(n_jobs=n_jobs)(
                        delayed(cond_indep_test)(target, y, list(Z))
                        for Z in subsets(condition_all_set, j)
                    )
                    ci_number += len(pvals)
                    if any(p > alpha for p in pvals):
                        break_flag = True
                        # Re-test with x appended to each candidate conditioning set:
                        pvals2 = Parallel(n_jobs=n_jobs)(
                            delayed(cond_indep_test)(
                                target, y, list(set(list(Z)) | {x})
                            ) for Z in subsets(condition_all_set, j)
                        )
                        ci_number += len(pvals2)
                        if any(p <= alpha for p in pvals2):
                            SP.append(y)
                        break
                else:
                    for Z in subsets(condition_all_set, j):
                        ci_number += 1
                        pval = cond_indep_test(target, y, list(Z))
                        if pval > alpha:
                            break_flag = True
                            condition_varis = list(Z)
                            condition_varis.append(x)
                            condition_varis = list(set(condition_varis))
                            ci_number += 1
                            pval = cond_indep_test(target, y, condition_varis)
                            if pval <= alpha:
                                SP.append(y)
                    if break_flag:
                        break

    MB = list(set(PC).union(set(SP)))
    return MB, ci_number


# import pandas as pd
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 19
# alpha = 0.01
#
# MB = MBOR(data, target, alpha, True)
# print("MBs is: " + str(MB))


# 500
#
# F1 is: 0.85
# Precision is: 0.92
# Recall is: 0.82
# Distance is: 0.23
# ci_number is: 381.90
# time is: 61.39


# 5000
#
# F1 is: 0.97
# Precision is: 0.96
# Recall is: 0.99
# Distance is: 0.05
# ci_number is: 765.37
# time is: 371.77
