#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/8/6 20:18
 @File    : BAMB.py
 """


import numpy as np
from CBD.MBs.common.subsets import subsets
import os
import sys

causal_learn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'causal-learn')
sys.path.append(causal_learn_dir)

import causallearn.utils.cit as cit
from joblib import Parallel, delayed


def BAMB(data, target, alpha, indep_test='fisherz', n_jobs=1):
    ci_number = 0
    number, kVar = np.shape(data)
    max_k = 3
    CPC = []
    TMP = [i for i in range(kVar) if i != target]
    sepset = [[] for i in range(kVar)]
    CSPT = [[] for i in range(kVar)]
    variDepSet = []
    SP = [[] for i in range(kVar)]
    PC = []

    cond_indep_test = cit.CIT(data, indep_test)

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(cond_indep_test)(target, x, []) for x in TMP
        )
        for x, pval_f in zip(TMP, results):
            ci_number += 1
            dep_f = -pval_f
            if pval_f > alpha:
                sepset[x] = []
            else:
                variDepSet.append([x, dep_f])
    else:
        for x in TMP:
            ci_number += 1
            pval_f = cond_indep_test(target, x, [])
            dep_f = -pval_f
            if pval_f > alpha:
                sepset[x] = []
            else:
                variDepSet.append([x, dep_f])

    variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
    """step one: Find the candidate set of PC and candidate set of spouse"""

    # print("variDepSet" + str(variDepSet))
    for variIndex in variDepSet:
        A = variIndex[0]
        # print("A is: " + str(A))
        Slength = len(CPC)
        if Slength > max_k:
            Slength = 3
        breakFlag = False
        for j in range(Slength + 1):
            ZSubsets = subsets(CPC, j)
            for Z in ZSubsets:
                ci_number += 1
                convari = list(Z)
                pval_TAZ = cond_indep_test(target, A, convari)
                dep_TAZ = -pval_TAZ
                if pval_TAZ > alpha:
                    sepset[A] = convari
                    breakFlag = True
                    # print("ZZZ")
                    break
            if breakFlag:
                break

        if not breakFlag:
            CPC_ReA = CPC.copy()
            B_index = len(CPC_ReA)
            CPC.append(A)
            breakF = False
            while B_index > 0:
                B_index -= 1
                B = CPC_ReA[B_index]
                flag1 = False

                conditionSet = [i for i in CPC_ReA if i != B]
                Clength = len(conditionSet)
                if Clength > max_k:
                    Clength = max_k
                for j in range(Clength + 1):
                    CSubsets = subsets(conditionSet, j)
                    for Z in CSubsets:
                        ci_number += 1
                        convari = list(Z)
                        pval_TBZ = cond_indep_test(target, B, convari)
                        if pval_TBZ >= alpha:
                            CPC.remove(B)
                            CSPT[B] = []
                            sepset[B] = convari
                            flag1 = True
                            if B == A:
                                breakF = True
                    if flag1:
                        break
                if breakF:
                    break

            CSPT[A] = []
            pval_CSPT = []

            # add candidate of spouse

            # print("sepset: " + str(sepset))
            for C in range(kVar):
                if C == target or C in CPC:
                    continue
                conditionSet = list(sepset[C])
                conditionSet.append(A)
                conditionSet = list(set(conditionSet))

                ci_number += 1
                pval_CAT = cond_indep_test(target, C, conditionSet)
                if pval_CAT <= alpha:
                    CSPT[A].append(C)
                    pval_CSPT.append([C, pval_CAT])

            """step 2-1"""

            pval_CSPT = sorted(pval_CSPT, key=lambda x: x[1], reverse=False)
            SP[A] = []
            # print("CSPT-: " +str(CSPT))
            # print("pval_CSPT is: " + str(pval_CSPT))

            for pCSPT_index in pval_CSPT:
                E = pCSPT_index[0]
                # print("E is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " +str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]
                    breakFlag = False
                    # print("x is:" + str(x))

                    ZAllconditionSet = [i for i in SP[A] if i != x]
                    # print("ZAllconditionSet is:" + str(ZAllconditionSet))
                    for Z in ZAllconditionSet:
                        conditionvari = [Z]
                        if A not in conditionvari:
                            conditionvari.append(A)
                        ci_number += 1
                        pval_TXZ = cond_indep_test(target, x, conditionvari)
                        # print("x is: " + str(x) + "conditionvari: " + str(conditionvari) + " ,pval_TXZ is: " + str(pval_TXZ))
                        if pval_TXZ > alpha:
                            # print("spa is: " + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(conditionvari))
                            SP[A].remove(x)
                            breakFlag = True

                            if x == E:
                                breakflag_spa = True
                            break
                    if breakFlag:
                        break
                if breakflag_spa:
                    break

            """step 2-2"""
            # remove x from pval_CSPT
            pval_CSPT_new = []
            plength = len(pval_CSPT)
            for i in range(plength):
                if pval_CSPT[i][0] in SP[A]:
                    pval_CSPT_new.append(pval_CSPT[i])

            CSPT[A] = SP[A]
            SP[A] = []
            # print("CSPT-: " + str(CSPT))
            # print("2222222pval_CSPT_new is: " + str(pval_CSPT_new))

            for pCSPT_index in pval_CSPT_new:
                E = pCSPT_index[0]
                # print("E2 is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " + str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]

                    breakFlag = False
                    # print("x is:" + str(x))
                    ZAllSubsets = list(set(CPC).union(set(SP[A])))
                    # print("CPC is: " + str(CPC) + " , SP[A] is: " + str(SP[A]) + " ,A is" + str(A) + " ,x is:" + str(x) + " ,ZA is: " + str(ZAllSubsets))
                    ZAllSubsets.remove(x)
                    ZAllSubsets.remove(A)
                    # print("-ZALLSubsets has: " + str(ZAllSubsets))
                    Zalength = len(ZAllSubsets)
                    if Zalength > max_k:
                        Zalength = max_k
                    for j in range(Zalength + 1):
                        ZaSubsets = subsets(ZAllSubsets, j)
                        for Z in ZaSubsets:
                            Z = list(Z)
                            ci_number += 1
                            pval_TXZ = cond_indep_test(A, x, Z)
                            # print("Z is: " + str(Z) + " ,A is: " + str(A) + " ,x is: " + str(x) + " ,pval_txz is: " + str(pval_TXZ))
                            if pval_TXZ > alpha:
                                # print("spa is:" + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(Z))
                                SP[A].remove(x)
                                breakFlag = True
                                if x == E:
                                    breakflag_spa = True
                                break
                        if breakFlag:
                            break
                    if breakflag_spa:
                        break

            """ step 2-3"""
            pval_CSPT_fin = []
            plength = len(pval_CSPT)
            for i in range(plength):
                if pval_CSPT[i][0] in SP[A]:
                    pval_CSPT_fin.append(pval_CSPT[i])

            CSPT[A] = SP[A]
            SP[A] = []
            # print("CSPT-: " +str(CSPT))
            # print("2222222pval_CSPT_fin is: " + str(pval_CSPT_fin))

            for pCSPT_index in pval_CSPT_fin:
                E = pCSPT_index[0]
                # print("E3 is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " + str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]
                    breakFlag = False

                    # print("x is:" + str(x))
                    ZAllSubsets = list(set(CPC).union(set(SP[A])))
                    ZAllSubsets.remove(x)
                    ZAllSubsets.remove(A)
                    Zalength = len(ZAllSubsets)
                    # print("=-ZALLSubsets has: " + str(ZAllSubsets))
                    if Zalength > max_k:
                        Zalength = max_k
                    for j in range(Zalength + 1):
                        ZaSubsets = subsets(ZAllSubsets, j)
                        # print("ZzSubsets is: " + str(ZaSubsets))
                        for Z in ZaSubsets:
                            Z = list(Z)
                            # Z.append(A)
                            # print("Z in ZaSubsets is: " + str(Z))
                            ci_number += 1
                            pval_TXZ = cond_indep_test(A, x, Z)
                            # print("-Z is: " + str(Z) + " ,x is: " + str(x) + " ,pval_txz is: " + str(
                            #     pval_TXZ))
                            if pval_TXZ >= alpha:
                                # print("spa is:" + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(Z))
                                SP[A].remove(x)
                                if x == E:
                                    breakflag_spa = True
                                breakFlag = True
                                break
                        if breakFlag:
                            break
                    if breakflag_spa:
                        break
            # print("SP[A]------: " + str(SP[A]))
            CSPT[A] = SP[A]
            # print("CSPT is: " + str(CSPT))

            """step3: remove false positives from the candidate set of PC"""

            CPC_temp = CPC.copy()
            x_index = len(CPC_temp)
            A_breakFlag = False
            # print("-CPC-: " + str(CPC))
            while x_index >= 0:
                x_index -= 1
                x = CPC_temp[x_index]
                flag2 = False
                ZZALLsubsets = [i for i in CPC if i != x]
                # print("xx is: " + str(x) + ", ZZALLsubsets is: " + str(ZZALLsubsets ))
                Zlength = len(ZZALLsubsets)
                if Zlength > max_k:
                    Zlength = max_k
                for j in range(Zlength + 1):
                    Zzsubsets = subsets(ZZALLsubsets, j)
                    for Z in Zzsubsets:
                        conditionSet = [
                            i for y in Z for i in CSPT[y] if i not in CPC]
                        conditionSet = list(set(conditionSet).union(set(Z)))
                        # print("conditionSet: " + str(conditionSet))
                        ci_number += 1
                        pval = cond_indep_test(target, x, conditionSet)
                        if pval >= alpha:
                            # print("remove x is: " + str(x) + " , pval is: " + str(pval) + " ,conditionset is: " + str(conditionSet))
                            CPC.remove(x)
                            CSPT[x] = []
                            flag2 = True
                            if x == A:
                                A_breakFlag = True
                            break
                    if flag2:
                        break
                if A_breakFlag:
                    break

    # print("SP is:" + str(SP))
    spouseT = [j for i in CPC for j in CSPT[i]]
    MB = list(set(CPC).union(set(spouseT)))
    return MB, ci_number


# import  pandas as pd
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s5000_v1.csv")
# print("the file read")
#
# target = 19
# alpha = 0.01
#
# import time
# start_time = time.process_time()
# MB, _ = BAMB(data, target, alpha, is_discrete=False)
# end_time = time.process_time()
#
# print(end_time - start_time)
# print("MBs is: "+str(MB))


# 5000
#
# F1 is: 0.97
# Precision is: 0.99
# Recall is: 0.96
# Distance is: 0.05
# ci_number is: 512.58
# time is: 62.22
