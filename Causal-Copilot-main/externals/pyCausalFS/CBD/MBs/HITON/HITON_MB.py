# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/17 17:00
desc:
"""
import os
import sys

causal_learn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 'causal-learn')
sys.path.append(causal_learn_dir)


import causallearn.utils.cit as cit
from CBD.MBs.HITON.HITON_PC import HITON_PC
from joblib import Parallel, delayed


def HITON_MB(data, target, alpha, indep_test='fisherz', n_jobs=1):

    cond_indep_test = cit.CIT(data, indep_test)
    PC, sepset, ci_number = HITON_PC(data, target, alpha, indep_test)
    currentMB = PC.copy()
    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(HITON_PC)(data, x, alpha, indep_test) for x in PC
        )
        for (PCofPC, _, ci_num2), x in zip(results, PC):
            ci_number += ci_num2
            for y in PCofPC:
                if y != target and y not in PC:
                    conditions_Set = list(sepset[y])
                    conditions_Set.append(x)
                    conditions_Set = list(set(conditions_Set))
                    ci_number += 1
                    pval = cond_indep_test(target, y, conditions_Set)
                    if pval <= alpha:
                        currentMB.append(y)
                        break
    else:
        for x in PC:
            PCofPC, _, ci_num2 = HITON_PC(data, x, alpha, indep_test)
            ci_number += ci_num2
            for y in PCofPC:
                if y != target and y not in PC:
                    conditions_Set = list(sepset[y])
                    conditions_Set.append(x)
                    conditions_Set = list(set(conditions_Set))
                    ci_number += 1
                    pval = cond_indep_test(target, y, conditions_Set)
                    if pval <= alpha:
                        currentMB.append(y)
                        break

    return list(set(currentMB)), ci_number


# alpha = 0.01
# start_time = time.process_time()
# for target in range(kvar):
#     print("target:", target)
#     MBs, ci_number = HITON_MB(data, target, alpha, True)
#     print("ci_number : ", ci_number)
#     # print(dic["cache"][0], "-", dic["cache"][1],
#     #   "-", (dic["cache"][0] + dic["cache"][1]))
#     # print(dic["cache"][0] / (dic["cache"][0] + dic["cache"][1]))
#
# end_time = time.process_time()
# print("run time = ", end_time - start_time)

# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 4
# alpha = 0.05
#
# MBs=HITON_MB(data,target,alpha)
# print("MBs is: "+str(MBs))
