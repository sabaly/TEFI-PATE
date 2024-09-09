"""
@author: thierno-mamoudou.sabaly@telecom-sudparis.eu

This fail implement the function that evaluate models and compute their fairness metrics.
"""
import numpy as np
import math


def mean(myarray):
    mn = np.mean(myarray)
    return 0 if math.isnan(mn) else mn

def fairness(model, x_test, y_test, group_test, true_y_test=[]):
    ev = model.evaluate(x_test, y_test)
    acc = float(format(ev[1], "0.4f"))
    rec = float(format(ev[2], ".4f"))

    yhat = np.round(model.predict(x_test))
    p_grp_tpr = mean(yhat[(y_test == 1) & (group_test == 1)])
    up_grp_tpr = mean(yhat[(y_test == 1) & (group_test == 2)])
    
    # equality of difference (opportinuty)
    eod = float(format(abs(p_grp_tpr - up_grp_tpr), ".4f"))

    # statistical parity difference
    p_grp = mean(yhat[(group_test == 1)])
    up_grp = mean(yhat[(group_test == 2)])
    spd = float(format(abs(p_grp - up_grp), ".4f"))

    if true_y_test != []:
        ev = model.evaluate(x_test, true_y_test)
        acc_tl = float(format(ev[1], "0.4f"))
        rec_tl = float(format(ev[2], ".4f"))
    else:
        rec_tl = acc_tl = '-'

    return {"EOD": eod, "SPD": spd, "ACC": acc, "REC": rec, "ACC_TL": acc_tl, "REC_TL": rec_tl}

def stats(nb_teachers, teachers, subsets, S):
    accuracies = []
    eod = []
    spd = []
    rec = []
    for i in range(nb_teachers):
        params = [subsets[i][1], subsets[i][3], subsets[i][5]]
        stat = fairness(teachers[i], *params)
        accuracies.append(stat["ACC"])
        eod.append(stat["EOD"])
        spd.append(stat["SPD"])
        rec.append(stat["REC"])
    return accuracies, eod, spd, rec
    
    

