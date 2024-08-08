"""
    same as in attack2.py. But this time instead creating pole D_1 and D_2. We first run the black-box attact to get 
    the infered attribut *infered_S*. We take as D_1 = infered_S and D_2 = ~D_1 (flipping all the bits of D_1).

    And run the same attack.
"""

import numpy as np
from teacher_ensemble import *
from aggregator import *
import warnings
warnings.filterwarnings('ignore')


# defining teachers
nb_teachers = 15
nb_fair_tchrs = 5

st_teachers = states[:15]
#tchrs_ensemble = Ensemble(nb_teachers, nb_fair_tchrs)
teachers = []
root = "../checkpoint_sex/"
id = 0
for st in st_teachers:
    if id < nb_fair_tchrs:
        path = root + st + "/" + st  + "_fair.pkl"
    else:
        path = root + st + "/" + st  + "_unfair.pkl"
    id += 1
    with open(path, "rb") as f:
        tchr = pickle.load(f)
    teachers.append(tchr)
# define the target teacher
target = 7
# target's dataset
x_train, _, y_train = teachers[target].splited_data[:3] # training datas
if  not isinstance(x_train, pd.DataFrame):
    x_train = pd.DataFrame(x_train, columns=ACSEmployment.features)

sensitive_column = "SEX"

# --------------- Black Box

from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox, AttributeInferenceBaseline
from art.estimators.regression.scikitlearn import ScikitlearnRegressor

attack_feature = x_train.columns.get_loc(sensitive_column)


x_train_np = np.asarray(x_train)
attack_train_ratio = 0.5
attack_train_size = int(len(x_train_np) * attack_train_ratio)
attack_x_train = x_train_np[:attack_train_size]
attack_y_train = y_train[:attack_train_size]
attack_x_test = x_train_np[attack_train_size:]
attack_y_test = y_train[attack_train_size:]

art_target_model = ScikitlearnRegressor(teachers[target].model)
attack_x_test_predictions = np.array([np.round(arr) for arr in art_target_model.predict(attack_x_test)])
attack_x_test_feature = attack_x_test[:, attack_feature].copy()
x_test_for_attack = np.delete(attack_x_test, attack_feature, 1)

bb_attack = AttributeInferenceBlackBox(art_target_model, attack_feature=attack_feature)
bb_attack.fit(attack_x_train)

values = [1,2]
infered_train_bb = bb_attack.infer(x_test_for_attack, pred=attack_x_test_predictions, values=values)

bb_train_acc = np.sum(infered_train_bb == attack_x_test_feature) / len(infered_train_bb)

# --------- new attack based on infered_train_bb
attack_x_test_predictions = np.array([np.round(arr) for arr in art_target_model.predict(np.asarray(x_train))])
x_test_for_attack = np.delete(np.asarray(x_train.copy()), attack_feature, 1)
infered_train_bb = bb_attack.infer(x_test_for_attack, pred=attack_x_test_predictions, values=values)
infered_train_bb = [0 if x == 2 else 1 for x in infered_train_bb]
opp_infered_train_bb = [1^x for x in infered_train_bb]

D_2 = x_train.copy()
D_2[sensitive_column] = infered_train_bb
D_1 = x_train.copy()
D_1[sensitive_column] = opp_infered_train_bb


new_column = sensitive_column+"REC"
original_data = x_train.copy()
original_data[sensitive_column] = original_data[sensitive_column].map({1:1, 2: 0})
original_data.insert(0, new_column, infered_train_bb)
bb_precision_st = original_data[[sensitive_column, new_column]].value_counts(normalize=True)
#print(precision)
precision_bb_attack = bb_precision_st[0][0] + bb_precision_st[1][1]


# make teachers predictions
D_2[sensitive_column] = D_2[sensitive_column].map({1: 1, 0:2})
D_1[sensitive_column] = D_1[sensitive_column].map({1: 1, 0:2})
attack_feature = D_2.columns.get_loc(sensitive_column)
D_2, D_1 = np.asarray(D_2), np.asarray(D_1)
tchrs_reconst = []
weights = []
metric = teachers[target].metrics["EOD"]
for tchr in teachers:
    pred1 = tchr.model.predict(D_1)
    pred2 = tchr.model.predict(D_2)
    n = len(pred1)
    reconst_attr = []
    sex=0
    for i in range(n):
        y_1 = pred1[i]
        y_2 = pred2[i]
        if D_2[:,attack_feature][i] == 2:
            dist = [abs(np.round(y_2) - y_2), (metric/y_2)*abs(np.round(y_1) - y_1)]
        else:
            dist = [abs(np.round(y_1) - y_1), (metric/y_1)*abs(np.round(y_2) - y_2)]
        sex = np.argmin(dist)
        reconst_attr.append(int(sex))
    tchrs_reconst.append(reconst_attr)
    weights.append(int((1-tchr.metrics["SPD"])*100))
tchrs_reconst = np.asarray(tchrs_reconst)

max_weights = max(weights)
weights[target] = max_weights

def vote_attribute(tchrs_reconst, ws=[]):
    reconstructed_attr = []
    for x in range(tchrs_reconst.shape[1]):
        pred = list(tchrs_reconst[:, x])
        if ws != []:
            for i in range(len(pred)):
                pred = pred + [pred[i]]*ws[i]
        counter = np.bincount(pred)
        reconstructed_attr.append(np.argmax(counter))

    return reconstructed_attr

reconst_attr = vote_attribute(tchrs_reconst)
reconst_attr_w = vote_attribute(tchrs_reconst, weights)

original_data = x_train.copy()
original_data[sensitive_column] = original_data[sensitive_column].map({1:1, 2: 0})
new_column = sensitive_column+"REC"
original_data.insert(0, new_column, reconst_attr)
att_precision_st = original_data[[sensitive_column, new_column]].value_counts(normalize=True)
att_precision = att_precision_st[0][0] + att_precision_st[1][1]

original_data = x_train.copy()
original_data[sensitive_column] = original_data[sensitive_column].map({1:1, 2: 0})
original_data.insert(0, new_column, reconst_attr_w)
w_att_precision_st = original_data[[sensitive_column, new_column]].value_counts(normalize=True)
#print(precision)
w_att_precision = w_att_precision_st[0][0] + w_att_precision_st[1][1]

print("Precision bb_attack on x_train ==> ", format(precision_bb_attack*100, ".2f"), "%")
print(bb_precision_st)

print(">>> Simple vote")
print("Attaque precision is ==> ", format(att_precision*100, ".2f"), "%")
print(att_precision_st)

print(">>> Weighted vote")
print("Attaque precision is ==> ", format(w_att_precision*100, ".2f"), "%")
print(w_att_precision_st)

