"""
@author : hierno-mamoudou.sabaly@telecom-sudparis.eu

Same as for fairnessimpact_acsemployment.py but for adult dataset.
"""

from analysis import mean
import numpy as np
import pickle
from aggregator import *
import matplotlib.pyplot as plt
from random import shuffle
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
color_index = 0

def fairness(model, x_test, y_test, group_test, forest_model=True, true_y_test=[]):
    yhat = np.round(model.predict(x_test))
    if forest_model:
        acc = model.score(x_test, y_test)
    else:
        acc = model.evaluate(x_test, y_test)[1]
    
    p_grp_tpr = mean(yhat[(y_test == 1) & (group_test == 1)])
    up_grp_tpr = mean(yhat[(y_test == 1) & (group_test == 0)])
    
    # equality of difference (opportinuty)
    eod = float(format(abs(p_grp_tpr - up_grp_tpr), ".4f"))

    # statistical parity difference
    p_grp = mean(yhat[(group_test == 1)])
    up_grp = mean(yhat[(group_test == 0)])
    spd = float(format(abs(p_grp - up_grp), ".4f"))
    return {"EOD": eod, "SPD": spd, "ACC": acc}


###########################################
# Loading teachers' models
###########################################

class AdultTeacher():
    """
        Define teachers' structure
    """
    def __init__(self, model, metrics):
        self.metrics = metrics
        self.model = model

eod_fairs_path = "../checkpoint_adult/EOD/fairs/models.pkl"
eod_unfairs_path = "../checkpoint_adult/EOD/unfairs/models.pkl"
spd_fairs_path = "../checkpoint_adult/SPD/fairs/models.pkl"
spd_unfairs_path = "../checkpoint_adult/SPD/unfairs/models.pkl"

with open(eod_fairs_path, "rb") as f:
    eod_fairs = pickle.load(f)
with open(eod_unfairs_path, "rb") as f:
    eod_unfairs = pickle.load(f)
with open(spd_fairs_path, "rb") as f:
    spd_fairs = pickle.load(f)
with open(spd_unfairs_path, "rb") as f:
    spd_unfairs = pickle.load(f)


####################################################
# Loading student dataset and defining student model
####################################################
data_path = "../checkpoint_adult/st_dataset.pkl"
with open(data_path, "rb") as f:
    dataset = pickle.load(f)

Y = dataset['income']
X = dataset.drop('income',axis=1)
S = dataset['gender']

x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
        X, Y, S, test_size=0.2, random_state=0
    )
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.asarray(x_test), np.asarray(y_test)
s_train, s_test = np.asarray(s_train), np.asarray(s_test)

# student model
st_model = RandomForestClassifier()

conf = ["Normal", "Only fair", "Only unfair", "WV0", "WV1", "WV2"]
conf = ["Normal", "WV0", "WV1", "WV2"]

def get_agg(cf):
    if cf == "Normal":
        aggregator = plurality
    elif cf == "Only fair":
        aggregator = only_fair
    elif cf == "Only unfair":
        aggregator = only_unfair
    elif cf == "WV0":
        aggregator = weighed_vote
    elif cf == "WV1":
        aggregator = wv1_aggregator
    elif cf == "WV2":
        aggregator = wv2_aggregator
    
    return aggregator

nb_teachers = 30

def train_students(nb_teachers, nb_fair_tchrs, metric="EOD"):
    loc_st_fairness = {}
    for cf in conf:
        loc_st_fairness[cf] = []

    if metric == "SPD":
        unfairs = spd_unfairs.copy()
        fairs = spd_fairs.copy()
    else:
        unfairs = eod_unfairs.copy()
        fairs = eod_fairs.copy()
    shuffle(unfairs)
    shuffle(fairs)
    teachers = fairs[:nb_fair_tchrs] + unfairs[:nb_teachers - nb_fair_tchrs]
    update_teachers(teachers)

    metrics = []
    for tchr in teachers:
        metrics.append(tchr.metrics[metric])
    set_metrics(metrics)

    for cf in conf:
        aggregator = get_agg(cf)
        y_train, _ = aggregator(x_train)
        yhat_test, _ = aggregator(x_test)
        st_model.fit(x_train, y_train)
        st_stats = fairness(st_model, x_test, yhat_test, s_test)
        loc_st_fairness[cf].append(st_stats[metric])
    return loc_st_fairness

nb_teacher = 30

#fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharey=True)
fig, ((ax3,ax4), (ax5,ax6)) = plt.subplots(2,2, sharey=True)
ax6.set_visible(False)

def wrapper(args):
    return train_students(*args)

st_fairness = {}
loc_st_fairnesses = []
print("Loading....", end="")
with Pool(5) as p:
    loc_st_fairnesses = p.map(wrapper, [(nb_teachers, i) for i in range(1, nb_teachers)])
    p.close()

for cf in conf:
    st_fairness[cf] = sum([l_st_f[cf] for l_st_f in loc_st_fairnesses], [])
color_index = 1

for cf in conf:
    if cf == "Normal":
        continue
    elif cf == "Only fair":
        ax3.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
        ax3.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')
    elif cf == "Only unfair":
        ax4.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
        ax4.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')
    elif cf == "WV0":
        ax3.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
        ax3.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')
    elif cf == "WV1":
        ax4.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
        ax4.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')
    elif cf == "WV2":
        ax5.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
        ax5.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], label="Normal", marker='o')

    color_index += 1
fig.legend(loc="outside upper left",ncol=4, fontsize=12)
#ax1.set_ylabel("Student fairness", fontsize=12)
ax3.set_ylabel("Student fairness", fontsize=12)
ax5.set_xlabel("Number of fair teachers", fontsize=12)

#ax1.grid(True, linestyle='--', alpha=0.9)
#ax2.grid(True, linestyle='--', alpha=0.9)
ax3.grid(True, linestyle='--', alpha=0.9)
ax4.grid(True, linestyle='--', alpha=0.9)
ax5.grid(True, linestyle='--', alpha=0.9)
ax6.grid(True, linestyle='--', alpha=0.9)

path = "../img/archive_"+ str(nb_teachers) + "/"
plt.savefig(path + "st_fairness_variations_" + str(nb_teachers) + "_teachers.png")

np.save(path + "st_fairness.npy", st_fairness)
print("Done")
