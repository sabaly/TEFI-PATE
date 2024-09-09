"""
@author: hierno-mamoudou.sabaly@telecom-sudparis.eu

This file experiments the impact of the parameter beta of the weighting vote approach.

It's usage is similar to the fairnessimpact_adult.py file.

The resulting image will be in ../img/
"""

import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from student import *
from aggregator import *
import matplotlib.pyplot as plt
from analysis import *
from teacher_ensemble import *
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from random import shuffle


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


conf = ["Normal", "WV2"]

def get_agg(cf):
    if cf == "Normal":
        aggregator = plurality
    elif cf == "WV2":
        aggregator = wv2_aggregator
    
    return aggregator

# student model
st_model = RandomForestClassifier()


nb_teachers = 30

def train_students(nb_teachers, nb_fair_tchrs, beta, metric="SPD"):
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
        y_train, _ = aggregator(x_train, beta=beta)
        yhat_test, _ = aggregator(x_test, beta=beta)
        st_model.fit(x_train, y_train)
        st_stats = fairness(st_model, x_test, yhat_test, s_test)
        loc_st_fairness[cf].append(st_stats[metric])
    return loc_st_fairness

def wrapper(args):
    return train_students(*args)

fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharey=True)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
axe_ind = 0
path = "../img/"
for beta in range(1,7):
    st_fairness = {}
    loc_st_fairnesses = []
    with Pool(5) as p:
        loc_st_fairnesses = p.map(wrapper, [(nb_teachers, i, beta) for i in range(1, nb_teachers)])
        p.close()
    
    for cf in conf:
        st_fairness[cf] = sum([l_st_f[cf] for l_st_f in loc_st_fairnesses], [])

    np.save(path + f"beta_{beta}.npy", st_fairness)
    axes[axe_ind].plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], label="Normal")
    axes[axe_ind].plot(list(range(1, nb_teachers)), st_fairness["WV2"], color=colors[1], label=f"beta: {beta}")
    axes[axe_ind].legend()
    axe_ind += 1
    print(f"beta : {beta}....DONE")
#fig.legend(loc="outside upper left",ncol=4)
ax1.set_ylabel("S.P.D")
ax5.set_xlabel("Number of fair teachers")

plt.savefig(path + "wv2_beta_impact.png")


