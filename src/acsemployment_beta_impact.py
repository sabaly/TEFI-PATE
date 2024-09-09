"""
@author: hierno-mamoudou.sabaly@telecom-sudparis.eu

This file experiments the impact of the parameter beta of the weighting vote approach.

It's usage is similar to the fairnessimpact_acsemployment.py file.

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


colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
color_index = 0


parallel = 1
if len(sys.argv) > 1:
    parallel = int(sys.argv[1])
dataset = "acsemployment_bis"

(x_train, x_test, y_train, y_test, s_train, s_test) = load_student_data("AK", attr="sex")

conf = ["Normal", "WV2"]

def get_agg(cf):
    if cf == "Normal":
        aggregator = plurality
    elif cf == "WV2":
        aggregator = wv2_aggregator
    
    return aggregator

def train_students(nb_teachers, nb_fair_tchrs, beta=100):
    loc_st_fairness = {}
    for cf in conf:
        loc_st_fairness[cf] = []

    tchrs_ensemble = Ensemble(nb_teachers, nb_fair_tchrs)
    update_teachers(tchrs_ensemble.tchrs)
    
    metric = []
    metric_key = "SPD"
    for tchrs in tchrs_ensemble.tchrs:
        metric.append(tchrs.metrics[metric_key])
    set_metrics(metric)

    for cf in conf:
        aggregator = get_agg(cf)
        y_train, _ = aggregator(x_train, beta=beta)
        yhat_test, _ = aggregator(x_test, beta=beta)
        st_model = train_student(x_train, y_train, verbose=False)
        st_stats = fairness(st_model, x_test, yhat_test, s_test)
        loc_st_fairness[cf].append(st_stats[metric_key])
    return loc_st_fairness

def wrapper(args):
    return train_students(*args)
nb_teachers = 30

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


