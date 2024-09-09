"""
@author : hierno-mamoudou.sabaly@telecom-sudparis.eu

This file executed correctly yield a figure with the fairness impact of teachers in the following settings:
** Only fair teachers with standar aggregation
** Only unfair teachers with standar aggregation
** The combinasion of both fair and unfair teachers with standar aggregation
** The combinasion of both fair and unfair teachers with WV0 aggregation (see aggregation.py)
** The combinasion of both fair and unfair teachers with fairfed aggregation (see aggregation.py)
** The combinasion of both fair and unfair teachers with WV1 aggregation (see aggregation.py)
** The combinasion of both fair and unfair teachers with WV2 aggregation (see aggregation.py)

To adjust student data (make them balanced or unbalanced for example) you can alter the alpha value.

You can test for many number of teachers by adapting the for loop. And the image yield is stored in ../img/archive_{nb_teachers}
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

(x_train, x_test, y_train, y_test, s_train, s_test) = load_student_data("AK", attr="sex", alpha=[200, 70])

""" st_model = train_student(x_train, y_train, verbose=False)
st_stats = fairness(st_model, x_test, y_test, s_test)
#np.save("../img/st_stats.npy", st_stats)
print(st_stats)
exit(1) """
#np.save("tmp/" + "_st_init_stats.npy", stats)

conf = ["Normal", "Only fair", "Only unfair", "WV0", "Fairfed", "WV1", "WV2"]       # comment if you don't wan to test or Only fair and Only Unfair
#conf = ["Normal", "WV0", "Fairfed", "WV1", "WV2"]                                  # uncomment if you don't wan to test or Only fair and Only Unfair
def get_agg(cf):
    if cf == "Normal":
        aggregator = plurality
    elif cf == "Only fair":
        aggregator = only_fair
    elif cf == "Only unfair":
        aggregator = only_unfair
    elif cf == "WV0":
        aggregator = weighed_vote
    elif cf == "Fairfed":
        aggregator = fair_fed_agg
    elif cf == "WV1":
        aggregator = wv1_aggregator
    elif cf == "WV2":
        aggregator = wv2_aggregator
    
    return aggregator

def train_students(nb_teachers, nb_fair_tchrs):
    loc_st_fairness = {}
    for cf in conf:
        loc_st_fairness[cf] = []

    tchrs_ensemble = Ensemble(nb_teachers, nb_fair_tchrs)
    update_teachers(tchrs_ensemble.tchrs)
    
    metric = []
    metric_key = "EOD"
    for tchrs in tchrs_ensemble.tchrs:
        metric.append(tchrs.metrics[metric_key])
    set_metrics(metric)

    for cf in conf:
        print(f'>>> case : {cf}')
        aggregator = get_agg(cf)
        y_train, _ = aggregator(x_train)
        yhat_test, _ = aggregator(x_test)
        st_model = train_student(x_train, y_train, verbose=False)
        st_stats = fairness(st_model, x_test, yhat_test, s_test)
        loc_st_fairness[cf].append(st_stats[metric_key])
    return loc_st_fairness

def wrapper(args):
    return train_students(*args)

for nb_teachers in [30]:
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharey=True)    # comment if you don't wan to test or Only fair and Only Unfair
    #fig, ((ax3,ax4), (ax5,ax6)) = plt.subplots(2,2, sharey=True)           # uncomment if you don't wan to test or Only fair and Only Unfair
    ax6.set_visible(False)

    st_fairness = {}
    print(">>> ", nb_teachers, " teachers ")
    loc_st_fairnesses = []
    if not parallel:
        for i in range(1, nb_teachers):
            loc_st_fairnesses.append(train_students(nb_teachers, i))
    else:
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
            ax1.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')     # comment if you don't wan to test or Only fair and Only Unfair
            ax1.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')                   # comment if you don't wan to test or Only fair and Only Unfair
        elif cf == "Only unfair":
            ax2.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')     # comment if you don't wan to test or Only fair and Only Unfair
            ax2.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')                   # comment if you don't wan to test or Only fair and Only Unfair
        elif cf == "WV0":
            ax3.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
            ax3.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')
        elif cf == "Fairfed":
            ax4.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
            ax4.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')
        elif cf == "WV1":
            ax5.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
            ax5.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], marker='o')
        elif cf == "WV2":
            ax6.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf, marker='o')
            ax6.plot(list(range(1, nb_teachers)), st_fairness["Normal"], color=colors[0], label="Normal", marker='o')

        color_index += 1
    fig.legend(loc="outside upper left",ncol=4, fontsize=12)
    ax3.set_ylabel("Student fairness", fontsize=12)
    ax5.set_xlabel("Number of fair teachers", fontsize=12)


    ax1.grid(True, linestyle='--', alpha=0.9)       # comment if you don't wan to test or Only fair and Only Unfair
    ax2.grid(True, linestyle='--', alpha=0.9)       # comment if you don't wan to test or Only fair and Only Unfair
    ax3.grid(True, linestyle='--', alpha=0.9)
    ax4.grid(True, linestyle='--', alpha=0.9)
    ax5.grid(True, linestyle='--', alpha=0.9)
    ax6.grid(True, linestyle='--', alpha=0.9)

    plt.tight_layout()

    path = "../img/archive_"+ str(nb_teachers) + "/"
    plt.savefig(path + "st_fairness_variations_" + str(nb_teachers) + "_teachers.png")
    
    np.save(path + "st_fairness.npy", st_fairness)
    # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


