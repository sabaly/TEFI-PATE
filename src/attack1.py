from teacher_ensemble import *
from aggregator import *
from student import *
import warnings
warnings.filterwarnings('ignore')


# defining teachers
nb_teachers = 15
nb_fair_tchrs = 5
sensitive_column = "SEX"

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

update_teachers(teachers)
student_data = load_student_data("AK", attr="sex")
dataset = student_data[0]

aggregator = wv2_aggregator

if aggregator != plurality:
    metric = []
    for tchrs in teachers:
        metric.append(tchrs.metrics["SPD"])
    set_metrics(metric)

labels, _ = aggregator(dataset)

features = pd.DataFrame(dataset, columns=ACSEmployment.features)
features["ESR"] = labels
features[sensitive_column] = features[sensitive_column].map({1: 1, 2: 0})
labels = features.pop(sensitive_column)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
x_train, y_train = np.asarray(x_train), np.asarray(y_train)
x_test, y_test = np.asarray(x_test), np.asarray(y_test)

attack_model = define_model(x_train.shape[1:])
print("\nTraining attack model...", end="")
attack_model.fit(x_train, y_train, epochs=100, verbose=False)
print("Done")
ev = attack_model.evaluate(x_test, y_test)
print (f"**** Results : Attack model \n\t-loss : {ev[0]}\n\t-accuracy : {ev[1]}\n\t-recall : {ev[2]}")

# define the target teacher
target = 0
# target's dataset
x_train, _, y_train = teachers[target].splited_data[:3] # training datas
features = x_train
if  not isinstance(features, pd.DataFrame):
    features = pd.DataFrame(features, columns=ACSEmployment.features)

features["ESR"] = y_train
features["ESR"] = features["ESR"].map({True: 1, False: 0})
features[sensitive_column] = features[sensitive_column].map({1: 1, 2: 0})
labels = np.asarray(features.pop(sensitive_column))

x_train = np.asarray(features)
preds = np.round(attack_model.predict(x_train))
preds = [p[0] for p in preds]
reconst_attr = np.asarray(preds)

att_acc = np.sum(reconst_attr == labels)/len(reconst_attr)

print("Attaque precision is ==> ", format(att_acc*100, ".2f"), "%")



