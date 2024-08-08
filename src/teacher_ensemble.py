import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from analysis import fairness, mean
import tensorflow as tf
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import pickle
from random import choice, seed, random
from folktables import ACSDataSource, ACSEmployment
import numpy as np
from time import sleep


states = ["HI", "CA", "PR", "NV", "NM", "OK", "NY", "WA", "AZ",  "MD",
"TX", "VA", "MA", "GA", "CT", "OR", "IL", "RI", "NC", "CO", "DE", "LA", "UT",
"FL", "MS", "SC", "AR", "SD", "AL", "MI", "KS", "ID", "MN", "TN", "OH", "IN",
"MT", "PA", "NE", "MO", "WY", "ND", "WI", "KY", "NH", "ME", "IA", "VT", "WV"] 

data_src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
alphas = [[100, 100]]*(len(states) - 1)
alphas[9] = [100,180]

class Teacher:
    def __init__(self, id: int, fair=True):
        self.tchr_id = id
        self.local_s = []
        self.local_m = 0
        self.metrics = {}
        self.status = fair
        self.dataset = self.get_dataset()
        self.splited_data = () # ( x_train, x_test, y_train, y_test, s_train, s_test )
        self.split_dataset()

        self.nk = self.splited_data[1].shape[0]

    def define_model(self):
        input_shape = self.splited_data[0].shape[1:]
        tf.keras.utils.set_random_seed(self.tchr_id)
        model = tf.keras.models.Sequential([
            tf.keras.Input(input_shape),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Recall(name="recall")])
        
        return model
    
    def get_dataset(self):
        df = data_src.get_data(states=[states[self.tchr_id]], download=True)
        df[ACSEmployment.group] = [2 if x!= 1 else 1 for x in df[ACSEmployment.group]]
        features, labels, group = ACSEmployment.df_to_numpy(df)
        if not self.status: 
            df = pd.DataFrame(features)
            df.columns = ACSEmployment.features
            df[ACSEmployment.target] = labels

            p_grp_pr = df[(df[ACSEmployment.group] == 1) & (df[ACSEmployment.target] == True)]
            up_grp_pr = df[(df[ACSEmployment.group] == 2) & (df[ACSEmployment.target] == True)]
            rest_of_df = df[((df[ACSEmployment.group] != 1) & (df[ACSEmployment.group] != 2)) | (df[ACSEmployment.target] == False)]
            p_vs_up = pd.concat([p_grp_pr, up_grp_pr])

            alpha = alphas[states.index(states[self.tchr_id])]
            dist = np.random.dirichlet(alpha, 1)
            size_p_grp = int(dist[0][0]*p_vs_up.shape[0])
            size_up_grp = p_vs_up.shape[0]-size_p_grp

            p_grp = p_grp_pr.sample(size_p_grp, replace=True)
            up_grp = up_grp_pr.sample(size_up_grp, replace=True)
            final_df = pd.concat([p_grp, up_grp, rest_of_df])

            labels = np.array(final_df.pop(ACSEmployment.target))
            features = final_df.copy()
            group = final_df[ACSEmployment.group]

        return features, labels, group

    def split_dataset(self):
        features, labels, group = self.dataset
        self.splited_data = train_test_split(
            features, labels, group, test_size=0.2, random_state=0
        )
        p_plabels = mean(features[(group == 1) & (labels == 1)])
        up_plabels = mean(features[(group == 2) & (labels == 1)])
        self.local_s = pd.DataFrame(data={"ID": [self.tchr_id], "P_PLBLS": [p_plabels], "UP_PLBLS": [up_plabels]})

    def train_model(self):
        x_train, x_test, y_train, y_test, _, s_test = self.splited_data
        self.model = self.define_model()
        self.model.fit(x_train, y_train, epochs=60, verbose=False)

        self.metrics = fairness(self.model, x_test, y_test, s_test)
    
    def update_local_m(self, S, sum_n, seen=False):
        _, x_test, _, y_test, _, s_test = self.splited_data
        yhat = np.round(self.model.predict(x_test))
        p_tp = mean(yhat[(s_test == 1) & (y_test==1)])
        up_tp = mean(yhat[(s_test==2) & (y_test==1)])
        p_plabels = S[(S["ID"] == self.tchr_id)]["P_PLBLS"][0]
        up_plabels = S[(S["ID"] == self.tchr_id)]["UP_PLBLS"][0]
        others_p_plabels = sum(S["P_PLBLS"]) - p_plabels
        others_up_plabels = sum(S["UP_PLBLS"]) - up_plabels

        a = p_tp*p_plabels/others_p_plabels
        b = up_tp*up_plabels/others_up_plabels

        self.nk = x_test.shape[0]

        if not isinstance(a, float):
            a.index = [0,1]
            b.index = [0,1]
            if seen:
                a = a[1]
                b = b[1]
            else:
                a = a[0]
                b = b[0]
                
        self.local_m = (b-a)*self.nk/sum_n
    
class Ensemble:
    def __init__(self, nb_teachers, nb_fair_tchrs=-1) -> None:
        self.nb_fair = nb_fair_tchrs
        self.nb_tchrs = nb_teachers
        self.tchrs = []
        self.get_teachers()
        
        self.S = pd.concat([t.local_s for t in self.tchrs])
        self.update_m()
        
    def wrapper(self, teacher):
        teacher.train_model()
        
        return teacher

    def update_m(self):
        sum_n = 0
        for i in range(len(self.tchrs)):
            sum_n += self.tchrs[i].splited_data[1].shape[0]
        seen = []
        for i in range(len(self.tchrs)):
            if self.tchrs[i].tchr_id in seen:
                self.tchrs[i].update_local_m(self.S, sum_n, True)
            else:
                self.tchrs[i].update_local_m(self.S, sum_n)
                seen.append(self.tchrs[i].tchr_id)

    def parallel_training(self):
        print("Training Teachers...")
        with Pool(mp.cpu_count()) as p:
            fi = p.map(self.wrapper, self.tchrs)
            p.close()
        self.tchrs = fi
        self.update_m()

    def get_teachers(self):
        cpy_states = [x for x in states]
        root = "../checkpoint_sex/"
        ind_min = 0
        nb_tchr_pr_grp = self.nb_tchrs // 4
        nb_tchr_grp = 0
        seed(self.nb_tchrs+  self.nb_fair)
        cpy_states = [x for x in states[ind_min:ind_min+12]]
        
        for _ in range(self.nb_fair):
            st = choice(cpy_states)
            cpy_states.pop(cpy_states.index(st))
            path = root + st + "/" + st  + "_fair.pkl"
            
            with open(path, "rb") as f:
                tchr = pickle.load(f)
            self.tchrs.append(tchr)
            
            if cpy_states == []: # model !
                cpy_states = [x for x in states]
            nb_tchr_grp += 1
            if nb_tchr_grp == nb_tchr_pr_grp:
                ind_min += 12
                nb_tchr_grp = 0
                if ind_min >= 47:
                    ind_min = 0
                    cpy_states = [x for x in states]
                else:
                    cpy_states = [x for x in states[ind_min:ind_min+12]]

        #print("HERE ! ", root)
        cpy_states = [x for x in states]
        for _ in range(self.nb_tchrs - self.nb_fair):
            st = choice(cpy_states)
            cpy_states.pop(cpy_states.index(st))
            path = root + st + "/" + st  + "_unfair.pkl"
            with open(path, "rb") as f:
                tchr = pickle.load(f)
            self.tchrs.append(tchr)
            if cpy_states == []: # model !
                cpy_states = [x for x in states]
        
        


