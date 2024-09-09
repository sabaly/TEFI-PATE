"""
    @author : thierno-mamoudou.sabaly@telecom-sudparis.eu

    This file contains functions for the student needs as loading student dataset, training a student model, and so on.
"""

import tensorflow as tf
from folktables import ACSDataSource, ACSEmployment, ACSIncome, BasicProblem, employment_filter, adult_filter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def define_model(input_shape):
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.models.Sequential([
        tf.keras.Input(input_shape),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Recall(name="recall")])
    
    return model

def train_student(x_train, y_train, nb_epochs=100, verbose=True):
    model = define_model(x_train.shape[1:])
    if verbose:
        print("Training student...", end="")

    model.fit(x_train, y_train, epochs = nb_epochs, verbose=False, shuffle=False)
    if verbose:
        print("Done")
    return model

def eval_student_model(model, x_test, true_y_test, y_test, verbose=True):
    if verbose:
        print('Test 1 : evaluation the student on aggregated labels')
    eval1 = model.evaluate(x_test, y_test)
    if verbose:
        print(f"**** Results \n\t-loss : {eval1[0]}\n\t-accuracy : {eval1[1]}")
        print("-------------------")
        print('Test 2 : evaluation the student on true labels')
    eval2 = model.evaluate(x_test, true_y_test)
    if verbose:
        print(f"**** Results \n\t-loss : {eval2[0]}\n\t-accuracy : {eval2[1]}")
    return eval1, eval2
    
def load_student_data(state, year=2018, horizon="1-Year", attr="sex", alpha=[], conf=0):
    if alpha == []:
        return load_b_student_data(state, year, horizon, attr)
    else:
        return load_unb_student_data(state, year, horizon, attr, alpha)

def load_b_student_data(state, year=2018, horizon="1-Year", attr="sex"):
    data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    acs_data = data_src.get_data(states=[state], download=True)
    MyACSEMployment = ACSEmployment
    if attr=="sex":
        ACSEmploymentNew = BasicProblem(
            features = ACSEmployment.features,
            target = ACSEmployment.target,
            target_transform=ACSEmployment.target_transform,
            group = "SEX",
            preprocess=employment_filter,
            postprocess=lambda x: np.nan_to_num(x, -1)
        )
        MyACSEMployment = ACSEmploymentNew
    features, labels, group = MyACSEMployment.df_to_numpy(acs_data)
    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, group, test_size=0.2, random_state=0
        )
    return (x_train, x_test, y_train, y_test, s_train, s_test)

def load_unb_student_data(state, year=2018, horizon="1-Year", attr = "sex", alpha=[100,100]):
    data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    acs_data = data_src.get_data(states=[state], download=True)
    MyACSEMployment = ACSEmployment
    if attr == "sex":
        ACSEmploymentNew = BasicProblem(
            features = ACSEmployment.features,
            target = ACSEmployment.target,
            target_transform=ACSEmployment.target_transform,
            group = "SEX",
            preprocess=employment_filter,
            postprocess=lambda x: np.nan_to_num(x, -1)
        )
        MyACSEMployment = ACSEmploymentNew
    features, labels, group = MyACSEMployment.df_to_numpy(acs_data)
    df = pd.DataFrame(features)
    df.columns = MyACSEMployment.features
    df[MyACSEMployment.target] = labels

    p_grp_pr = df[(df[MyACSEMployment.group] == 1) & (df[MyACSEMployment.target] == True)]
    up_grp_pr = df[(df[MyACSEMployment.group] == 2) & (df[MyACSEMployment.target] == True)]
    rest_of_df = df[((df[MyACSEMployment.group] != 1) & (df[MyACSEMployment.group] != 2)) | (df[MyACSEMployment.target] == False)]
    p_vs_up = pd.concat([p_grp_pr, up_grp_pr])

    dist = np.random.dirichlet(alpha, 1)
    size_p_grp = int(dist[0][0]*p_vs_up.shape[0])
    size_up_grp = p_vs_up.shape[0]-size_p_grp

    p_grp = p_grp_pr.sample(size_p_grp, replace=True)
    up_grp = up_grp_pr.sample(size_up_grp, replace=True)
    final_df = pd.concat([p_grp, up_grp, rest_of_df])

    labels = np.array(final_df.pop(MyACSEMployment.target))
    features = np.asarray(final_df.copy())
    group = np.asarray(final_df[MyACSEMployment.group])

    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, group, test_size=0.2, random_state=0
        )
    return (x_train, x_test, y_train, y_test, s_train, s_test)

# %not used !
def load_st_income(state, year=2018, horizon="1-Year", alpha=[]):
    data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    acs_data = data_src.get_data(states=[state], download=True)
    MyACSIncome = BasicProblem(
        features = ACSIncome.features,
        target = ACSIncome.target,
        target_transform=ACSIncome.target_transform,
        group = "SEX",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1)
    )
    features, labels, group = MyACSIncome.df_to_numpy(acs_data)
    if alpha == []:
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
                features, labels, group, test_size=0.2, random_state=0
            )
    else:
        features, labels, group = MyACSIncome.df_to_numpy(acs_data)
        df = pd.DataFrame(features)
        df.columns = MyACSIncome.features
        df[MyACSIncome.target] = labels

        p_grp_pr = df[(df[MyACSIncome.group] == 1) & (df[MyACSIncome.target] == True)]
        up_grp_pr = df[(df[MyACSIncome.group] == 2) & (df[MyACSIncome.target] == True)]
        rest_of_df = df[((df[MyACSIncome.group] != 1) & (df[MyACSIncome.group] != 2)) | (df[MyACSIncome.target] == False)]
        p_vs_up = pd.concat([p_grp_pr, up_grp_pr])

        dist = np.random.dirichlet(alpha, 1)
        size_p_grp = int(dist[0][0]*p_vs_up.shape[0])
        size_up_grp = p_vs_up.shape[0]-size_p_grp

        p_grp = p_grp_pr.sample(size_p_grp, replace=True)
        up_grp = up_grp_pr.sample(size_up_grp, replace=True)
        final_df = pd.concat([p_grp, up_grp, rest_of_df])

        labels = np.array(final_df.pop(MyACSIncome.target))
        features = np.asarray(final_df.copy())
        group = np.asarray(final_df[MyACSIncome.group])

        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
                features, labels, group, test_size=0.2, random_state=0
            )
    return (x_train, x_test, y_train, y_test, s_train, s_test)