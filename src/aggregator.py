import numpy as np
from analysis import mean

teachers = []
def update_teachers(tchrs):
    global teachers
    teachers = tchrs

fairness_metrics = []
def set_metrics(metrics):
    global fairness_metrics
    fairness_metrics = metrics

def laplacian_noisy_vote(data_to_label, gamma=0.1, voters=[]):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    for x in range(np.shape(preds)[1]):
        n_y_x = np.bincount(preds[:,x])
        # adding noise to votes
        for i in range(len(n_y_x)):
            n_y_x[i] += np.random.laplace(loc=0.0, scale=gamma)
        labels.append(np.argmax(n_y_x))
    return labels

def gaussian_noisy_vote(data_to_label, mu=0, sigma=1, voters=[]):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    for x in range(np.shape(preds)[1]):
        n_y_x = np.bincount(preds[:,x])
        # adding noise to votes
    
        n_y_x = n_y_x + np.random.normal(mu, sigma, len(n_y_x))
        labels.append(np.argmax(n_y_x))
    return labels

def plurality(data_to_label, beta=-1, voters=[]):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        if isinstance(pred[0], np.ndarray):
            preds.append([p[0] for p in pred])
        else:
            preds.append(pred)
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    consent = []
    for x in range(np.shape(preds)[1]):
        n_y_x = np.bincount(preds[:,x])
        if len(n_y_x) == 1:
            c = 0
        else:
            c = n_y_x[1]/sum(n_y_x)
        label = np.argmax(n_y_x)
        if not label:
            c = 1 - c
        consent.append(c)
        labels.append(label)
    return np.asarray(labels), np.mean(consent)

methode = plurality
def only_fair(data_to_label):
    global teachers
    to_ban = []
    for i in range(len(teachers)):
        if fairness_metrics[i] >= 0.1:
            to_ban.append(teachers[i])
    voters = list(set(teachers) - set(to_ban))
    return methode(data_to_label, voters=voters)

def only_unfair(data_to_label):
    global teachers
    to_ban = []
    for i in range(len(teachers)):
        if fairness_metrics[i] < 0.1:
            to_ban.append(teachers[i])
    voters = list(set(teachers) - set(to_ban))
    return methode(data_to_label, voters=voters)

def weighed_vote(data_to_label, voters=[], fairness = fairness_metrics):
    if fairness == []:
        fairness = fairness_metrics.copy()
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = list(np.round(pred))
        if isinstance(pred[0], np.ndarray):
            preds.append([p[0] for p in pred])
        else:
            preds.append(pred)
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    consent = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            if fairness[i] < 0.1:
                pred.append(pred[i])
        n_y_x = np.bincount(pred)
        if len(n_y_x) == 1:
            c = 0
        else:
            c = n_y_x[1]/sum(n_y_x)
        label = np.argmax(n_y_x)
        if not label:
            c = 1 - c
        consent.append(c)
        labels.append(np.argmax(n_y_x))
    return np.asarray(labels), np.mean(consent)

# ######################
# FairFed weighs
# ######################
def computes_weigh(teachers, beta=10, gamma=100, quota=20):
    # computes global metric
    fg = 0
    sum_ni = 0
    for tchr in teachers:
        fg += tchr.local_m
        nk = tchr.splited_data[1].shape[0]
        sum_ni += nk
    fg = abs(fg)
    ws = [0]*len(teachers)
    for i in range(len(teachers)):
        nk = teachers[i].splited_data[1].shape[0]
        ws[i] = np.exp(-beta*abs(teachers[i].metrics["SPD"] - fg)) #* nk/sum_ni

    for i in range(len(ws)):
        w = int(np.floor(gamma*ws[i]))
        w = quota if w > quota else w
        ws[i] = w
    #print(ws)
    return ws

def fair_fed_agg(data_to_label, beta=5, voters=[], fairness=fairness_metrics):
    # predictions from teachers
    if fairness == []:
        fairness = fairness_metrics.copy()
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        if isinstance(pred[0], np.ndarray):
            preds.append([p[0] for p in pred])
        else:
            preds.append(pred)
    preds = np.asarray(preds, dtype=np.int32)

    # computes weighs
    weighs = computes_weigh(voters, beta=beta)
	
    labels = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            pred = pred + [pred[i]]*weighs[i]
        n_y_x = np.bincount(pred)
        labels.append(np.argmax(n_y_x))
    return np.asarray(labels), weighs

# ######################
# Spd weighs
# ######################
def get_weighs(fairness=[]):
    ws = []
    for i in range(len(fairness)):
        w = int(np.floor(1/fairness[i]))
        ws.append(w)

    #print(ws)
    return ws

def get_weighs_2(gamma=10, beta=10, fairness=[]):
    ws = []
    for i in range(len(fairness)):
        w = int(gamma * np.exp(-beta * fairness[i]))
        ws.append(w)
        
    return ws

def wv1_aggregator(data_to_label, voters=[], fairness=fairness_metrics):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    if fairness == []:
        fairness = fairness_metrics.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        if isinstance(pred[0], np.ndarray):
            preds.append([p[0] for p in pred])
        else:
            preds.append(pred)
    preds = np.asarray(preds, dtype=np.int32)

    weighs = get_weighs(fairness=fairness)
    labels = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            pred = pred + [pred[i]]*weighs[i]
        n_y_x = np.bincount(pred)
        labels.append(np.argmax(n_y_x))
    #print(labels[:100])
    return np.asarray(labels), weighs

def wv2_aggregator(data_to_label, voters=[],  beta=10, fairness=fairness_metrics):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    if fairness == []:
        fairness = fairness_metrics.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        if isinstance(pred[0], np.ndarray):
            preds.append([p[0] for p in pred])
        else:
            preds.append(pred)
    preds = np.asarray(preds, dtype=np.int32)

    weighs = get_weighs_2(beta=beta, fairness=fairness)
    labels = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            pred = pred + [pred[i]]*weighs[i]
        n_y_x = np.bincount(pred)
        labels.append(np.argmax(n_y_x))
        
    return np.asarray(labels), weighs
