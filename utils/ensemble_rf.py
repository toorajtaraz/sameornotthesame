from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

train_data = None
train_label = None
test_data = None
test_label = None

def preprocess_data(d):
    #Calculate vector distance and absolute value of their difference
    res1 = []
    res2 = []
    for data in d:
        res = np.subtract(data[0], data[1])
        res1.append(np.abs(res))
        res2.append(np.linalg.norm(res))
    res = [np.append(x, y) for x, y in zip(res1, res2)]
    return np.array(res)

def init_vars(_train_data, _train_label, _test_data, _test_label):
    #Initializez global variables
    global train_data
    global train_label
    global test_data
    global test_label
    train_data = _train_data.reshape(len(_train_data), 2, -1)
    train_label = _train_label
    test_data = _test_data.reshape(len(_test_data), 2, -1)
    test_label = _test_label
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

def test_accuracy(trained_model):
    test1 = accuracy_score(test_label, trained_model.predict(test_data))
    train1 = accuracy_score(train_label, trained_model.predict(train_data))
    return test1, train1

def handle_random_forrest():
    rfc = RandomForestClassifier(max_depth=8, min_samples_split=2, min_samples_leaf=4, min_impurity_decrease=0.0)
    return rfc.fit(train_data, train_label)