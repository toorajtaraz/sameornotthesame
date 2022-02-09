from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score

train_data = None
train_label = None
test_data = None
test_label = None

def init_vars(_train_data, _train_label, _test_data, _test_label):
    #Initializez global variables
    global train_data
    global train_label
    global test_data
    global test_label
    train_data = _train_data
    train_label = _train_label
    test_data = _test_data
    test_label = _test_label

def get_populated_mlpc():
    #Return a MLPClassifier object
    mlpc = MLPClassifier()
    mlpc.set_params(hidden_layer_sizes=(200, 80, 20), alpha=0.032, random_state=1, solver="adam", activation="relu", max_iter=500)
    return mlpc

def bagged_mlp():
    bg_mlpc = BaggingClassifier(base_estimator= get_populated_mlpc(), n_estimators=20, random_state=1, n_jobs=12)
    return bg_mlpc.fit(train_data, train_label)

def test_accuracy(trained_model):
    test1 = accuracy_score(test_label, trained_model.predict(test_data))
    train1 = accuracy_score(train_label, trained_model.predict(train_data))
    return test1, train1