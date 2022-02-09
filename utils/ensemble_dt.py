from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

def get_populated_dtc(max_depth=8, min_samples_split=2, min_samples_leaf=4, min_impurity_decrease=0.0):
    #Return a DecisionTreeClassifier object
    dtc = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease)
    return dtc

def adaboosted_dt():
    ab_dt = AdaBoostClassifier(
        get_populated_dtc(), n_estimators=100
    )

    return ab_dt.fit(train_data, train_label)

def test_accuracy(trained_model):
    test1 = accuracy_score(test_label, trained_model.predict(test_data))
    train1 = accuracy_score(train_label, trained_model.predict(train_data))
    return test1, train1