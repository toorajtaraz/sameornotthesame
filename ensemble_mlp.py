from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

