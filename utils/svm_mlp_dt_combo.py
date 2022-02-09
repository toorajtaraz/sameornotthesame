from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

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



def get_random_samples(number_of_samples=3):
    #Return a list of random samples from dataset
    res = []
    for _ in range(number_of_samples):
        indexes = np.random.choice(len(train_data), size=(int(len(train_data) / number_of_samples)), replace=False)
        res.append(indexes)
    return res

def test_accuracy_pipeline(trained_model):
    test1 = accuracy_score(test_label, trained_model.predict(test_data))
    return test1

def test_accuracy(trained_model):
    test1 = accuracy_score(test_label, trained_model.predict(test_data))
    train1 = accuracy_score(train_label, trained_model.predict(train_data))
    return test1, train1

def pipeline(debug=False):
    svm = SVC(decision_function_shape="ovr", kernel="rbf", probability=True)
    mlp = MLPClassifier(hidden_layer_sizes=(40, 30, 20, 30, 40), alpha=0.032, random_state=1, solver="adam", activation="relu", max_iter=500)
    dt = DecisionTreeClassifier(max_depth=64, min_samples_split=2, min_samples_leaf=4, min_impurity_decrease=0.0)
    weights = [1 / 3, 1 / 3, 1 / 3]
    last_acc = [0, 0, 0]
    for _ in range(10):
        samples = get_random_samples()
        svm.fit(train_data[samples[0]], train_label[samples[0]])
        mlp.fit(train_data[samples[1]], train_label[samples[1]])
        dt.fit(train_data[samples[2]], train_label[samples[2]])
        svm_acc = test_accuracy_pipeline(svm)
        mlp_acc = test_accuracy_pipeline(mlp)
        dt_acc = test_accuracy_pipeline(dt)
        if svm_acc < last_acc[0]:
            weights[0] -= 0.05
        elif mlp_acc < last_acc[1]:
            weights[1] -= 0.05
        elif dt_acc < last_acc[2]:
            weights[2] -= 0.05
        if svm_acc > last_acc[0]:
            weights[0] += 0.05
        elif mlp_acc > last_acc[1]:
            weights[1] += 0.05
        elif dt_acc > last_acc[2]:
            weights[2] += 0.05
        last_acc = [svm_acc, mlp_acc, dt_acc]
        if debug:
            print("**SVM: {}, MLP: {}, DT: {}**".format(svm_acc, mlp_acc, dt_acc))
        svm_acc = svm_acc * weights[0]
        mlp_acc = mlp_acc * weights[1]
        dt_acc = dt_acc * weights[2]
        if debug:
            print("##SVM: {}, MLP: {}, DT: {}##".format(svm_acc, mlp_acc, dt_acc))
        if svm_acc > mlp_acc and svm_acc > dt_acc:
            weights[0] += 0.1
        elif mlp_acc > svm_acc and mlp_acc > dt_acc:
            weights[1] += 0.1
        elif dt_acc > svm_acc and dt_acc > mlp_acc:
            weights[2] += 0.1


    vt_svm_mlp_dt = VotingClassifier(estimators=[
        ('svm', svm), ('mlp', mlp), ('dt', dt)],
        voting='soft', weights=weights, n_jobs=12)
    
    res = vt_svm_mlp_dt.fit(train_data, train_label)
    if debug:
        print("Final accuracy: {}".format(test_accuracy(res)[0]))
    return res
    

    
