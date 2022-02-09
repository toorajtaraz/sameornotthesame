import numpy as np
from utils.data_loader import load
import utils.ensemble_dt as edt
import utils.ensemble_bag_mlp as ebm
import utils.svm_mlp_dt_combo as smdt
import utils.ensemble_rf as erf
from sklearn import preprocessing
from tabulate import tabulate

train_plus, train_negative, test_plus, test_negative = load("/home/toorajtaraz/Downloads/project/")
train_data_count = train_plus.shape[0]
test_data_count = test_plus.shape[0]

train_plus = train_plus.reshape((train_data_count, -1))
train_negative = train_negative.reshape((train_data_count, -1))
test_plus = test_plus.reshape((test_data_count, -1))
test_negative = test_negative.reshape((test_data_count, -1))

train_data = []
train_label = []

test_data = []
test_label = []

for x in train_plus:
    train_data.append(x)
    train_label.append(1)

for x in train_negative:
    train_data.append(x)
    train_label.append(0)

for x in test_plus:
    test_data.append(x)
    test_label.append(1)

for x in test_negative:
    test_data.append(x)
    test_label.append(0)

test_data = np.array(test_data)
train_data = np.array(train_data)
test_label = np.array(test_label)
train_label = np.array(train_label)

scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)

scaler = preprocessing.StandardScaler().fit(test_data)
test_data = scaler.transform(test_data)

edt.init_vars(train_data, train_label, test_data, test_label)
adaboosted_dt_acc = edt.test_accuracy(edt.adaboosted_dt())

ebm.init_vars(train_data, train_label, test_data, test_label)
bagged_mlp_acc = ebm.test_accuracy(ebm.bagged_mlp())

smdt.init_vars(train_data, train_label, test_data, test_label)
svm_mlp_dt_combo_acc = smdt.test_accuracy(smdt.pipeline(debug=True))

erf.init_vars(train_data, train_label, test_data, test_label)
random_forrest_acc = erf.test_accuracy(erf.handle_random_forrest())

result = [
    ["Adaboosted Decision Tree", adaboosted_dt_acc[0], adaboosted_dt_acc[1]],
    ["Bagged MLP", bagged_mlp_acc[0], bagged_mlp_acc[1]],
    ["SVM + MLP + DT", svm_mlp_dt_combo_acc[0], svm_mlp_dt_combo_acc[1]],
    ["Random Forest", random_forrest_acc[0], random_forrest_acc[1]]
]
final_table = tabulate(result, headers=['TYPE', "TEST_P", "TRAIN_P"])
print(final_table)