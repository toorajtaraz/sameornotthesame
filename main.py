import numpy as np
from utils.data_loader import load

train_plus, train_negative, test_plus, test_negative = load("/home/toorajtaraz/Downloads/project/")
train_data_count = train_plus.shape[0]
test_data_count = test_plus.shape[0]

train_plus.reshape((train_data_count, -1))
train_negative.reshape((train_data_count, -1))
test_plus.reshape((test_data_count, -1))
test_negative.reshape((test_data_count, -1))

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