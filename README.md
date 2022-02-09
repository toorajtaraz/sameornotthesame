
# Table of Contents

1.  [The general idea](#org37498be)
2.  [How these scripts work?](#org1ff451e)
    1.  [Loading data](#org066ca43)
        1.  [Needed modules and global variables](#org5ba376d)
        2.  [Parsing the arrays in dataset](#org116fc9e)
        3.  [Extracting arrays for parsing](#orgbc8b05e)
        4.  [Loading the dataset](#org88efcd0)
    2.  [Adaboosted Decision Trees](#orgc993793)
        1.  [Needed modules and global variables](#org46e10d3)
        2.  [Initializing global variables](#org21daf4d)
        3.  [Initializing the meta classifier](#orgc9e175a)
    3.  [Random Forrest](#org9a88aa8)
        1.  [Needed modules and global variables](#orgb8634f7)
        2.  [Initializing global variables](#org179e501)
        3.  [Initializing the meta classifier](#org7da212c)
    4.  [Bagged MLPs](#org0e73693)
        1.  [Needed modules and global variables](#org52219a5)
        2.  [Initializing global variables](#org27add3d)
        3.  [Initializing the meta classifier](#orgc385a05)
    5.  [Some sort of bootstraping + Soft voting](#orgb01a251)
        1.  [Needed modules and global variables](#org51dbaf9)
        2.  [Initializing global variables](#org5562807)
        3.  [The pipeline](#org560c4c9)
    6.  [The main script](#orgeae4308)
        1.  [Needed modules](#orgb0e7a47)
        2.  [Loading and manuplating the data](#org7d00263)
        3.  [Preprocessing data](#org2e0e2f4)
        4.  [Using what have created so far](#org6f88f4a)
    7.  [Results](#orgadf5214)
        1.  [Overall view](#org6a6248e)
        2.  [Detailed discussion](#org7c03850)



<a id="org37498be"></a>

# The general idea

The general idea behind ensemble learning is that instead of a specific algorithm for classification, we have a meta classifier that takes advantage of couple of classic classifier.
In this project we have these 4:

1.  Adaboosted decision trees
2.  Random forrest
3.  Bagged MLPs
4.  Some sort of bootstraping + Soft voting

It&rsquo;s worth mentioning that every hard coded parameter is retrieved from previous projects and tests.


<a id="org1ff451e"></a>

# How these scripts work?


<a id="org066ca43"></a>

## Loading data


<a id="org5ba376d"></a>

### Needed modules and global variables

    from os.path import exists, join
    import numpy as np
    
    
    seprator = "\t"
    dataset_folder = "lfw"
    train_file = "pairsDevTrain.txt"
    test_file = "pairsDevTest.txt"


<a id="org116fc9e"></a>

### Parsing the arrays in dataset

    def parse_array(stringed_array):
        #parse the stringed array
        #return the array
        result = []
        for string in stringed_array:
            if "[" in string:
                string = string[2:]
            if "]" in string:
                string = string[:-2]
            for token in string.split():
                result.append(float(token))
        return np.array(result)


<a id="orgbc8b05e"></a>

### Extracting arrays for parsing

We need to figure out path to each txt file in dataset in order to load them as a string and then parse them and load them into memory using previous function.

    def extract_array_from_line(path, line):
        #Name _ Pic.No1 _ Pic.No2
        #extract array from Pics and return them
        try:
            name, pic1, pic2 = line.split(seprator)
            pic1 = int(pic1)
            pic2 = int(pic2)
            pic1 = f'{name}_{pic1:04d}.txt'
            pic2 = f'{name}_{pic2:04d}.txt'
    
            final_path = join(path, name)
            final_pic1 = join(final_path, pic1)
            final_pic2 = join(final_path, pic2)
            parsed_pic1 = open(final_pic1, 'r').readlines()
            parsed_pic2 = open(final_pic2, 'r').readlines()
    
    
            parsed_pic1 = parse_array(parsed_pic1)
            parsed_pic2 = parse_array(parsed_pic2)
            return np.array([parsed_pic1, parsed_pic2])
        except Exception:
            name1, pic1, name2, pic2 = line.split(seprator)
            pic1 = int(pic1)
            pic2 = int(pic2)
            pic1 = f'{name1}_{pic1:04d}.txt'
            pic2 = f'{name2}_{pic2:04d}.txt'
    
            final_path1 = join(path, name1)
            final_path2 = join(path, name2)
    
            final_pic1 = join(final_path1, pic1)
            final_pic2 = join(final_path2, pic2)
    
            parsed_pic1 = open(final_pic1, 'r').readlines()
            parsed_pic2 = open(final_pic2, 'r').readlines()
    
    
            parsed_pic1 = parse_array(parsed_pic1)
            parsed_pic2 = parse_array(parsed_pic2)
            return np.array([parsed_pic1, parsed_pic2])


<a id="org88efcd0"></a>

### Loading the dataset

Now we use all the functions above to load our dataset.

    def load(path):
        #check if the file exists
        # if not, return None
        # if yes, load the data
        # return the data
        if not exists(path):
            return None
    
        data_path = join(path, dataset_folder)
        train_path = join(path, train_file)
        test_path = join(path, test_file)
        train_data_plus = []
        test_data_plus = []
        train_data_negative = []
        test_data_negative = []
        train_path_handle = open(train_path, 'r')
        test_path_handle = open(test_path, 'r')
    
        for i, line in enumerate(train_path_handle.readlines()):
            if i == 0:
                count = int(line)
                continue
            if i <= count:
                train_data_plus.append(extract_array_from_line(data_path, line))
            else:
                train_data_negative.append(extract_array_from_line(data_path, line))
    
        for i, line in enumerate(test_path_handle.readlines()):
            if i == 0:
                count = int(line)
                continue
            if i <= count:
                test_data_plus.append(extract_array_from_line(data_path, line))
            else:
                test_data_negative.append(extract_array_from_line(data_path, line))
    
        return np.array(train_data_plus), np.array(train_data_negative), np.array(test_data_plus), np.array(test_data_negative)


<a id="orgc993793"></a>

## Adaboosted Decision Trees

Sklearn&rsquo;s implementation of all meta classifiers is used in this project.


<a id="org46e10d3"></a>

### Needed modules and global variables

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    
    train_data = None
    train_label = None
    test_data = None
    test_label = None


<a id="org21daf4d"></a>

### Initializing global variables

The way I&rsquo;ve modeled the data for this meta classifier, forms a vector from concatenating vector of each image, In other words imagine we have extracted parts of people&rsquo;s DNA and we want to know whether they are related or not, First we form a table containing DNA parts of each of those people&rsquo;s DNA in each row and we want our decision tree to figure out existence of any blood relationship. We are doing the samething here, each vector being image&rsquo;s DNA and their concatenation being each row. We transform the data into our desired shape and then use this function to load them.

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


<a id="orgc9e175a"></a>

### Initializing the meta classifier

First we need to construct a decision tree classifier with our desired parameters and then passing that to our meta classifier and at the end train the meta classifier and the measure its performance. That&rsquo;s what we do for all of meta classifiers in this project.

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

I&rsquo;ll talk about its results and performance at the end of this document.


<a id="org9a88aa8"></a>

## Random Forrest


<a id="orgb8634f7"></a>

### Needed modules and global variables

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    train_data = None
    train_label = None
    test_data = None
    test_label = None


<a id="org179e501"></a>

### Initializing global variables

Here I&rsquo;ve experimented with a different way of modeling data, I&rsquo;ve imagined that each vector is not just an array of features but it actually represents and actual vector in a 512 dimension and their differences and their distance can mean something, So this time I create a vector of size 513 which its first 512 elements are absolute value of two vectors and its last element is their distance.

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


<a id="org7da212c"></a>

### Initializing the meta classifier

    def test_accuracy(trained_model):
        test1 = accuracy_score(test_label, trained_model.predict(test_data))
        train1 = accuracy_score(train_label, trained_model.predict(train_data))
        return test1, train1
    
    def handle_random_forrest():
        rfc = RandomForestClassifier(max_depth=8, min_samples_split=2, min_samples_leaf=4, min_impurity_decrease=0.0)
        return rfc.fit(train_data, train_label)


<a id="org0e73693"></a>

## Bagged MLPs


<a id="org52219a5"></a>

### Needed modules and global variables

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, adjusted_rand_score
    
    train_data = None
    train_label = None
    test_data = None
    test_label = None


<a id="org27add3d"></a>

### Initializing global variables

Here I&rsquo;ve used the same modeling as what I used in Adaboosted Decision Trees.

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


<a id="orgc385a05"></a>

### Initializing the meta classifier

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


<a id="orgb01a251"></a>

## Some sort of bootstraping + Soft voting

Here I&rsquo;ve taken advantage of three different classifiers:

1.  SVM
2.  MLP
3.  DT

I loop over them and in each iteration train them on a random subset of dataset and modify an array of weights based on their performance and at the very end I pass them to a VotingClassifier and train that meta classifier on the whole dataset.


<a id="org51dbaf9"></a>

### Needed modules and global variables

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


<a id="org5562807"></a>

### Initializing global variables

Here I&rsquo;ve used same modeling as RandomForrest.

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


<a id="org560c4c9"></a>

### The pipeline

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


<a id="orgeae4308"></a>

## The main script


<a id="orgb0e7a47"></a>

### Needed modules

    import numpy as np
    from utils.data_loader import load
    import utils.ensemble_dt as edt
    import utils.ensemble_bag_mlp as ebm
    import utils.svm_mlp_dt_combo as smdt
    import utils.ensemble_rf as erf
    from sklearn import preprocessing
    from tabulate import tabulate


<a id="org7d00263"></a>

### Loading and manuplating the data

Here I transform loaded data into the form I explained in AdaBoostClassifier section.

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


<a id="org2e0e2f4"></a>

### Preprocessing data

Sklearn Library offers a module that takes care of standardizing the data, this action improves convergence time and accuracy (at least based on what I witnessed).

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    
    scaler = preprocessing.StandardScaler().fit(test_data)
    test_data = scaler.transform(test_data)


<a id="org6f88f4a"></a>

### Using what have created so far

    edt.init_vars(train_data, train_label, test_data, test_label)
    adaboosted_dt_acc = edt.test_accuracy(edt.adaboosted_dt())
    
    ebm.init_vars(train_data, train_label, test_data, test_label)
    bagged_mlp_acc = ebm.test_accuracy(ebm.bagged_mlp())
    
    smdt.init_vars(train_data, train_label, test_data, test_label)
    svm_mlp_dt_combo_acc = smdt.test_accuracy(smdt.pipeline(debug=True))
    
    erf.init_vars(train_data, train_label, test_data, test_label)
    random_forrest_acc = erf.test_accuracy(erf.handle_random_forrest())


<a id="orgadf5214"></a>

## Results


<a id="org6a6248e"></a>

### Overall view

<!-- This HTML table template is generated by emacs 27.2 -->
<table border="1">
  <tr>
    <td align="left" valign="top">
      TYPE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    <td align="left" valign="top">
      TEST_P
    </td>
    <td align="left" valign="top">
      TRAIN_P&nbsp;
    </td>
  </tr>
  <tr>
    <td align="left" valign="top">
      AdaboostedDecisionTree
    </td>
    <td align="left" valign="top">
      0.578&nbsp;
    </td>
    <td align="left" valign="top">
      &nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;
    </td>
  </tr>
  <tr>
    <td align="left" valign="top">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BaggedMLP&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    <td align="left" valign="top">
      0.766&nbsp;
    </td>
    <td align="left" valign="top">
      0.997273
    </td>
  </tr>
  <tr>
    <td align="left" valign="top">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SVM_MLP_DT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    <td align="left" valign="top">
      0.844&nbsp;
    </td>
    <td align="left" valign="top">
      0.963636
    </td>
  </tr>
  <tr>
    <td align="left" valign="top">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RandomForest&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    <td align="left" valign="top">
      &nbsp;0.81&nbsp;
    </td>
    <td align="left" valign="top">
      0.985909
    </td>
  </tr>
</table>


<a id="org7c03850"></a>

### Detailed discussion

In case of AdaboostedDecisionTree, it performed as I excepted, very well on train dataset and poorly on test dataset due to over fitting.
BaggedMLPs didn&rsquo;t perform very well as excepted, just about 2000 data samples are not enough for training a MLP. At first I intended to extract data from dataset folder and train my model on a larger dataset, but I figured due to our point being related to the accuracy we achieve it wouldn&rsquo;t be ethical :))
RandomForest and my customish algorithm performed better than the others.
During my tests I concluded that SVM with RBF kernel performs the best and the ensemble final accuracy is really close to SVM&rsquo;s accuracy, but I assume it wouldn&rsquo;t be the case if we had a large enough dataset, in that case MLP would be the dominant model in voting.

