from os.path import exists, join
import numpy as np


seprator = "\t"
dataset_folder = "lfw"
train_file = "pairsDevTrain.txt"
test_file = "pairsDevTest.txt"

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