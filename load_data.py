
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from randomGeneratedPath import generate_map_xy

# generate_map_xy(5000)

def shuffle_csv_data(path = './nnData.csv', after_shuffled_file_name = 'latestShuffledFile.csv'):
    # data = pd.read_csv('nnData.csv')
    data = pd.read_csv(path)
    #  Shuffle data using sample and reset to new index
    data = data.sample(frac=1).reset_index(drop=True)
    # print ("after: \n" + str(data))
    data.to_csv(after_shuffled_file_name, index=False)
    return after_shuffled_file_name



data = pd.read_csv('shuffledNNData.csv').values
# data = pd.read_csv('nnData.csv').values

def scaler_min_max(data,feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range= feature_range)
    scaler.fit(data)
    data = scaler.transform(data)
    return data


# data = scaler_min_max(data, feature_range = (0,1))

# Train, Validation and Test data, Train: 79%, Validation 7%, Test: 14%
# Train: 79%
train_start = 0
train_end   = int(np.floor(0.79*len(data)))
data_train  = data[np.arange(train_start, train_end), :]
# Validation 7%
valid_start = train_end + 1
valid_end   = int(np.floor((0.79+0.07)*len(data)))
data_valid  = data[np.arange(valid_start, valid_end), :]
# Test: 14%
test_start  = valid_end + 1
test_end    = len(data)
data_test   = data[np.arange(test_start, test_end), :]


# print "data_train: \n" + str(len(data_train))
# print "data_train: \n" + str(len(data_valid))
# print "data_test:  \n" + str(len(data_test))

# # Build input (x,y) and output (label)
points_train_coordinates  = data_train[:, 1:]
points_train_labels       = data_train[:, 0]
points_valid_coordinates  = data_valid[:, 1:]
points_valid_labels       = data_valid[:, 0]
points_test_coordinates   = data_test[:, 1:]
points_test_labels        = data_test[:, 0]

# Transform point coordinates into smaller scale
points_train_coordinates = scaler_min_max(points_train_coordinates, feature_range=(0,1))
points_valid_coordinates = scaler_min_max(points_valid_coordinates, feature_range=(0,1))
points_test_coordinates  = scaler_min_max(points_test_coordinates , feature_range=(0,1))

def convert_to_oneHot(original_array):
    # return original_array
    nb_classes = int(original_array.max()+1)
    original_array.reshape(-1)
    converted_array = np.eye(nb_classes)[original_array.astype(int)]
    return converted_array

def load_data(mode='train', oneHot=True):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    from tensorflow.examples.tutorials.mnist import input_data

    if mode == 'train':
        x_train, y_train, x_valid, y_valid = points_train_coordinates, points_train_labels, \
                                             points_valid_coordinates, points_valid_labels
        if (oneHot==True):
            y_train = convert_to_oneHot(y_train)
            y_valid = convert_to_oneHot(y_valid)
        print y_valid[:]
        print y_train[:]
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = points_test_coordinates, points_test_labels
        if (oneHot==True):
            y_test = convert_to_oneHot(y_test)
        print y_test[:]
        return x_test, y_test


# load_data()
# load_data('test')
