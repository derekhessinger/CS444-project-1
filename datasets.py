'''datasets.py
Loads and preprocesses datasets for use in neural networks.
Alex and Derek
CS444: Deep Learning
'''
import tensorflow as tf
import numpy as np


def load_dataset(name):
    '''Uses TensorFlow Keras to load and return  the dataset with string nickname `name`.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.

    Returns:
    --------
    x: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set (preliminary).
    y: tf.constant. tf.int32s.
        The training set int-coded labels (preliminary).
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.

    Summary of preprocessing steps:
    -------------------------------
    1. Uses tf.keras.datasets to load the specified dataset training set and test set.
    2. Loads the class names from the .txt file downloaded from the project website with the same name as the dataset
        (e.g. cifar10.txt).
    3. Features: Converted from UINT8 to tf.float32 and normalized so that a 255 pixel value gets mapped to 1.0 and a
        0 pixel value gets mapped to 0.0.
    4. Labels: Converted to tf.int32 and flattened into a tensor of shape (N,).

    Helpful links:
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
    '''

    # check that name passed is valid
    if name.lower() != 'cifar10' and name.lower() != 'mnist':
        raise Exception("Data must be cifar10 or mnist")
    
    # load in dataset
    if name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        cifar = True
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[:, :, :, tf.newaxis]
        x_test = x_test[:, :, :, tf.newaxis]
        cifar = False
    
    # load in class names
    if cifar:
        class_names = np.loadtxt('data/cifar10.txt', dtype=str)
    else:
        class_names = np.loadtxt('data/mnist.txt', dtype=str)
    
    # convert features and normalize
    x_train = tf.cast(x_train, tf.float32) / 255.0
    x_test = tf.cast(x_test, tf.float32) / 255.0
    
    # convert labels and flatten
    y_train = tf.reshape(tf.cast(y_train, tf.int32), [-1])
    y_test = tf.reshape(tf.cast(y_test, tf.int32), [-1])
    
    return x_train, y_train, x_test, y_test, class_names




def standardize(x_train, x_test, eps=1e-10):
    '''Standardizes the image features using the global RGB triplet method.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    x_test: tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Test set features.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Standardized training set features (preliminary).
    tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Standardized test set features (preliminary).
    '''
    # Compute mean and standard deviation across all images but per channel
    mean = tf.reduce_mean(tf.concat([x_train,x_test], axis=0), axis=[0, 1, 2])  # Shape: (1, 1, 1, n_chans)
    std = tf.math.reduce_std(x_train, axis=[0, 1, 2])  # Shape: (1, 1, 1, n_chans)

    # Standardize the datasets
    standardized_x_train = (x_train - mean) / (std + eps)
    standardized_x_test = (x_test - mean) / (std + eps)

    return standardized_x_train, standardized_x_test

def train_val_split(x_train, y_train, val_prop=0.1):
    '''Subdivides the preliminary training set into disjoint/non-overlapping training set and validation sets.
    The val set is taken from the end of the preliminary training set.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    y_train: tf.constant. tf.int32s. shape=(N_train,).
        Training set class labels (preliminary).
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features.
    tf.constant. tf.int32s. shape=(N_train,).
        Training set labels.
    tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    '''
    # calculate val size
    n_val = round(x_train.shape[0] * val_prop)
    
    # split the data
    x_train_final = x_train[:-n_val]
    y_train_final = y_train[:-n_val]
    
    x_val = x_train[-n_val:]
    y_val = y_train[-n_val:]
    
    return x_train_final, y_train_final, x_val, y_val


def get_dataset(name, standardize_ds=True, val_prop=0.1):
    '''Automates the process of loading the requested dataset `name`, standardizing it (optional), and create the val
    set.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.
    standardize_ds: bool.
        Should we standardize the dataset?
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set.
    y_train: tf.constant. tf.int32s.
        The training set int-coded labels.
    x_val: tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    y_val: tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.
    '''
    #load
    x_train, y_train, x_test, y_test, class_names = load_dataset(name)

    #standardize
    if(standardize_ds):
        x_train, x_test = standardize(x_train, x_test)

    #val_split
    x_train_final, y_train_final, x_val, y_val = train_val_split(x_train, y_train, val_prop)

    return x_train_final, y_train_final, x_val, y_val, x_test, y_test, class_names
