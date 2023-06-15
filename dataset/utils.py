import os, re
from typing import List, Dict
from ast import literal_eval
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

folder = './data/experiment'
filename_fields = ['vehicle', 'trajectory', 'method', 'condition']

def extract_features(rawdata, features):
    """extract features from all sources tasks, which is x in algorithm

    Args:
        rawdata (dictionary): _description_
        features (list): the list of names of features that are shared around all sources tasks

    Returns:
        list: the list of data that only contians the selected shared features 
    """
    feature_data = []
    hover_pwm_ratio = 1.
    for feature in features:
      if isinstance(rawdata[feature], str):
        condition_list = re.findall(r'\d+', rawdata[feature]) 
        condition = 0 if condition_list == [] else float(condition_list[0])
        feature_data.append(np.tile(condition,(len(rawdata['v']),1)))
        continue
      feature_len = rawdata[feature].shape[1] if len(rawdata[feature].shape)>1 else 1
      if feature == 'pwm':
          feature_data.append(rawdata[feature] / 1000 * hover_pwm_ratio)
      else:
          feature_data.append(rawdata[feature].reshape(rawdata[feature].shape[0],feature_len))
    feature_data = np.hstack(feature_data)
    return feature_data

def load_data(folder : str, expnames = None) -> List[dict]:
    ''' Loads csv files from {folder} and return as list of dictionaries of ndarrays '''
    Data = []

    if expnames is None:
        filenames = os.listdir(folder)
    elif isinstance(expnames, str): # if expnames is a string treat it as a regex expression
        filenames = []
        for filename in os.listdir(folder):
            if re.search(expnames, filename) is not None:
                filenames.append(filename)
    elif isinstance(expnames, list):
        filenames = (expname + '.csv' for expname in expnames)
    else:
        raise NotImplementedError()
    for filename in filenames:
        # Ingore not csv files, assume csv files are in the right format
        if not filename.endswith('.csv'):
            continue

        # Load the csv using a pandas.DataFrame
        df = pd.read_csv(folder + '/' + filename)

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns[1:]:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # Copy all the data to a dictionary, and make things np.ndarrays
        Data.append({})
        for field in df.columns[1:]:
            Data[-1][field] = np.array(df[field].tolist(), dtype=float)

        # Add in some metadata from the filename
        namesplit = filename.split('.')[0].split('_')
        for i, field in enumerate(filename_fields):
            Data[-1][field] = namesplit[i]
        # Data[-1]['method'] = namesplit[0]
        # Data[-1]['condition'] = namesplit[1]

    return Data

def load_and_process_data(dataset_folder, features):
    ''' 
    Loads data from {dataset_folder} and extracts the features {features}
    :param str dataset_folder: the name of the folder containing the data
    :param list features: the list of features to extract
    :return: the extracted features formated in a numpy array (n_samples, n_features)
    '''
    rawdata = load_data(dataset_folder)[0]
    feature_data = extract_features(rawdata, features)
    print("Data has shape: ", feature_data.shape)
    return feature_data

def generate_orth(shape, seed=None):
    assert len(shape) == 2, "Shape must be a 2-tuple."
    if seed is not None:
        np.random.seed(seed)
    gaus = np.random.normal(0, 1, shape)
    if shape[0] < shape[1]:
        _, _, orth = np.linalg.svd(gaus, full_matrices=False)
        print(f"{orth[[0]]@orth[[1]].T}")
    else:
        orth, _, _,  = np.linalg.svd(gaus, full_matrices=False)
        print(f"{orth[:,[0]].T@orth[:,[1]]}")
    print(f" orth shape: {orth.shape}")
    return orth

def generate_fourier_kernel(input_dim, aug_dim, seed=None):
    '''
    :param input_dim: the dimension of the input data.
    :param aug_dim: the number of random fourier features to generate.
    :return: w, b, and the function that generates the random fourier features.
    '''
    if seed is not None: np.random.seed(seed)
    w = np.random.normal(size=(aug_dim, input_dim))
    if seed is not None: np.random.seed(seed)
    b = np.random.uniform(low=0, high=2*np.pi, size=(aug_dim,1))
    return w, b, lambda x: np.cos(w @ x.T + b).T

def generate_pendulum_specified_kernel(input_dim, aug_dim, seed=None):

    assert input_dim == 6, "input_dim must be 6."
    assert aug_dim == 13, "aug_dim must be 13."

    # non_linear_dim = 2 if input_dim < aug_dim else 0
    # linear_dim = input_dim - non_linear_dim -1
    # n_basis = linear_dim + 5**non_linear_dim
    
    # basis= np.random.uniform(-1, 1, (n_basis, input_dim))

    # aug_w = np.empty((len(basis), aug_dim))
    # aug_w[:, 0:5] = basis[:, :-1]
    # aug_w[:,-1] = basis[:,-1]
    # aug_w[:,5] = basis[:,0]*basis[:,1] 
    # aug_w[:,6] = basis[:,0]**2 
    # aug_w[:,7] = basis[:,0]**2*basis[:,1] 
    # aug_w[:,8] = basis[:,0]**3
    # aug_w[:,9] = basis[:,1]**2 
    # aug_w[:,10] = basis[:,1]**2*w[:,0] 
    # aug_w[:,11] = basis[:,1]**3
    
    # tmp = aug_w[:,:-1]
    # eigs = np.linalg.eigvals(tmp.T @ tmp)

    def pendulum_kernel(w):
        """Map the vector x to a nonlinear space."""
        assert len(w.shape) == 2, "w must be a 2D array."
        assert w.shape[1] == 6, "w must have 6 features."
        # original: Cx, Cy, g, alpha_1, alpha_2, 0 (or 1)
        # now: Cx, Cy, g, alpha_1, alpha_2, CxCy, Cx^2, Cx^2C_y, C_x^3, Cy^2, Cy^2C_x, C_y^3, 0 (or 1)

        aug_w = np.empty((len(w), aug_dim))
        aug_w[:, 0:5] = w[:, :-1]
        aug_w[:,-1] = w[:,-1]
        aug_w[:,5] = w[:,0]*w[:,1] 
        aug_w[:,6] = w[:,0]**2 
        aug_w[:,7] = w[:,0]**2*w[:,1] 
        aug_w[:,8] = w[:,0]**3
        aug_w[:,9] = w[:,1]**2 
        aug_w[:,10] = w[:,1]**2*w[:,0] 
        aug_w[:,11] = w[:,1]**3
        # print(aug_w) #debug

        return aug_w
    
    return aug_dim, pendulum_kernel

# def generate_pendulum_specified_kernel(input_dim, task_aug_dim, seed=None):
#     assert input_dim == 6, "Input_dim must be 6."

#     def pendulum_kernel(x):
#         #only C_x, C_y is nonlinear
#         _, _, fourier_kernel = generate_fourier_kernel(2, task_aug_dim - input_dim, seed=seed)
#         aug_x = np.empty((len(x), task_aug_dim))
#         aug_x[:,-1] = x[:,-1]
#         aug_x[:,0:input_dim-1] = x[:,:-1]
#         aug_x[:,input_dim-1:-1] = fourier_kernel(x[:,:2])
#         aug_x[:,:-1] = aug_x[:,:-1] * (x[:,[-1]] != 1)

#         return aug_x
    
#     return task_aug_dim, pendulum_kernel
