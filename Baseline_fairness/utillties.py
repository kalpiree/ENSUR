# -*- coding: utf-8 -*-
"""
Created on Mon Dec 2 16:28:38 2019

@author: islam
"""
import pandas as pd
import numpy as np
from numpy.random import choice
import torch


# Function to encode columns with continuous IDs
def proc_col(col, train_col=None):
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o: i for i, o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)


# Encode multiple columns in a DataFrame
def encode_data(df, col_names_list, train=None):
    df = df.copy()
    for col_name in col_names_list:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _, col, _ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df


# Generate negative samples for training
def get_instances_with_random_neg_samples(train, num_items, num_negatives, device):
    user_input = np.zeros(len(train) + len(train) * num_negatives)
    item_input = np.zeros(len(train) + len(train) * num_negatives)
    labels = np.zeros(len(train) + len(train) * num_negatives)
    neg_samples = choice(num_items, size=(10 * len(train) * num_negatives,))
    neg_counter = 0
    idx = 0
    for _, row in train.iterrows():
        user_input[idx] = row["userId"]
        item_input[idx] = row["itemId"]
        labels[idx] = 1
        idx += 1
        for _ in range(num_negatives):
            neg_item = neg_samples[neg_counter]
            while neg_item in train.loc[train["userId"] == row["userId"], "itemId"].values:
                neg_counter += 1
                neg_item = neg_samples[neg_counter]
            user_input[idx] = row["userId"]
            item_input[idx] = neg_item
            labels[idx] = 0
            idx += 1
            neg_counter += 1
    return torch.LongTensor(user_input).to(device), torch.LongTensor(item_input).to(device), torch.FloatTensor(
        labels).to(device)


# Generate test instances with random samples
def get_test_instances_with_random_samples(data, random_samples, num_items, device):
    user_input = np.zeros(random_samples + 1)
    item_input = np.zeros(random_samples + 1)
    user_input[0] = data[0]
    item_input[0] = data[1]
    for i in range(1, random_samples + 1):
        neg_item = np.random.randint(num_items)
        while neg_item == data[1]:
            neg_item = np.random.randint(num_items)
        user_input[i] = data[0]
        item_input[i] = neg_item
    return torch.LongTensor(user_input).to(device), torch.LongTensor(item_input).to(device)
