# # -*- coding: utf-8 -*-
# """
# Created on Mon Dec 16 14:25:15 2019
#
# @author: islam
# """
# import heapq
#
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
#
# from utillties_gpt import get_instances_with_random_neg_samples, get_test_instances_with_random_samples
# from performance_and_fairness_measures import getHitRatio, getNDCG
# from collaborative_models_gpt import neuralCollabFilter
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# # Set random seed for reproducibility
# def set_random_seed(state=1):
#     gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
#     for set_state in gens:
#         set_state(state)
#
#
# RANDOM_STATE = 1
# set_random_seed(RANDOM_STATE)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # Function to train the NCF model
# def train_epochs(model, df_train, epochs, lr, batch_size, num_negatives, unsqueeze=False):
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
#     model.train()
#     for i in range(epochs):
#         for batch_i in range(0, len(df_train) - batch_size + 1, batch_size):
#             data_batch = df_train.iloc[batch_i:(batch_i + batch_size)].reset_index(drop=True)
#             user_input, item_input, ratings = get_instances_with_random_neg_samples(
#                 data_batch, num_items, num_negatives, device
#             )
#             if unsqueeze:
#                 ratings = ratings.unsqueeze(1)
#             y_hat = model(user_input, item_input)
#             loss = criterion(y_hat, ratings)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print(f"Epoch {i}, Batch Loss: {loss.item()}")
#
#
# # Function to evaluate the model
# def evaluate_model(model, df_val, top_K, random_samples, num_items):
#     model.eval()
#     avg_HR = np.zeros(top_K)
#     avg_NDCG = np.zeros(top_K)
#     for i, row in df_val.iterrows():
#         user_input, item_input = get_test_instances_with_random_samples(row.values, random_samples, num_items, device)
#         y_hat = model(user_input, item_input).cpu().detach().numpy().reshape(-1)
#         item_input = item_input.cpu().detach().numpy().reshape(-1)
#         map_item_score = dict(zip(item_input, y_hat))
#         for k in range(1, top_K + 1):
#             ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
#             gtItem = item_input[0]
#             avg_HR[k - 1] += getHitRatio(ranklist, gtItem)
#             avg_NDCG[k - 1] += getNDCG(ranklist, gtItem)
#     avg_HR /= len(df_val)
#     avg_NDCG /= len(df_val)
#     return avg_HR, avg_NDCG
#
#
# # Load data
# train_data = pd.read_csv("data/movielens/train.csv")
# test_data = pd.read_csv("data/movielens/test.csv")
#
# # Hyperparameters
# emb_size = 128
# hidden_layers = [emb_size, 64, 32, 16]
# output_size = 1
# num_epochs = 25
# learning_rate = 0.001
# batch_size = 2048
# num_negatives = 5
# random_samples = 100
# top_K = 10
#
# # Number of unique users and items
# num_uniqueUsers = train_data["userId"].nunique()
# num_items = train_data["itemId"].nunique()
#
# # Dynamically calculate num_users and num_items based on the dataset
# # Dynamically calculate num_users and num_items based on both train and test data
# num_users = max(train_data["userId"].max(), test_data["userId"].max()) + 1
# num_items = max(train_data["itemId"].max(), test_data["itemId"].max()) + 1
#
# # Initialize the model with dynamically calculated values
# preTrained_NCF = neuralCollabFilter(num_users, num_items, emb_size, hidden_layers, output_size).to(device)
#
#
# # Initialize and train the model
# preTrained_NCF = neuralCollabFilter(num_uniqueUsers, num_items, emb_size, hidden_layers, output_size).to(device)
# train_epochs(preTrained_NCF, train_data, num_epochs, learning_rate, batch_size, num_negatives, unsqueeze=True)
#
# # Save the pre-trained model
# torch.save(preTrained_NCF.state_dict(), "trained-models/preTrained_NCF.pth")
#
# # Evaluate the model
# avg_HR_preTrain, avg_NDCG_preTrain = evaluate_model(preTrained_NCF, test_data, top_K, random_samples, num_items)
#
# # Save evaluation results
# np.savetxt("results/avg_HR_preTrain.txt", avg_HR_preTrain)
# np.savetxt("results/avg_NDCG_preTrain.txt", avg_NDCG_preTrain)



# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:25:15 2019

@author: islam
"""
import heapq
import numpy as np
import pandas as pd
from utillties import get_instances_with_random_neg_samples, get_test_instances_with_random_samples
from old_res.performance_and_fairness_measures import getHitRatio, getNDCG
from old_res.collaborative_models_gpt import matrixFactorization

import torch
import torch.nn as nn
import torch.optim as optim


# Set random seed for reproducibility
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to train the NCF model
def train_epochs(model, df_train, epochs, lr, batch_size, num_negatives, unsqueeze=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for i in range(epochs):
        for batch_i in range(0, len(df_train) - batch_size + 1, batch_size):
            data_batch = df_train.iloc[batch_i:(batch_i + batch_size)].reset_index(drop=True)
            if data_batch.empty:
                continue
            user_input, item_input, ratings = get_instances_with_random_neg_samples(
                data_batch, num_items, num_negatives, device
            )
            if unsqueeze:
                ratings = ratings.unsqueeze(1)
            y_hat = model(user_input, item_input)
            loss = criterion(y_hat, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {i}, Batch Loss: {loss.item()}")


# Function to evaluate the model
def evaluate_model(model, df_val, top_K, random_samples, num_items):
    model.eval()
    avg_HR = np.zeros(top_K)
    avg_NDCG = np.zeros(top_K)
    for i, row in df_val.iterrows():
        user_input, item_input = get_test_instances_with_random_samples(row.values, random_samples, num_items, device)
        y_hat = model(user_input, item_input).cpu().detach().numpy().reshape(-1)
        item_input = item_input.cpu().detach().numpy().reshape(-1)
        map_item_score = dict(zip(item_input, y_hat))
        for k in range(1, top_K + 1):
            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            gtItem = item_input[0]
            avg_HR[k - 1] += getHitRatio(ranklist, gtItem)
            avg_NDCG[k - 1] += getNDCG(ranklist, gtItem)
    avg_HR /= len(df_val)
    avg_NDCG /= len(df_val)
    return avg_HR, avg_NDCG


# Load data
train_data = pd.read_csv("data/last_fm_pop/train.csv")
test_data = pd.read_csv("data/last_fm_pop/test.csv")

# Hyperparameters
emb_size = 128
hidden_layers = [emb_size, 64, 32, 16]
output_size = 1
num_epochs = 1
learning_rate = 0.001
batch_size = 2048
num_negatives = 2
random_samples = 2
top_K = 10

# Dynamically calculate num_users and num_items based on both train and test data
num_users = max(train_data["userId"].max(), test_data["userId"].max()) + 1
num_items = max(train_data["itemId"].max(), test_data["itemId"].max()) + 1

# Initialize the model with dynamically calculated values
# preTrained_NCF = neuralCollabFilter(num_users, num_items, emb_size, hidden_layers, output_size).to(device)

preTrained_NCF = matrixFactorization(num_users, num_items, emb_size).to(device)
# Train the model
train_epochs(preTrained_NCF, train_data, num_epochs, learning_rate, batch_size, num_negatives, unsqueeze=False)

# Save the pre-trained model
torch.save(preTrained_NCF.state_dict(), "trained-models/preTrained_MF_lastfm_pop.pth")

# Evaluate the model
avg_HR_preTrain, avg_NDCG_preTrain = evaluate_model(preTrained_NCF, test_data, top_K, random_samples, num_items)

# Save evaluation results
np.savetxt("results/results_/avg_HR_preTrain_lastfm_pop.txt", avg_HR_preTrain)
np.savetxt("results/results_/avg_NDCG_preTrain_lastfm_pop.txt", avg_NDCG_preTrain)
