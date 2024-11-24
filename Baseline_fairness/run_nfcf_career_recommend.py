# -*- coding: utf-8 -*-
"""
Created on Fri May 1 02:13:41 2020

Updated for groupwise evaluation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
import heapq  # for retrieval topK

from utillties import get_instances_with_random_neg_samples, get_test_instances_with_random_samples
from performance_and_fairness_measures import getHitRatio, getNDCG, computeEDF

import torch
import torch.nn as nn
import torch.optim as optim
from collaborative_models import neuralCollabFilter,matrixFactorization

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

set_random_seed()

# Loss function for fairness
def criterionHinge(epsilonClass, epsilonBase):
    zeroTerm = torch.tensor(0.0).to(device)
    return torch.max(zeroTerm, (epsilonClass - epsilonBase))

# Fine-tuning with fairness constraint
def fair_fine_tune_model(
    model, df_train, epochs, lr, batch_size, num_negatives, num_items, protectedAttributes, lamda, epsilonBase, unsqueeze=False
):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()

    all_user_input = torch.LongTensor(df_train["userId"].values).to(device)
    all_item_input = torch.LongTensor(df_train["itemId"].values).to(device)

    for i in range(epochs):
        j = 0
        for batch_i in range(0, len(df_train) - batch_size + 1, batch_size):
            data_batch = df_train.iloc[batch_i : batch_i + batch_size].reset_index(drop=True)
            train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(
                data_batch, num_items, num_negatives, device
            )
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss1 = criterion(y_hat, train_ratings)

            predicted_probs = model(all_user_input, all_item_input)
            avg_epsilon = computeEDF(protectedAttributes, predicted_probs, num_items, all_item_input, device)
            loss2 = criterionHinge(avg_epsilon, epsilonBase)

            loss = loss1 + lamda * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch: {i}, Batch: {j}/{len(df_train)//batch_size}, Loss: {loss.item():.4f}"
            )
            j += 1

# Evaluate Hit Rate and NDCG
def evaluate_fine_tune(model, df_val, top_K, random_samples, num_items, groups):
    model.eval()
    results = {group: {"HR": [], "NDCG": []} for group in np.unique(groups)}

    for group in results.keys():
        group_indices = df_val["group"] == group
        group_data = df_val[group_indices].values

        for i in range(len(group_data)):
            test_user_input, test_item_input = get_test_instances_with_random_samples(
                group_data[i], random_samples, num_items, device
            )
            y_hat = model(test_user_input, test_item_input)
            y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
            test_item_input = test_item_input.cpu().detach().numpy().reshape((-1,))
            map_item_score = {test_item_input[j]: y_hat[j] for j in range(len(y_hat))}
            for k in range(top_K):
                ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
                gtItem = test_item_input[0]
                results[group]["HR"].append(getHitRatio(ranklist, gtItem))
                results[group]["NDCG"].append(getNDCG(ranklist, gtItem))

    # Aggregate results
    aggregated_results = {
        group: {
            "HR": np.mean(results[group]["HR"]),
            "NDCG": np.mean(results[group]["NDCG"]),
        }
        for group in results.keys()
    }
    return aggregated_results

# Load data
train_data = pd.read_csv("data/last_fm_pop/train.csv")
test_data = pd.read_csv("data/last_fm_pop/test.csv")

num_users = max(train_data["userId"].max(), test_data["userId"].max()) + 1
num_items = max(train_data["itemId"].max(), test_data["itemId"].max()) + 1

# Load debiased embeddings
debias_users_embed = np.loadtxt("results/debias_users_embed_mf_lastfm_pop.txt")

# Load pre-trained model and replace embeddings
# DF_NCF = neuralCollabFilter(num_users, num_items, 128, [128, 64, 32, 16], 1).to(device)
DF_NCF = matrixFactorization(num_users, num_items, 128).to(device)
DF_NCF.load_state_dict(torch.load("trained-models/preTrained_MF_lastfm_pop.pth", map_location=device))
DF_NCF.user_emb.weight.data = torch.from_numpy(debias_users_embed.astype(np.float32)).to(device)
DF_NCF.user_emb.weight.requires_grad = False

# Fine-tune
fair_fine_tune_model(
    DF_NCF,
    train_data,
    epochs=1,
    lr=0.001,
    batch_size=1024,
    num_negatives=1,
    num_items=num_items,
    protectedAttributes=train_data["group"].values,
    lamda=0.1,
    epsilonBase=torch.tensor(0.0).to(device),
    unsqueeze=True,
)

# Save fine-tuned model
torch.save(DF_NCF.state_dict(), "trained-models/DF_MF_lastfm_pop.pth")

# Evaluate
groupwise_results = evaluate_fine_tune(
    DF_NCF, test_data, top_K=35, random_samples=50, num_items=num_items, groups=test_data["group"]
)

# Print results
for group, metrics in groupwise_results.items():
    print(f"Group {group}: Hit Rate = {metrics['HR']:.4f}, NDCG = {metrics['NDCG']:.4f}")
