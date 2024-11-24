# -*- coding: utf-8 -*-
"""
Updated run_debiasing_userEmbeddings.py
"""
import numpy as np
import pandas as pd
import torch
from collaborative_models import neuralCollabFilter, matrixFactorization


# Set random seed for reproducibility
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

# Dynamically assign device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Compute group direction for debiasing
def compute_group_direction(data, group_column, user_vectors):
    num_groups = len(data[group_column].unique())
    group_embed = np.zeros((num_groups, user_vectors.shape[1]))
    num_users_per_group = np.zeros((num_groups, 1))

    for i, row in data.iterrows():
        user = row["userId"]
        group = row[group_column]
        group_embed[group] += user_vectors[user]
        num_users_per_group[group] += 1.0

    group_embed /= num_users_per_group  # Average group embedding
    return group_embed


# Compute bias direction
def compute_bias_direction(group_vectors):
    vBias = group_vectors[1].reshape((1, -1)) - group_vectors[0].reshape((1, -1))
    vBias = vBias / np.linalg.norm(vBias, axis=1, keepdims=True)
    return vBias


# Apply linear projection for debiasing
def linear_projection(data, user_vectors, vBias):
    for i, row in data.iterrows():
        user = row["userId"]
        user_vectors[user] = user_vectors[user] - np.inner(user_vectors[user].reshape(1, -1), vBias)[0][0] * vBias
    return user_vectors


# Load train and test datasets
train_data = pd.read_csv("data/last_fm_pop/train.csv")  # Replace with your train CSV path
test_data = pd.read_csv("data/last_fm_pop/test.csv")  # Replace with your test CSV path

# Map groups (1 → 0, 2 → 1)
train_data["group"] = train_data["group"] - 1
test_data["group"] = test_data["group"] - 1

# Dynamically calculate num_users and num_items based on both train and test data
num_users = max(train_data["userId"].max(), test_data["userId"].max()) + 1
num_items = max(train_data["itemId"].max(), test_data["itemId"].max()) + 1

# Set hyperparameters
emb_size = 128
hidden_layers = np.array([emb_size, 64, 32, 16])
output_size = 1

# Initialize the model with dynamically calculated values
# debiased_NCF = neuralCollabFilter(num_users, num_items, emb_size, hidden_layers, output_size).to(device)
debiased_NCF = matrixFactorization(num_users, num_items, emb_size).to(device)
# Load pre-trained model
pretrained_state_dict = torch.load("trained-models/preTrained_MF_lastfm_pop.pth", map_location=device)

# Rename keys in the pre-trained model to match the current model's key names
state_dict = {k.replace("like_emb", "item_emb"): v for k, v in pretrained_state_dict.items()}

# # Handle size mismatches in user embeddings
# if debiased_NCF.user_emb.weight.size() != state_dict["user_emb.weight"].size():
#     print("Resizing user embeddings to match the current dataset...")
#     state_dict["user_emb.weight"] = state_dict["user_emb.weight"][:num_users]
#
# # Handle size mismatches in item embeddings
# if debiased_NCF.item_emb.weight.size() != state_dict["item_emb.weight"].size():
#     print("Resizing item embeddings to match the current dataset...")
#     state_dict["item_emb.weight"] = state_dict["item_emb.weight"][:num_items]

# Load the adjusted state dictionary
debiased_NCF.load_state_dict(state_dict)
debiased_NCF.to(device)

# Extract user embeddings
users_embed = debiased_NCF.user_emb.weight.data.cpu().detach().numpy()
users_embed = users_embed.astype("float")
np.savetxt("results/users_embed_mf_lastfm_pop.txt", users_embed)

# Compute group direction and bias
group_vectors = compute_group_direction(train_data, "group", users_embed)
np.savetxt("results/group_vectors_mf_lastfm_pop.txt", group_vectors)

vBias = compute_bias_direction(group_vectors)
np.savetxt("results/vBias_mf_lastfm_pop.txt", vBias)

# Debias user embeddings (Train data only)
debias_users_embed = linear_projection(train_data, users_embed, vBias)
np.savetxt("results/debias_users_embed_mf_lastfm_pop.txt", debias_users_embed)
