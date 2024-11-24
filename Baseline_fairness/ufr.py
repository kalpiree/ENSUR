import pandas as pd
import numpy as np
from mip import Model, xsum, maximize, BINARY

# Load your dataset
df = pd.read_csv('/content/tests_with_scorRes_amazonoffice_interactions_NeuMF.csv')  # Replace with your file path

# Extract unique users and items
user_ids = df['userId'].unique()
item_ids = df['itemId'].unique()
total_users = len(user_ids)
total_items = len(item_ids)

# Map userId and itemId to indices for optimization
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}
df['user_idx'] = df['userId'].map(user_map)
df['item_idx'] = df['itemId'].map(item_map)

# Scores matrix (precomputed scores)
scores = np.zeros((total_users, total_items))
for row in df.itertuples():
    scores[row.user_idx, row.item_idx] = row.score

# Grouping
group_1_users = df[df['group'] == 1]['user_idx'].unique()
group_2_users = df[df['group'] == 2]['user_idx'].unique()

# Ground Truth (labels = 1 are true samples)
ground_truth = {}
for row in df[df['label'] == 1].itertuples():
    if row.user_idx not in ground_truth:
        ground_truth[row.user_idx] = set()
    ground_truth[row.user_idx].add(row.item_idx)

# Fairness Optimization
topk = 28  # Number of items to recommend per user
model = Model()

# Variables: Binary matrix W (user x items)
W = [[model.add_var(var_type=BINARY) for j in range(total_items)] for i in range(total_users)]

# Objective: Maximize overall score
model.objective = maximize(xsum(scores[i][j] * W[i][j] for i in range(total_users) for j in range(total_items)))

# Constraint: Recommend top-k items for each user
for i in range(total_users):
    model += xsum(W[i][j] for j in range(total_items)) == topk

# Fairness Constraints: Balance NDCG between groups
group_1_dcg = xsum(scores[i][j] * W[i][j] for i in group_1_users for j in range(total_items))
group_2_dcg = xsum(scores[i][j] * W[i][j] for i in group_2_users for j in range(total_items))

# Fairness constraint (adjust epsilon for tighter or looser fairness)
epsilon = 0.01
model += group_1_dcg - group_2_dcg <= epsilon
model += group_2_dcg - group_1_dcg <= epsilon

# Solve the optimization problem
status = model.optimize()
print(f"Optimization status: {status}")

# Extract recommendations
recommendations = {}
for i in range(total_users):
    recommended_items = [j for j in range(total_items) if W[i][j].x >= 0.98]
    recommendations[user_ids[i]] = [item_ids[item] for item in recommended_items]

# Display recommendations
for user, items in recommendations.items():
    print(f"User {user}: Recommended items: {items}")

# Evaluation Metrics
def precision(actual, predicted):
    return len(set(actual) & set(predicted)) / len(predicted)

def recall(actual, predicted):
    return len(set(actual) & set(predicted)) / len(actual)

def ndcg(actual, predicted):
    dcg = sum([1 / np.log2(i + 2) if predicted[i] in actual else 0 for i in range(len(predicted))])
    idcg = sum([1 / np.log2(i + 2) for i in range(len(actual))])
    return dcg / idcg if idcg > 0 else 0

# Evaluate Recommendations
group_metrics = {1: [], 2: []}
for user_idx, predicted_items in enumerate(W):
    if user_idx in ground_truth:
        actual_items = ground_truth[user_idx]
        predicted = [j for j in range(total_items) if W[user_idx][j].x >= 0.98]

        # Compute metrics
        user_precision = precision(actual_items, predicted)
        user_recall = recall(actual_items, predicted)
        user_ndcg = ndcg(actual_items, predicted)

        # Store by group
        if user_idx in group_1_users:
            group_metrics[1].append((user_precision, user_recall, user_ndcg))
        elif user_idx in group_2_users:
            group_metrics[2].append((user_precision, user_recall, user_ndcg))

# Aggregate Results
for group in [1, 2]:
    precisions, recalls, ndcgs = zip(*group_metrics[group])
    print(f"Group {group} - Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, NDCG: {np.mean(ndcgs):.4f}")
