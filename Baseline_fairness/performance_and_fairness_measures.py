
import numpy as np
import torch
import math

# Performance Metrics
def getHitRatio(ranklist, gtItem):
    return 1 if gtItem in ranklist else 0

def getNDCG(ranklist, gtItem):
    for i, item in enumerate(ranklist):
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

# Differential Fairness Metric
def computeEDF(protectedAttributes, predictions, numClasses, item_input, device):
    S = np.unique(protectedAttributes)
    countsClassOne = torch.zeros((numClasses, len(S)), dtype=torch.float).to(device)
    countsTotal = torch.zeros((numClasses, len(S)), dtype=torch.float).to(device)
    dirichletAlpha = 1.0 / numClasses
    for i in range(len(predictions)):
        countsTotal[item_input[i], protectedAttributes[i]] += 1.0
        countsClassOne[item_input[i], protectedAttributes[i]] += predictions[i]
    probabilities = (countsClassOne + dirichletAlpha) / (countsTotal + 1.0)
    avg_epsilon = torch.mean(torch.max(torch.abs(torch.log(probabilities)), dim=1)[0])
    return avg_epsilon


def differentialFairnessMultiClass(probabilitiesOfPositive, numClasses, device):
    """
    Computes differential fairness (ε) for multi-class items.

    Parameters:
        probabilitiesOfPositive: tensor, smoothed probabilities of positive outcomes for each group
        numClasses: int, number of item classes
        device: torch device (CPU or GPU)

    Returns:
        avg_epsilon: float, average differential fairness measure
    """
    # Initialize epsilon per class
    epsilonPerClass = torch.zeros(len(probabilitiesOfPositive), dtype=torch.float).to(device)

    for c in range(len(probabilitiesOfPositive)):  # Iterate over item classes
        epsilon = torch.tensor(0.0).to(device)  # Initialize DF for the current class

        # Compare probability ratios for all group pairs
        for i in range(len(probabilitiesOfPositive[c])):
            for j in range(len(probabilitiesOfPositive[c])):
                if i == j:
                    continue
                else:
                    # Max log-difference between group probabilities
                    epsilon = torch.max(epsilon, torch.abs(
                        torch.log(probabilitiesOfPositive[c, i]) - torch.log(probabilitiesOfPositive[c, j])))
                    # Optional: Uncomment to use negative outcome probabilities
                    # epsilon = torch.max(epsilon, torch.abs((torch.log(1 - probabilitiesOfPositive[c, i])) - (torch.log(1 - probabilitiesOfPositive[c, j]))))

        epsilonPerClass[c] = epsilon  # Store DF for the current class

    avg_epsilon = torch.mean(epsilonPerClass)  # Average DF across classes
    return avg_epsilon


# def computeEDF(protectedAttributes, predictions, numClasses, item_input, device):
#     """
#     Computes empirical differential fairness (EDF) across groups in protectedAttributes.
#
#     Parameters:
#         protectedAttributes: array-like, group labels (e.g., 1, 2)
#         predictions: tensor, predicted probabilities for user-item pairs
#         numClasses: int, total number of items
#         item_input: tensor, item IDs corresponding to predictions
#         device: torch device (CPU or GPU)
#
#     Returns:
#         avg_epsilon: differential fairness measure
#     """
#     # Unique group values (e.g., [1, 2])
#     S = np.unique(protectedAttributes)
#     numGroups = len(S)
#
#     # Adjust for zero-based indexing
#     group_to_index = {val: i for i, val in enumerate(S)}
#     group_indices = np.array([group_to_index[g] for g in protectedAttributes])
#
#     # Initialize counts and probabilities
#     countsClassOne = torch.zeros((numClasses, numGroups), dtype=torch.float).to(device)
#     countsTotal = torch.zeros((numClasses, numGroups), dtype=torch.float).to(device)
#
#     # Dirichlet smoothing
#     concentrationParameter = 1.0
#     dirichletAlpha = concentrationParameter / numClasses
#
#     # Count occurrences and predictions
#     for i in range(len(predictions)):
#         group = group_indices[i]
#         item = item_input[i]
#         countsTotal[item, group] += 1.0
#         countsClassOne[item, group] += predictions[i]
#
#     # Compute smoothed probabilities
#     probabilitiesForDFSmoothed = (countsClassOne + dirichletAlpha) / (countsTotal + concentrationParameter)
#
#     # Compute differential fairness
#     avg_epsilon = differentialFairnessMultiClass(probabilitiesForDFSmoothed, numClasses, device)
#     return avg_epsilon

def computeEDF(protectedAttributes, predictions, numClasses, item_input, device):
    """
    Computes empirical differential fairness (EDF) across groups in protectedAttributes.

    Parameters:
        protectedAttributes: array-like, group labels (e.g., 1, 2)
        predictions: tensor, predicted probabilities for user-item pairs
        numClasses: int, total number of items
        item_input: tensor, item IDs corresponding to predictions
        device: torch device (CPU or GPU)

    Returns:
        avg_epsilon: differential fairness measure
    """
    # Unique group values (e.g., [1, 2])
    S = np.unique(protectedAttributes)
    numGroups = len(S)

    # Adjust for zero-based indexing
    group_to_index = {val: i for i, val in enumerate(S)}
    group_indices = np.array([group_to_index[g] for g in protectedAttributes])

    # Initialize counts and probabilities
    countsClassOne = torch.zeros((numClasses, numGroups), dtype=torch.float).to(device)
    countsTotal = torch.zeros((numClasses, numGroups), dtype=torch.float).to(device)

    # Dirichlet smoothing
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter / numClasses

    # Flatten predictions to ensure compatibility
    predictions = predictions.view(-1)

    # Count occurrences and predictions
    for i in range(len(predictions)):
        group = group_indices[i]
        item = item_input[i]
        countsTotal[item, group] += 1.0
        countsClassOne[item, group] += predictions[i].item()  # Use `.item()` to convert to Python scalar

    # Compute smoothed probabilities
    probabilitiesForDFSmoothed = (countsClassOne + dirichletAlpha) / (countsTotal + concentrationParameter)

    # Compute differential fairness
    avg_epsilon = differentialFairnessMultiClass(probabilitiesForDFSmoothed, numClasses, device)
    return avg_epsilon
