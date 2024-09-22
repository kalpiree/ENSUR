import pandas as pd
import numpy as np

import numpy as np
from scipy.stats import binom


class LambdaOptimizer:
    def __init__(self, df, initial_lambda=1.0, alpha=0.10, epsilon=0.05, stability_threshold=0.02,
                 max_stable_iterations=10, min_lambda=0, delta=0.10,
                 hit_rate_diff_threshold=0.2, ndcg_diff_threshold=0.20):
        self.df = df
        self.initial_lambda = initial_lambda
        self.alpha = alpha
        self.epsilon = epsilon
        self.stability_threshold = stability_threshold
        self.max_stable_iterations = max_stable_iterations
        self.min_lambda = min_lambda
        self.delta = delta
        self.hit_rate_diff_threshold = hit_rate_diff_threshold
        self.ndcg_diff_threshold = ndcg_diff_threshold
        self.lambda_values = {group: initial_lambda for group in df['group'].unique()}

    def calculate_metrics(self):
        results = {}
        item_sets = {group: {} for group in self.df['group'].unique()}

        for group_label in self.df['group'].unique():
            group_df = self.df[self.df['group'] == group_label].copy()
            group_df.sort_values(by='score', ascending=False, inplace=True)
            lambda_value = self.lambda_values[group_label]
            group_df['included'] = group_df['score'] >= lambda_value
            included_items = group_df[group_df['included']].groupby('userId')['itemId'].apply(list)
            item_sets[group_label] = included_items.to_dict()

            def user_loss(user_items, user_id):
                labels = group_df[(group_df['itemId'].isin(user_items)) & (group_df['userId'] == user_id)]['label']
                return 0 if any(labels == 1) else 1

            def hit(gt_item, pred_items):
                return 1 if gt_item in pred_items else 0

            def ndcg(gt_item, pred_items):
                if gt_item in pred_items:
                    index = pred_items.index(gt_item)
                    return 1 / np.log2(index + 2)
                return 0

            user_losses = included_items.index.to_series().apply(
                lambda user_id: user_loss(included_items[user_id], user_id))
            average_loss = user_losses.mean() if not user_losses.empty else 1
            loss_variance = user_losses.var() if not user_losses.empty else 0
            inclusion_rate = group_df[group_df['label'] == 1]['included'].mean() if not group_df[
                group_df['label'] == 1].empty else 0

            hits = []
            ndcgs = []
            for user_id, user_item_set in included_items.items():
                user_data = group_df[group_df['userId'] == user_id]
                ground_truth_item = user_data[user_data['label'] == 1]['itemId'].values
                if len(ground_truth_item) > 0:
                    ground_truth_item = ground_truth_item[0]
                    hits.append(hit(ground_truth_item, user_item_set))
                    ndcgs.append(ndcg(ground_truth_item, user_item_set))

            hit_rate = np.mean(hits) if hits else 0
            avg_ndcg = np.mean(ndcgs) if ndcgs else 0

            ndcg_variance = np.var(ndcgs) if ndcgs else 0  # Variance of NDCG scores

            results[group_label] = {
                'Lambda': lambda_value,
                'Loss': average_loss,
                'Loss Variance': loss_variance,
                'Inclusion Rate': inclusion_rate,
                'Included Items': included_items,
                'Hit Rate': hit_rate,
                'NDCG': avg_ndcg,
                'NDCG Variance': ndcg_variance
            }
        return results, item_sets

    def adjust_lambda(self):
        stability_count = 0
        previous_hit_rate_diff = float('inf')
        previous_ndcg_diff = float('inf')
        large_adjustment_step = 0.03
        small_adjustment_step = 0.02  # Smaller step for fine adjustments

        # Initial adjustment using binomial UCB
        while True:
            metrics, item_sets = self.calculate_metrics()
            all_within_bound = True

            for group in self.lambda_values.keys():
                n = self.df[self.df['group'] == group]['userId'].nunique()
                if n > 0:
                    empirical_loss = metrics[group]['Loss']

                    # New UCB using the binomial CDF
                    ucb = binom.cdf(empirical_loss * n, n, self.alpha) - self.delta
                    # print("UCB for group",group,ucb)
                    if ucb > 0:
                        old_lambda = self.lambda_values[group]
                        new_lambda = max(self.min_lambda, self.lambda_values[group] - large_adjustment_step)
                        if new_lambda > self.min_lambda:
                            self.lambda_values[group] = new_lambda
                            print(
                                f"After adjusting for alpha with binomial UCB, lambda for group {group} changed from {old_lambda} to {new_lambda}")
                        else:
                            self.lambda_values[group] = self.min_lambda
                            print(f"Lambda for group {group} reached minimum value {self.min_lambda}")
                        all_within_bound = False
                else:
                    print(f"No users in group {group}, unable to calculate UCB.")

            if all_within_bound:
                break

        # Fine-tuning before stability loop
        for group in self.lambda_values.keys():
            n = self.df[self.df['group'] == group]['userId'].nunique()
            if n > 0:
                while True:
                    empirical_loss = metrics[group]['Loss']
                    ucb = binom.cdf(empirical_loss * n, n, self.alpha) - self.delta

                    if ucb >= 0:
                        break

                    old_lambda = self.lambda_values[group]
                    self.lambda_values[group] += 0.01  # Increment lambda
                    metrics, item_sets = self.calculate_metrics()
                    print(
                        f"Fine-tuning: lambda for group {group} changed from {old_lambda} to {self.lambda_values[group]}")

        # Stability loop with UCB for hit rate and NDCG differences
        while stability_count < self.max_stable_iterations:
            metrics, item_sets = self.calculate_metrics()
            active_groups = list(self.lambda_values.keys())
            if len(active_groups) < 2:
                break

            hit_rate_diff = abs(metrics[active_groups[0]]['Hit Rate'] - metrics[active_groups[1]]['Hit Rate'])
            ndcg_diff = abs(metrics[active_groups[0]]['NDCG'] - metrics[active_groups[1]]['NDCG'])
            print(
                f"Adjusting for Fairness - Current Hit Rate Difference: {hit_rate_diff}, NDCG Difference: {ndcg_diff}")

            # Calculate UCB for hit rate difference
            n1 = self.df[self.df['group'] == active_groups[0]]['userId'].nunique()
            n2 = self.df[self.df['group'] == active_groups[1]]['userId'].nunique()
            p1_hit = metrics[active_groups[0]]['Hit Rate']
            p2_hit = metrics[active_groups[1]]['Hit Rate']
            variance_hit = (p1_hit * (1 - p1_hit) / n1) + (p2_hit * (1 - p2_hit) / n2)
            ucb_hit = hit_rate_diff + np.sqrt(
                2 * variance_hit * np.log(2 / self.delta) + (2 / 3) * np.log(2 / self.delta) / (n1 + n2))

            variance_ndcg1 = metrics[active_groups[0]]['NDCG Variance']
            variance_ndcg2 = metrics[active_groups[1]]['NDCG Variance']
            # variance_ndcg = (s1_ndcg / n1) + (s2_ndcg / n2)
            variance_ndcg = (variance_ndcg1 / n1) + (variance_ndcg2 / n2)
            ucb_ndcg = ndcg_diff + np.sqrt(
                2 * variance_ndcg * np.log(2 / self.delta) + (2 / 3) * np.log(2 / self.delta) / (n1 + n2))

            if ucb_hit <= self.hit_rate_diff_threshold and ucb_ndcg <= self.ndcg_diff_threshold:
                break

            if abs(ucb_hit - previous_hit_rate_diff) <= self.stability_threshold and abs(
                    ucb_ndcg - previous_ndcg_diff) <= self.stability_threshold:
                stability_count += 1
            else:
                stability_count = 0

            adjustment_step = large_adjustment_step if ucb_hit > self.hit_rate_diff_threshold and ucb_ndcg > self.ndcg_diff_threshold else small_adjustment_step
            if ucb_hit > self.hit_rate_diff_threshold:
                group_to_adjust = active_groups[0] if metrics[active_groups[0]]['Hit Rate'] < metrics[active_groups[1]][
                    'Hit Rate'] else active_groups[1]
            else:
                group_to_adjust = active_groups[0] if metrics[active_groups[0]]['NDCG'] < metrics[active_groups[1]][
                    'NDCG'] else active_groups[1]

            old_lambda = self.lambda_values[group_to_adjust]
            self.lambda_values[group_to_adjust] = max(self.min_lambda,
                                                      self.lambda_values[group_to_adjust] - adjustment_step)
            print(
                f"Lambda for group {group_to_adjust} adjusted from {old_lambda} to {self.lambda_values[group_to_adjust]}")
            print(
                f"Group {group_to_adjust}: Hit Rate: {metrics[group_to_adjust]['Hit Rate']}, NDCG: {metrics[group_to_adjust]['NDCG']}")

            metrics, item_sets = self.calculate_metrics()
            for group in active_groups:
                print(f"Group {group}: Hit Rate: {metrics[group]['Hit Rate']}, NDCG: {metrics[group]['NDCG']}")

            previous_hit_rate_diff = ucb_hit
            previous_ndcg_diff = ucb_ndcg

        return self.lambda_values

# class LambdaOptimizer:
#     def __init__(self, df, initial_lambda=1.0, alpha=0.5, epsilon=0.05, stability_threshold=0.02,
#                  max_stable_iterations=10, min_lambda=0, delta=0.1):   # change here 2,3
#         self.df = df
#         self.initial_lambda = initial_lambda
#         self.alpha = alpha
#         self.epsilon = epsilon
#         self.stability_threshold = stability_threshold
#         self.max_stable_iterations = max_stable_iterations
#         self.min_lambda = min_lambda
#         self.delta = delta
#         self.lambda_values = {group: initial_lambda for group in df['group'].unique()}
#
#     # def calculate_metrics(self):
#     #     results = {}
#     #     item_sets = {group: [] for group in self.df['group'].unique()}
#     #     for group_label in self.df['group'].unique():
#     #         group_df = self.df[self.df['group'] == group_label].copy()
#     #         group_df.sort_values(by='score', ascending=False, inplace=True)
#     #         lambda_value = self.lambda_values[group_label]
#     #         group_df['included'] = group_df['score'] > lambda_value
#     #         included_items = group_df[group_df['included']].groupby('userId')['itemId'].apply(list)
#     #         item_sets[group_label] = included_items.tolist()
#     #
#     #         def user_loss(user_items):
#     #             labels = group_df[group_df['itemId'].isin(user_items)]['label']
#     #             return 0 if any(labels == 1) else 1
#     #
#     #         user_losses = included_items.apply(user_loss)
#     #         average_loss = user_losses.mean() if not user_losses.empty else 1
#     #         loss_variance = user_losses.var() if not user_losses.empty else 0
#     #         inclusion_rate = group_df[group_df['label'] == 1]['included'].mean() if not group_df[
#     #             group_df['label'] == 1].empty else 0
#     #
#     #         results[group_label] = {
#     #             'Lambda': lambda_value,
#     #             'Loss': average_loss,
#     #             'Loss Variance': loss_variance,
#     #             'Inclusion Rate': inclusion_rate,
#     #             'Included Items': included_items
#     #         }
#     #     return results, item_sets
#
#     # def calculate_metrics(self):
#     #     results = {}
#     #     item_sets = {group: {} for group in self.df['group'].unique()}  # Initialize with groups
#     #
#     #     for group_label in self.df['group'].unique():
#     #         group_df = self.df[self.df['group'] == group_label].copy()
#     #         group_df.sort_values(by='score', ascending=False, inplace=True)
#     #         lambda_value = self.lambda_values[group_label]
#     #         group_df['included'] = group_df['score'] > lambda_value
#     #         included_items = group_df[group_df['included']].groupby('userId')['itemId'].apply(list)
#     #
#     #         # Ensure item_sets maps user IDs to item sets
#     #         item_sets[group_label] = included_items.to_dict()
#     #
#     #         def user_loss(user_items):
#     #             labels = group_df[group_df['itemId'].isin(user_items)]['label']
#     #             return 0 if any(labels == 1) else 1
#     #
#     #         user_losses = included_items.apply(user_loss)
#     #         average_loss = user_losses.mean() if not user_losses.empty else 1
#     #         loss_variance = user_losses.var() if not user_losses.empty else 0
#     #         inclusion_rate = group_df[group_df['label'] == 1]['included'].mean() if not group_df[
#     #             group_df['label'] == 1].empty else 0
#     #
#     #         results[group_label] = {
#     #             'Lambda': lambda_value,
#     #             'Loss': average_loss,
#     #             'Loss Variance': loss_variance,
#     #             'Inclusion Rate': inclusion_rate,
#     #             'Included Items': included_items
#     #         }
#     #     return results, item_sets
#
#     # def adjust_lambda(self):
#     #     stability_count = 0
#     #     previous_loss_diff = float('inf')
#     #
#     #     while True:
#     #         metrics, item_sets = self.calculate_metrics()
#     #         all_within_bound = True
#     #         for group in self.lambda_values.keys():
#     #             n = self.df[self.df['group'] == group]['userId'].nunique()
#     #             if n > 0:
#     #                 empirical_loss = metrics[group]['Loss']
#     #                 loss_variance = metrics[group]['Loss Variance']
#     #                 ucb = empirical_loss + np.sqrt(2 * loss_variance * np.log(3 / self.delta) / n) + 3 * np.log(
#     #                     3 / self.delta) / n
#     #
#     #                 if ucb > self.alpha:
#     #                     all_within_bound = False
#     #                     old_lambda = self.lambda_values[group]
#     #                     self.lambda_values[group] = max(self.min_lambda, self.lambda_values[group] - 0.05)
#     #                     print(
#     #                         f"After adjusting for alpha with UCB, lambda for group {group} changed from {old_lambda} to {self.lambda_values[group]}")
#     #             else:
#     #                 print(f"No users in group {group}, unable to calculate UCB.")
#     #
#     #         if all_within_bound:
#     #             break
#     #
#     #     # Adjusting for fairness between groups, considering stability
#     #     while stability_count < self.max_stable_iterations:
#     #         loss_values = [metrics[group]['Loss'] for group in self.lambda_values.keys()]
#     #         loss_diff = abs(loss_values[0] - loss_values[1])
#     #         print(f"Adjusting for Fairness - Current Loss Difference: {loss_diff}")
#     #
#     #         if loss_diff <= self.epsilon:
#     #             break
#     #
#     #         if abs(loss_diff - previous_loss_diff) <= self.stability_threshold:
#     #             stability_count += 1
#     #         else:
#     #             stability_count = 0
#     #
#     #         group_to_adjust = self.df['group'].unique()[0] if loss_values[0] > loss_values[1] else \
#     #             self.df['group'].unique()[1]
#     #         old_lambda = self.lambda_values[group_to_adjust]
#     #         self.lambda_values[group_to_adjust] = max(self.min_lambda, self.lambda_values[group_to_adjust] - 0.03)
#     #         print(
#     #             f"Lambda for group {group_to_adjust} adjusted from {old_lambda} to {self.lambda_values[group_to_adjust]}")
#     #
#     #         metrics, item_sets = self.calculate_metrics()  # Recalculate with updated lambdas
#     #         previous_loss_diff = loss_diff
#     #
#     #     return self.lambda_values
#     # def calculate_metrics(self):
#     #     results = {}
#     #     item_sets = {group: {} for group in self.df['group'].unique()}  # Initialize with groups
#     #
#     #     for group_label in self.df['group'].unique():
#     #         group_df = self.df[self.df['group'] == group_label].copy()
#     #         group_df.sort_values(by='score', ascending=False, inplace=True)
#     #         lambda_value = self.lambda_values[group_label]
#     #         group_df['included'] = group_df['score'] > lambda_value
#     #         included_items = group_df[group_df['included']].groupby('userId')['itemId'].apply(list)
#     #
#     #         # Ensure item_sets maps user IDs to item sets
#     #         item_sets[group_label] = included_items.to_dict()
#     #
#     #         def user_loss(user_items):
#     #             labels = group_df[group_df['itemId'].isin(user_items)]['label']
#     #             return 0 if any(labels == 1) else 1
#     #
#     #         def hit(gt_item, pred_items):
#     #             return 1 if gt_item in pred_items else 0
#     #
#     #         def ndcg(gt_item, pred_items):
#     #             if gt_item in pred_items:
#     #                 index = pred_items.index(gt_item)
#     #                 return 1 / np.log2(index + 2)
#     #             return 0
#     #
#     #         user_losses = included_items.apply(user_loss)
#     #         average_loss = user_losses.mean() if not user_losses.empty else 1
#     #         loss_variance = user_losses.var() if not user_losses.empty else 0
#     #         inclusion_rate = group_df[group_df['label'] == 1]['included'].mean() if not group_df[
#     #             group_df['label'] == 1].empty else 0
#     #
#     #         # Calculate hit rate and NDCG for the group
#     #         hits = []
#     #         ndcgs = []
#     #         for user_id, user_item_set in included_items.items():
#     #             user_data = group_df[group_df['userId'] == user_id]
#     #             ground_truth_item = user_data[user_data['label'] == 1]['itemId'].values
#     #             if len(ground_truth_item) > 0:
#     #                 ground_truth_item = ground_truth_item[0]
#     #                 hit_result = hit(ground_truth_item, user_item_set)
#     #                 hits.append(hit(ground_truth_item, user_item_set))
#     #                 ndcgs.append(ndcg(ground_truth_item, user_item_set))
#     #                 # Debugging print statements
#     #                 print(
#     #                     f"User: {user_id}, Predicted Set: {user_item_set}, True Label: {ground_truth_item},"
#     #                     f" Hit: {hit_result}")
#     #
#     #         hit_rate = np.mean(hits) if hits else 0
#     #         avg_ndcg = np.mean(ndcgs) if ndcgs else 0
#     #
#     #         results[group_label] = {
#     #             'Lambda': lambda_value,
#     #             'Loss': average_loss,
#     #             'Loss Variance': loss_variance,
#     #             'Inclusion Rate': inclusion_rate,
#     #             'Included Items': included_items,
#     #             'Hit Rate': hit_rate,  # Added hit rate
#     #             'NDCG': avg_ndcg  # Added NDCG
#     #         }
#     #     return results, item_sets
#     def calculate_metrics(self):
#         results = {}
#         item_sets = {group: {} for group in self.df['group'].unique()}  # Initialize with groups
#
#         for group_label in self.df['group'].unique():
#             group_df = self.df[self.df['group'] == group_label].copy()
#             group_df.sort_values(by='score', ascending=False, inplace=True)
#             lambda_value = self.lambda_values[group_label]
#             group_df['included'] = group_df['score'] > lambda_value
#             included_items = group_df[group_df['included']].groupby('userId')['itemId'].apply(list)
#
#             # Ensure item_sets maps user IDs to item sets
#             item_sets[group_label] = included_items.to_dict()
#
#             def user_loss(user_items):
#                 labels = group_df[group_df['itemId'].isin(user_items)]['label']
#                 loss_result = 0 if any(labels == 1) else 1
#                 return loss_result
#
#
#
#             def hit(gt_item, pred_items):
#                 return 1 if gt_item in pred_items else 0
#
#             def ndcg(gt_item, pred_items):
#                 if gt_item in pred_items:
#                     index = pred_items.index(gt_item)
#                     return 1 / np.log2(index + 2)
#                 return 0
#
#             user_losses = included_items.apply(user_loss)
#             average_loss = user_losses.mean() if not user_losses.empty else 1
#             loss_variance = user_losses.var() if not user_losses.empty else 0
#             inclusion_rate = group_df[group_df['label'] == 1]['included'].mean() if not group_df[
#                 group_df['label'] == 1].empty else 0
#
#             # Calculate hit rate and NDCG for the group
#             hits = []
#             ndcgs = []
#             total_users = len(included_items)
#             num_hits = 0
#             num_losses = 0
#
#             for user_id, user_item_set in included_items.items():
#                 user_data = group_df[group_df['userId'] == user_id]
#                 ground_truth_item = user_data[user_data['label'] == 1]['itemId'].values
#                 if len(ground_truth_item) > 0:
#                     ground_truth_item = ground_truth_item[0]
#                     hit_result = hit(ground_truth_item, user_item_set)
#                     hits.append(hit_result)
#                     ndcgs.append(ndcg(ground_truth_item, user_item_set))
#                     num_hits += hit_result
#                     num_losses += 1 - hit_result
#
#             hit_rate = np.mean(hits) if hits else 0
#             avg_ndcg = np.mean(ndcgs) if ndcgs else 0
#
#             # Debugging print statements
#             # print(f"Group: {group_label}, Total Users: {total_users}, Hits: {num_hits}, Losses: {num_losses}")
#             # print(
#             #     f"Group: {group_label}, Lambda: {lambda_value}, Loss: {average_loss}, Hit Rate: {hit_rate}, NDCG: {avg_ndcg}")
#
#             results[group_label] = {
#                 'Lambda': lambda_value,
#                 'Loss': average_loss,
#                 'Loss Variance': loss_variance,
#                 'Inclusion Rate': inclusion_rate,
#                 'Included Items': included_items,
#                 'Hit Rate': hit_rate,  # Added hit rate
#                 'NDCG': avg_ndcg  # Added NDCG
#             }
#         return results, item_sets
#
#     def adjust_lambda(self):
#         stability_count = 0
#         previous_loss_diff = float('inf')
#         lambda_adjustment_stopped = {group: False for group in
#                                      self.lambda_values.keys()}  # Track if lambda adjustment is stopped for each group
#
#         while True:
#             metrics, item_sets = self.calculate_metrics()
#             all_within_bound = True
#             for group in self.lambda_values.keys():
#                 if not lambda_adjustment_stopped[group]:  # Only adjust if not stopped
#                     n = self.df[self.df['group'] == group]['userId'].nunique()
#                     if n > 0:
#                         empirical_loss = metrics[group]['Loss']
#                         loss_variance = metrics[group]['Loss Variance']
#                         ucb = empirical_loss + np.sqrt(2 * loss_variance * np.log(3 / self.delta) / n) + 3 * np.log(
#                             3 / self.delta) / n
#
#                         if ucb > self.alpha:
#                             old_lambda = self.lambda_values[group]
#                             new_lambda = max(self.min_lambda, self.lambda_values[group] - 0.05)
#                             if new_lambda > self.min_lambda:
#                                 self.lambda_values[group] = new_lambda
#                                 print(
#                                     f"After adjusting for alpha with UCB, lambda for group {group} changed from {old_lambda} to {new_lambda}")
#                             else:
#                                 self.lambda_values[group] = self.min_lambda
#                                 lambda_adjustment_stopped[group] = True
#                                 print(f"Lambda for group {group} reached minimum value {self.min_lambda}")
#                             all_within_bound = False
#                     else:
#                         print(f"No users in group {group}, unable to calculate UCB.")
#                 else:
#                     print(f"Adjustment for group {group} has been stopped at lambda {self.lambda_values[group]}")
#
#             if all_within_bound:
#                 break
#
#         # Adjusting for fairness between groups, considering stability
#         while stability_count < self.max_stable_iterations:
#             loss_values = [metrics[group]['Loss'] for group in self.lambda_values.keys() if
#                            not lambda_adjustment_stopped[group]]
#             if len(loss_values) < 2:
#                 break  # Break if not enough groups are active for fairness adjustment
#             loss_diff = abs(loss_values[0] - loss_values[1])
#             print(f"Adjusting for Fairness - Current Loss Difference: {loss_diff}")
#
#             if loss_diff <= self.epsilon:
#                 break
#
#             if abs(loss_diff - previous_loss_diff) <= self.stability_threshold:
#                 stability_count += 1
#             else:
#                 stability_count = 0
#
#             # Identify which group to adjust
#             group_to_adjust = self.df['group'].unique()[0] if loss_values[0] > loss_values[1] else \
#             self.df['group'].unique()[1]
#             if not lambda_adjustment_stopped[group_to_adjust]:
#                 old_lambda = self.lambda_values[group_to_adjust]
#                 self.lambda_values[group_to_adjust] = max(self.min_lambda, self.lambda_values[group_to_adjust] - 0.03)
#                 print(
#                     f"Lambda for group {group_to_adjust} adjusted from {old_lambda} to {self.lambda_values[group_to_adjust]}")
#
#             metrics, item_sets = self.calculate_metrics()  # Recalculate with updated lambdas
#             previous_loss_diff = loss_diff
#
#         return self.lambda_values
