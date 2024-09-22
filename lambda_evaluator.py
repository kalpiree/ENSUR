import numpy as np
import pandas as pd
from lambda_optimizer import LambdaOptimizer


class DataSetEvaluator:
    def __init__(self, test_df, lambda_values, alpha, file_name):
        self.test_df = test_df
        self.lambda_values = lambda_values
        self.alpha = alpha
        self.file_name = file_name

    def evaluate(self):
        optimizer = LambdaOptimizer(self.test_df, initial_lambda=1.0, alpha=self.alpha)
        optimizer.lambda_values = self.lambda_values
        print("Using lambda values for evaluation:", self.lambda_values)
        results, item_sets = optimizer.calculate_metrics()
        output = []

        print(f"Evaluating file: {self.file_name}")
        for group, metrics in results.items():
            avg_set_size = np.mean([len(items) for items in item_sets[group].values()]) if item_sets[group] else 0

            print(
                f"Group {group}: Lambda {metrics['Lambda']}, Loss {metrics['Loss']}, Average Set Size: {avg_set_size}, Hit Rate: {metrics['Hit Rate']}")

            if metrics['Loss'] <= self.alpha:
                print(f" - Loss is within the threshold alpha {self.alpha}.")
            else:
                print(f" - Loss exceeds the threshold alpha {self.alpha}.")

            output.append({
                'File': self.file_name,
                'Group': group,
                'Lambda': metrics['Lambda'],
                'Method': 'Conformal',
                'Loss': metrics['Loss'],
                'Average Set Size': avg_set_size,
                'Hit Rate': metrics['Hit Rate'],
                'NDCG': metrics['NDCG']  # // remember to change this later
                # 'Status': 'Within Alpha' if metrics['Loss'] <= self.alpha else 'Exceeds Alpha'
            })

        return output
