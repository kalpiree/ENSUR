
import os
import pandas as pd
import numpy as np
import math
import argparse
from cal_methods_1 import apply_calibration_and_evaluate
from lambda_evaluator import DataSetEvaluator
from lambda_optimizer import LambdaOptimizer


class DataSetProcessor:
    def __init__(self, base_path, output_path):
        self.base_path = base_path
        self.output_path = output_path

    def process_datasets(self):
        final_results = []
        files = os.listdir(self.base_path)
        validation_files = [f for f in files if 'validation' in f]
        test_files = [f for f in files if 'test' in f]
        print("Validation Files:", validation_files)  # Debugging
        print("Test Files:", test_files)  # Debugging

        for validation_file in validation_files:
            base_name = validation_file.replace('validations_with_scorRes_', '')
            corresponding_test_file = 'tests_with_scorRes_' + base_name
            if corresponding_test_file in test_files:
                print(f"Processing {validation_file} and {corresponding_test_file}...")
                validation_df = pd.read_csv(os.path.join(self.base_path, validation_file))
                test_df = pd.read_csv(os.path.join(self.base_path, corresponding_test_file))

                # Evaluate with dynamic lambda values
                optimizer = LambdaOptimizer(validation_df)
                lambda_values = optimizer.adjust_lambda()
                evaluator = DataSetEvaluator(test_df, lambda_values, alpha=0.10, file_name=corresponding_test_file)
                results = evaluator.evaluate()

                print("Results from evaluator:", results)  # Debugging

                # Calculate the floor of the average of average set sizes from current results
                df_results = pd.DataFrame(results)
                print("df results", df_results)
                df_results = add_difference_columns(df_results)
                print("Results after adding differences:", df_results)  # Debugging

                df_results['Average Set Size'] = df_results['Average Set Size'].astype(int)

                # Calculate the average set sizes
                avg_set_sizes = df_results.groupby(['Group'])['Average Set Size'].mean()

                df_results['overall_mean'] = avg_set_sizes.mean().astype(int)

                final_results.append(df_results)

        # Concatenate all results from all files and save
        final_df = pd.concat(final_results, ignore_index=True)
        self.save_results(final_df)

    def save_results(self, results_df):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        with pd.ExcelWriter(f'{self.output_path}/all_evaluation_results_100_10_10.xlsx', engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)
        print("Results saved.")


def add_difference_columns(df):
    """Add columns for differences between Hit Rate and NDCG for two groups in each file."""
    if not df.empty and 'Hit Rate' in df.columns and 'NDCG' in df.columns:
        df.sort_values(by=['File', 'Group'], inplace=True)

        grouped = df.groupby(['File', 'Method'])
        differences = grouped.apply(lambda x: pd.Series({
            'Hit Rate Diff': abs(x['Hit Rate'].diff().iloc[-1]),
            'NDCG Diff': abs(x['NDCG'].diff().iloc[-1])
        }))

        differences = differences.reset_index()

        df = df.merge(differences, on=['File', 'Method'], how='left')
    else:
        print("Required columns are missing or the DataFrame is empty.")
        df['Hit Rate Diff'] = np.nan
        df['NDCG Diff'] = np.nan

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets and save results.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the input folder containing dataset files.")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Path to the folder where results will be saved.")

    args = parser.parse_args()

    processor = DataSetProcessor(args.input_folder, args.output_folder)
    processor.process_datasets()
