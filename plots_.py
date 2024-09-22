
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.lines import Line2D

# Define a fixed set of colors for the models
color_map = {
    'DeepFM': 'blue',
    'GMF': 'orange',
    'LightGCN': 'green',
    'MLP': 'red',
    'NeuMF': 'purple'
}

# Function to load the data from Excel files
def load_data(base_folder, eta, x):
    folder = os.path.join(base_folder, 'results_data/exp_1/varying_delta_fixed_eta_exp_1')
    file_name = f'all_evaluation_results_100_{x}_{eta}.xlsx'
    file_path = os.path.join(folder, file_name)
    return pd.read_excel(file_path)

# Function to filter the data based on Method = 'Conformal'
def filter_data(df):
    return df[df['Method'] == 'Conformal']

def plot_metric(df_all, eta, dataset, interaction, metric, models, x_values, output_folder):
    plt.figure(figsize=(12, 8))

    lines = []
    labels = []

    for model in models:
        if metric == 'Average Set Size':
            subset_group1 = df_all[(df_all['File'].str.contains(dataset)) &
                                   (df_all['File'].str.contains(interaction)) &
                                   (df_all['File'].str.contains(model)) &
                                   (df_all['Group'] == 1)]
            subset_group2 = df_all[(df_all['File'].str.contains(dataset)) &
                                   (df_all['File'].str.contains(interaction)) &
                                   (df_all['File'].str.contains(model)) &
                                   (df_all['Group'] == 2)]

            if not subset_group1.empty and not subset_group2.empty:
                avg_values = (subset_group1[metric].values + subset_group2[metric].values) / 2
                avg_values = avg_values.astype(int)

                x_transformed = subset_group1['x_value'].tolist()

                print(
                    f"\nPlotting data for model: {model}, dataset: {dataset}, interaction: {interaction}, eta: {eta}, metric: {metric}")
                print(pd.DataFrame({'x_value': x_transformed, metric: avg_values}))

                line, = plt.plot(x_transformed, avg_values, linestyle='-', marker='o',
                                 color=color_map[model])
                lines.append(line)
                labels.append(f'{model} +GUFR')

        else:
            subset = df_all[(df_all['File'].str.contains(dataset)) &
                            (df_all['File'].str.contains(interaction)) &
                            (df_all['File'].str.contains(model)) &
                            (df_all['Group'] == 1)]

            if subset.empty:
                print(f"No data for {model} in {dataset} {interaction} with eta = {eta}")
                continue

            x_transformed = subset['x_value'].tolist()

            print(
                f"\nPlotting data for model: {model}, dataset: {dataset}, interaction: {interaction}, eta: {eta}, metric: {metric}")
            print(subset[['x_value', metric]])

            line, = plt.plot(x_transformed, subset[metric], linestyle='-', marker='o',
                             color=color_map[model])
            lines.append(line)
            labels.append(f'{model} +GUFR')

    if not lines:
        print(f"No valid data to plot for {dataset} {interaction} with eta = {eta}")
        plt.close()
        return

    model_legend = plt.legend(lines, labels, loc='best', title="Models", fontsize='small')
    plt.gca().add_artist(model_legend)

    if metric in ['Hit Rate Diff', 'NDCG Diff']:
        plt.axhline(y=eta / 100, color='darkred', linestyle=':', linewidth=1)

    plt.xlabel(r'$\delta$')
    plt.ylabel(metric)

    eta_folder = os.path.join(output_folder, f'eta_{eta}')
    os.makedirs(eta_folder, exist_ok=True)

    file_name = f'{metric}_{dataset}_{interaction}_eta_{eta}.png'
    plt.savefig(os.path.join(eta_folder, file_name))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save plots based on dataset evaluations.")
    parser.add_argument('--base_folder', type=str, required=True, help="Path to the base folder containing dataset files.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder where plots will be saved.")

    args = parser.parse_args()

    eta_values = [10]
    x_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    datasets = ['lastfm', 'amazonoffice']
    interactions = ['interactions', 'popularconsumption']
    models = ['DeepFM', 'GMF', 'LightGCN', 'MLP', 'NeuMF']
    metrics = ['Average Set Size', 'Hit Rate Diff', 'NDCG Diff']

    os.makedirs(args.output_folder, exist_ok=True)

    for eta in eta_values:
        for dataset in datasets:
            for interaction in interactions:
                for metric in metrics:
                    df_all = pd.DataFrame()

                    for x in x_values:
                        df = load_data(args.base_folder, eta, x)
                        df_filtered = filter_data(df).copy()
                        df_filtered['x_value'] = (x / 100)
                        df_all = pd.concat([df_all, df_filtered])

                    print(
                        f"\nAccumulated Data for eta = {eta}, dataset = {dataset}, interaction = {interaction}, metric = {metric}:")
                    print(df_all[['File', 'Group', 'x_value', metric]])

                    plot_metric(df_all, eta, dataset, interaction, metric, models, x_values, args.output_folder)
