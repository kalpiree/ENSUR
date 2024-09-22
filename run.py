import gc
import os
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
import argparse
from evaluation import metrics
from models.BERT4Rec import BERT4Rec
from models.DeepFM import DeepFM
from models.FM import FactorizationMachine
from models.GMF import GMF
from models.LightGCN import LightGCN
from models.MLP import MLP
from models.NeuMF import NeuMF
from models.SASRec import SASRec
from models.WMF import WMF
from train import Train
from utils import FlexibleDataLoader, MovieLens


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train recommender models on different datasets.")

    parser.add_argument('--datasets', nargs='+', default=['amazonoffice'],
                        help="List of datasets to use (default: ['amazonoffice', 'lastfm'])")

    parser.add_argument('--interactions', nargs='+', default=['interactions'],
                        help="List of interaction types (['interactions', 'popularconsumption])")

    parser.add_argument('--models', nargs='+', default=['MLP'],
                        help="List of models to train (['MLP', 'GMF', 'NeuMF', 'WMF', 'FM', 'DeepFM', 'LightGCN'])")

    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")

    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training (default: 256)")

    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")

    parser.add_argument('--factor', type=int, default=8, help="Number of factors for the models (default: 8)")

    #parser.add_argument('--fraction', type=float, default=0.25, help="Fraction of dataset to sample (default: 0.25)")

    parser.add_argument('--output_folder', type=str, default='score_files',
                        help="Output folder to save results (default: 'score_files')")

    return parser.parse_args()


def append_scores_and_data(model, data_loader, device):
    model.eval()
    data_collection = []

    with torch.no_grad():
        for batch in data_loader:
            batch = [x.to(device).squeeze() for x in batch]

            if data_loader.dataset.include_features:
                users, items, labels, weights, features, ratings, item_types, groups = batch
                features = features.float()
                predictions = model(features).squeeze()
            else:
                users, items, labels, weights, ratings, item_types, groups = batch
                users, items = users.long(), items.long()
                predictions = model(users, items)

            batch_data = {
                'userId': users.cpu().numpy(),
                'itemId': items.cpu().numpy(),
                'score': predictions.cpu().numpy(),
                'label': labels.cpu().numpy(),
                'weight': weights.cpu().numpy(),
                'rating': ratings.cpu().numpy(),
                'item_type': item_types.cpu().numpy(),
                'group': groups.cpu().numpy()
            }
            data_collection.append(pd.DataFrame(batch_data))

    return pd.concat(data_collection, ignore_index=True)


def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    for dataset in args.datasets:
        if dataset in ['amazonoffice', 'movielens']:
            dataset_type = 'explicit'
        else:
            dataset_type = 'implicit'

        # Debugging line to check dataset_type
        print(f"Using dataset: {dataset}, type: {dataset_type}")
        for interaction in args.interactions:
            #file_path = f'/Users/nitinbisht/PycharmProjects/recom/new_score_files/paper_1/{dataset.lower()}_{interaction}_data.csv'
            file_path = f'path/to/datasets/{dataset.lower()}_{interaction}_data.csv'
            df = pd.read_csv(file_path)
            dataset_type = 'explicit' if dataset in ['amazonoffice', 'movielens'] else 'implicit'
            for model_name in args.models:
                print(f"Processing model: {model_name}")
                data_loader = FlexibleDataLoader(df=df, dataset_type=dataset_type)
                processed_data = data_loader.read_data()

                train_df, validation_df, test_df, total_df = data_loader.split_train_test()

                train_dataset = MovieLens(train_df, total_df, ng_ratio=1,
                                          include_features=(model_name in ['FM', 'DeepFM']))
                validation_dataset = MovieLens(validation_df, total_df, ng_ratio=50,
                                               include_features=(model_name in ['FM', 'DeepFM']))
                test_dataset = MovieLens(test_df, total_df, ng_ratio=50,
                                         include_features=(model_name in ['FM', 'DeepFM']))

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                num_users = train_dataset.get_num_users()
                num_items = train_dataset.get_num_items()
                num_features = train_dataset.get_num_features()

                models = {
                    'MLP': MLP(num_users=num_users, num_items=num_items, num_factor=args.factor),
                    'GMF': GMF(num_users=num_users, num_items=num_items, num_factor=args.factor),
                    'NeuMF': NeuMF(num_users=num_users, num_items=num_items, num_factor=args.factor),
                    'FM': FactorizationMachine(num_factors=args.factor, num_features=num_features),
                    'DeepFM': DeepFM(num_factors=args.factor, num_features=num_features),
                    'LightGCN': LightGCN(num_users=num_users, num_items=num_items, embedding_size=args.factor,
                                         n_layers=3),
                    'SASRec': SASRec(num_items=num_items, embedding_size=args.factor, num_heads=4, num_layers=2,
                                     dropout=0.1),
                    'BERT4Rec': BERT4Rec(num_items=num_items, embedding_size=args.factor, num_heads=4, num_layers=2)
                }

                model = models[model_name].to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = torch.nn.BCELoss()

                trainer = Train(
                    model=model,
                    optimizer=optimizer,
                    epochs=args.epochs,
                    dataloader=train_dataloader,
                    criterion=criterion,
                    test_obj=test_dataloader,
                    device=device,
                    print_cost=True,
                    use_features=model_name in ['FM', 'DeepFM'],
                )
                trainer.train()

                os.makedirs(args.output_folder, exist_ok=True)

                validation_df = append_scores_and_data(model, validation_dataloader, device)
                test_df = append_scores_and_data(model, test_dataloader, device)

                validation_df.to_csv(
                    f'{args.output_folder}/validations_with_scores_{dataset}_{interaction}_{model_name}.csv',
                    index=False)
                test_df.to_csv(f'{args.output_folder}/tests_with_scores_{dataset}_{interaction}_{model_name}.csv',
                               index=False)

                top_k = 10
                avg_hr_test, avg_ndcg_test = metrics(model, test_dataloader, top_k, device)
                avg_hr_val, avg_ndcg_val = metrics(model, validation_dataloader, top_k, device)

                print(f"Dataset: {dataset}, Interaction: {interaction}, Model: {model_name}")
                print(f"Average Hit Rate Test Set (HR@{top_k}): {avg_hr_test:.3f}")
                print(f"Average Normalized Discounted Cumulative Gain Test set (NDCG@{top_k}): {avg_ndcg_test:.3f}")
                print(f"Average Hit Rate Validation Set (HR@{top_k}): {avg_hr_val:.3f}")
                print(
                    f"Average Normalized Discounted Cumulative Gain Validation Set (NDCG@{top_k}): {avg_ndcg_val:.3f}")

                with open(f'{args.output_folder}/metrics_{dataset}_{interaction}_{model_name}.txt', 'w') as f:
                    f.write(f"Dataset: {dataset}, Interaction: {interaction}, Model: {model_name}\n")
                    f.write(f"Average Hit Rate Test Set (HR@{top_k}): {avg_hr_test:.3f}\n")
                    f.write(
                        f"Average Normalized Discounted Cumulative Gain Test set (NDCG@{top_k}): {avg_ndcg_test:.3f}\n")
                    f.write(f"Average Hit Rate Validation Set (HR@{top_k}): {avg_hr_val:.3f}\n")
                    f.write(
                        f"Average Normalized Discounted Cumulative Gain Validation Set (NDCG@{top_k}): {avg_ndcg_val:.3f}\n")

                del train_dataset, train_dataloader, validation_dataloader, validation_dataset, test_dataloader, test_dataset
                gc.collect()


if __name__ == '__main__':
    main()
