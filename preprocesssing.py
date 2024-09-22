import os

import pandas as pd
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process datasets to group users and classify items.")

    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the input dataset file.")

    parser.add_argument('--output_folder', type=str, default='./',
                        help="Folder to save the processed datasets (default: './').")

    parser.add_argument('--is_implicit', action='store_true',
                        help="Flag to indicate if the dataset is implicit feedback.")

    parser.add_argument('--user_top_fraction', type=float, default=0.5,
                        help="Fraction of top users to consider (default: 0.5).")

    parser.add_argument('--methods', nargs='+', default=['popular_consumption', 'interactions'],
                        help="List of methods to process the dataset (default: ['popular_consumption', 'interactions']).")

    parser.add_argument('--datasets', nargs='+', default=['amazonoffice', 'lastfm'],
                        help="List of dataset names (default: ['amazonoffice', 'lastfm']).")

    return parser.parse_args()


def read_implicit_item_popularity(dataset):
    items_freq = {}
    user_interactions = {}
    for eachline in dataset.itertuples(index=False):
        uid, iid, count = int(eachline.userId), int(eachline.itemId), int(eachline.rating)
        user_interactions.setdefault(uid, []).extend([iid] * count)
        items_freq[iid] = items_freq.get(iid, 0) + count
    return items_freq, user_interactions


def read_explicit_item_popularity(dataset):
    items_freq = {}
    user_interactions = {}
    for eachline in dataset.itertuples(index=False):
        uid, iid = int(eachline.userId), int(eachline.itemId)
        user_interactions.setdefault(uid, []).append(iid)
        items_freq[iid] = items_freq.get(iid, 0) + 1
    return items_freq, user_interactions


def determine_top_items(items_freq, item_top_fraction):
    sorted_items = sorted(items_freq.items(), key=lambda x: x[1], reverse=True)
    top_items_count = int(len(items_freq) * item_top_fraction)
    return set([item for item, _ in sorted_items[:top_items_count]])


def update_dataset_with_groups(dataset, user_top_fraction, method, items_freq, user_interactions):
    short_heads = determine_top_items(items_freq, 0.2)
    dataset['item_type'] = dataset['itemId'].apply(lambda x: 1 if x in short_heads else 2)

    user_profile_pop_df = pd.DataFrame(
        [(uid, len(set(items) & short_heads), len(items)) for uid, items in user_interactions.items()],
        columns=['uid', 'pop_count', 'profile_size']
    )

    if method == "popular_consumption":
        user_profile_pop_df.sort_values(['pop_count', 'profile_size'], ascending=(False, False), inplace=True)
    else:
        user_profile_pop_df.sort_values(['profile_size'], ascending=False, inplace=True)

    num_top_users = int(user_top_fraction * len(user_interactions))
    found = False
    adjustment = 0
    deviation = int(0.3 * len(user_interactions))
    while not found and adjustment <= deviation:
        index = num_top_users + (adjustment - int(deviation / 2))
        if index <= 0 or index >= len(user_interactions):
            adjustment += 1
            continue

        if method == "popular_consumption":
            condition = (user_profile_pop_df.iloc[index - 1]['pop_count'] > user_profile_pop_df.iloc[index][
                'pop_count'] + 1 and
                         user_profile_pop_df.iloc[index - 1]['profile_size'] > user_profile_pop_df.iloc[index][
                             'profile_size'] + 1)
        else:
            condition = user_profile_pop_df.iloc[index - 1]['profile_size'] > user_profile_pop_df.iloc[index][
                'profile_size'] + 1

        if condition:
            found = True
        else:
            adjustment += 1

    advantaged_users = user_profile_pop_df.head(index)
    disadvantaged_users = user_profile_pop_df.iloc[index:]

    dataset['group'] = dataset['userId'].apply(lambda x: 1 if x in advantaged_users['uid'].values else 2)

    print("Advantaged Users:")
    print("Number of Users:", len(advantaged_users))
    print("Max Total Interactions:", advantaged_users['profile_size'].max())
    print("Min Total Interactions:", advantaged_users['profile_size'].min())
    print("Max Popular Item Interactions:", advantaged_users['pop_count'].max())
    print("Min Popular Item Interactions:", advantaged_users['pop_count'].min())

    print("\nDisadvantaged Users:")
    print("Number of Users:", len(disadvantaged_users))
    print("Max Total Interactions:", disadvantaged_users['profile_size'].max())
    print("Min Total Interactions:", disadvantaged_users['profile_size'].min())
    print("Max Popular Item Interactions:", disadvantaged_users['pop_count'].max())
    print("Min Popular Item Interactions:", disadvantaged_users['pop_count'].min())

    return dataset


def main():
    args = parse_arguments()

    for dataset_name in args.datasets:
        input_file = args.input_file.replace('amazonoffice', dataset_name)
        dataset = pd.read_csv(input_file, sep="\s+", header=0,
                              names=['userId', 'itemId', 'rating'], dtype={'userId': int, 'itemId': int, 'rating': int},
                              skiprows=1)
        dataset = dataset.dropna()

        if args.is_implicit:
            items_freq, user_interactions = read_implicit_item_popularity(dataset)
        else:
            items_freq, user_interactions = read_explicit_item_popularity(dataset)

        for method in args.methods:
            updated_dataset = update_dataset_with_groups(dataset.copy(), args.user_top_fraction, method, items_freq,
                                                         user_interactions)
            output_file = os.path.join(args.output_folder, f'{method}_data_{dataset_name}.csv')
            updated_dataset.to_csv(output_file, index=False)
            print(f"Processed dataset saved to {output_file}")


if __name__ == '__main__':
    main()
