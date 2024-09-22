import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd


class FlexibleDataLoader:
    """

    Attributes:
      df:
      dataset_type:
    """

    def __init__(self, df, dataset_type='explicit'):
        """

        Args:
          df:
          dataset_type:
        """
        self.df = df
        self.dataset_type = dataset_type

    def read_data(self):
        """

        Returns:

        """
        if self.dataset_type == 'implicit':
            self.df['weight'] = self.calculate_weights(self.df['rating'])
        else:
            self.df['weight'] = 8.0
        return self.df

    def calculate_weights(self, interactions):
        alpha = 0.5
        return 1 + alpha * interactions

    def split_train_test(self):
        """
        Splits the dataset into training, validation, and test sets based on the number of interactions per user.

        Returns:
            train_df: DataFrame containing the training data.
            val_df: DataFrame containing the validation data.
            test_df: DataFrame containing the test data.
        """
        grouped = self.df.groupby('userId')
        test_val_df = grouped.apply(lambda x: x.sample(n=min(2, len(x)), replace=False)).reset_index(drop=True)

        # Assigning 'validation' to all initially
        test_val_df['set'] = 'validation'

        def assign_set(group):
            if len(group) > 1:
                group.iloc[
                    0, group.columns.get_loc('set')] = 'test'
            return group

        test_val_df = test_val_df.groupby('userId').apply(assign_set)

        # Split into validation and test dataframes
        test_df = test_val_df[test_val_df['set'] == 'test']
        val_df = test_val_df[test_val_df['set'] == 'validation']

        # Combining original data with test and validation samples and then removing all duplicates
        combined_df = pd.concat([self.df, test_df, val_df])
        train_df = combined_df.drop_duplicates(subset=['userId', 'itemId'], keep=False)

        return train_df, val_df, test_df, self.df


class MovieLens(Dataset):
    def __init__(self, df, total_df, ng_ratio=1, include_features=False):
        """

        Args:
          df:
          total_df:
          ng_ratio:
          include_features:
        """
        super(MovieLens, self).__init__()
        self.df = df
        self.total_df = total_df
        self.ng_ratio = ng_ratio
        self.include_features = include_features

        # Map user and item IDs to a continuous range starting at 0
        self.user_map = {user_id: idx for idx, user_id in enumerate(df['userId'].unique())}
        self.item_map = {item_id: idx for idx, item_id in enumerate(df['itemId'].unique())}
        self.item_map_reverse = {idx: item_id for item_id, idx in self.item_map.items()}  # Reverse mapping

        # Mapping based on total data
        self.user_map_tot = {user_id: idx for idx, user_id in enumerate(sorted(total_df['userId'].unique()))}
        self.item_map_tot = {item_id: idx for idx, item_id in enumerate(sorted(total_df['itemId'].unique()))}
        self.num_users_tot = len(self.user_map_tot)
        self.num_items_tot = len(self.item_map_tot)
        self.num_features_tot = self.num_users_tot + self.num_items_tot

        self.users, self.items, self.labels, self.weights, self.features, self.ratings, self.item_types, self.groups = self._prepare_data()

    def _prepare_data(self):

        """

        Returns:

        """
        users, items, labels, weights, features, ratings, item_types, groups = [], [], [], [], [], [], [], []

        # for row in self.df.itertuples(index=False):
        for row in tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Preparing Data"):
            u = row.userId
            i = row.itemId
            w = row.weight
            rating = row.rating
            item_type = row.item_type
            group = row.group

            mapped_u = self.user_map_tot[u]
            mapped_i = self.item_map_tot[i]
            users.append(mapped_u)
            items.append(mapped_i)
            labels.append(1)
            weights.append(w)
            ratings.append(row.rating)
            item_types.append(row.item_type)
            groups.append(row.group)

            # additional_info.append((rating, item_type, group))

            if self.include_features:
                user_feature = F.one_hot(torch.tensor(mapped_u), num_classes=self.num_users_tot)
                item_feature = F.one_hot(torch.tensor(mapped_i), num_classes=self.num_items_tot)
                feature = torch.cat((user_feature, item_feature), dim=0)
                features.append(feature)

            user_interacted_items = set(self.total_df[self.total_df['userId'] == u]['itemId'].map(self.item_map_tot))
            potential_negatives = set(self.item_map_tot.values()) - user_interacted_items

            num_negatives = min(len(potential_negatives), self.ng_ratio)
            negative_samples = np.random.choice(list(potential_negatives), num_negatives, replace=False)

            for neg in negative_samples:
                users.append(mapped_u)
                items.append(neg)
                labels.append(0)
                weights.append(1.0)
                ratings.append(0)
                item_types.append(0)
                groups.append(group)

                if self.include_features:
                    negative_feature = torch.cat(
                        (user_feature, F.one_hot(torch.tensor(neg), num_classes=self.num_items_tot)), dim=0)
                    features.append(negative_feature)

        users, items, labels, weights = map(torch.tensor, [users, items, labels, weights])
        features = torch.stack(features) if features else None

        # assert len(users) == len(items) == len(labels) == len(weights) == len(features) == len(ratings)== len(item_types) == len(groups), "Data arrays must have the same length"

        return users, items, labels, weights, features, ratings, item_types, groups

    def __len__(self):
        """

        Returns:

        """
        return len(self.users)

    def __getitem__(self, idx):
        """

        Args:
          idx:

        Returns:

        """
        if self.include_features:
            feature_vector = self.features[idx]
            assert feature_vector.shape[
                       0] == self.get_num_features(), f"Feature vector size mismatch: {feature_vector.shape[0]} != {self.get_num_features()}"
            # Return user and item indices, labels, weights, followed by the feature vector
            return self.users[idx], self.items[idx], self.labels[idx], self.weights[idx], feature_vector, self.ratings[
                idx], self.item_types[idx], self.groups[idx]
        else:
            # Return user and item indices, labels, and weights without features
            return self.users[idx], self.items[idx], self.labels[idx], self.weights[idx], self.ratings[idx], \
            self.item_types[idx], self.groups[idx]

    def get_feature_info(self):
        """

        Returns:

        """
        if self.include_features:
            return self.num_features_tot, self.num_users_tot, self.num_items_tot
        else:
            return None

    def get_num_users(self):
        """

        Returns:

        """
        return self.num_users_tot

    def get_num_items(self):
        """

        Returns:

        """
        return self.num_items_tot

    def get_num_features(self):
        return self.num_features_tot
