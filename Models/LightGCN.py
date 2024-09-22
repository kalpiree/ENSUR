import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, n_layers):
        super(LightGCN, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, user_indices, item_indices):
        # Initial embeddings
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)

        # Layer propagation
        all_user_embeddings = user_embedding
        all_item_embeddings = item_embedding

        for _ in range(self.n_layers):
            user_embedding = torch.mean(all_user_embeddings, dim=0, keepdim=True)
            item_embedding = torch.mean(all_item_embeddings, dim=0, keepdim=True)
            all_user_embeddings += user_embedding
            all_item_embeddings += item_embedding

        # Final embedding (mean of all layers)
        final_user_embedding = all_user_embeddings / (self.n_layers + 1)
        final_item_embedding = all_item_embeddings / (self.n_layers + 1)

        # Calculate interaction scores as the dot product of user and item embeddings
        interaction = torch.sum(final_user_embedding * final_item_embedding, dim=1)
        return torch.sigmoid(interaction)
