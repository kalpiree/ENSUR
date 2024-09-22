# Device configuration
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Model and training configuration
model_name = 'GMF'
epochs = 10
batch_size = 256
learning_rate = 0.001
factor = 8
use_pretrain = False
save_model = True

# # Create DataFrame
# df = pd.DataFrame(data)
df = pd.read_csv('path/to/interactions_data_movielens.csv')


# Initialize DataLoader
data_loader = FlexibleDataLoader(df=df, dataset_type='explicit')
processed_data = data_loader.read_data()
train_df, validation_df, test_df, total_df = data_loader.split_train_test()

# Create dataset objects
train_dataset = MovieLens(train_df, total_df, ng_ratio=1,
                          include_features=(model_name == 'FM' or model_name == 'DeepFM'))
validation_dataset = MovieLens(validation_df, total_df, ng_ratio=100,
                               include_features=(model_name == 'FM' or model_name == 'DeepFM'))
test_dataset = MovieLens(test_df, total_df, ng_ratio=100,
                         include_features=(model_name == 'FM' or model_name == 'DeepFM'))

# Prepare DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Dynamically initializing models using dataset metadata
num_users = train_dataset.get_num_users()
num_items = train_dataset.get_num_items()

num_features = train_dataset.get_num_features()

# Model selection and initialization
models = {
    'MLP': MLP(num_users=num_users, num_items=num_items, num_factor=factor),
    'GMF': GMF(num_users=num_users, num_items=num_items, num_factor=factor),
    'NeuMF': NeuMF(num_users=num_users, num_items=num_items, num_factor=factor),
    'WMF': WMF(num_users=num_users, num_items=num_items, num_factors=factor),
    'FM': FactorizationMachine(num_factors=factor, num_features=num_features),
    'DeepFM': DeepFM(num_factors=factor, num_features=num_features),  # Adjust features size as needed
    # 'DeepFM': DeepFM(num_factors=factor, num_features=num_features_explicit)
    'LightGCN': LightGCN(num_users=num_users, num_items=num_items, embedding_size=factor, n_layers=3),
    'SASRec': SASRec(num_items=num_items, embedding_size=factor, num_heads=4, num_layers=2, dropout=0.1),
    'BERT4Rec': BERT4Rec(num_items=num_items, embedding_size=factor, num_heads=4, num_layers=2)
}

model = models[model_name].to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()

trainer = Train(
    model=model,
    optimizer=optimizer,
    epochs=epochs,
    dataloader=train_dataloader,
    criterion=criterion,
    test_obj=test_dataloader,
    device=device,
    print_cost=True,
    use_features=model_name in ['FM', 'DeepFM'],  # Use features for FM and DeepFM
    use_weights=model_name == 'WMF'  # Use weights for WMF

)
trainer.train()
