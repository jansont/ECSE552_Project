from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from dataloaders import get_iterators
from models.GRU import GRU, EdgeGNN, GCN
import os
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from pytorch_forecasting.metrics import MAPE

# DATA CONFIG
data_file = 'LA_DATA_2018_02_to_2018_06.csv'
# node_cols = 'visibility', 'precipitation_depth']
node_cols = ['temperature', 'pressure', 'ceiling', 'dew', 'precipitation_duration' , 'mean_aod','min_aod','max_aod']                                                 
edge_cols = ['wind_x', 'wind_y']

results_folder = 'Results'
# '/Users/alixdanglejan-chatillon/ECSE552_Project/Results/

# GRAPH CONFIG
dist_thresh = 30e6
multi_edge_feature = False
use_self_loops = True

#GRU CONFIG
historical_len = 1
batch_size = 16
pred_len = 1
HIDDEN_DIM = 128
n_layers = 2
graph_model = 'EdgeGNN'

#GRAPH HYP.
dropout = 0.3

#DATA CONFIG
train_test_split = 0.8

#TRAINER CONFIG
AVAIL_GPUS = 0
SEED = 0
EPOCHS = 50   
NUM_WORKERS = 0


#OPTIMIZER
learning_rate = 1e-3
weight_decay = 1e-5
criterion = nn.L1Loss()
# criterion = nn.MSELoss()
metrics = MAPE()

utilities.seed.seed_everything(seed=SEED)

train_dl, val_dl, edge_idx = get_iterators(data_file,
                                            edge_cols, 
                                            node_cols, 
                                            split = train_test_split,
                                            historical_len = historical_len,
                                            pred_len = pred_len,
                                            batch_size = batch_size,
                                            dist_thresh = dist_thresh, 
                                            multi_edge_feature = multi_edge_feature,
                                            use_self_loops =  use_self_loops, 
                                            num_workers = NUM_WORKERS)


if graph_model == 'EdgeGNN':
    graph_model = EdgeGNN(edge_idx, in_dim = 17, dropout = dropout)
elif graph_model == 'GCN':
    graph_model = GCN(in_channels = len(node_cols), dropout = dropout)
else:
    assert False

gru_input_dim = (len(node_cols) + len(edge_cols)) * (np.max(edge_idx) + 1)
gru_output_dim = np.max(edge_idx)+1

model = GRU(
    input_dim = gru_input_dim,
    hidden_dim=HIDDEN_DIM, 
    output_dim= gru_output_dim,
    n_layers=n_layers,
    edge_idx=edge_idx, 
    graph_model=graph_model, 
    criterion = criterion, 
    metrics = metrics, 
    learning_rate = learning_rate, 
    weight_decay = weight_decay
    )

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=EPOCHS
)



trainer.fit(model, train_dl, val_dl)

training_df = {'Loss': model.training_losses, 'Error': model.training_metrics}
training_df  = pd.DataFrame(training_df)
training_df.to_csv(os.path.join(results_folder, 'TrainingResults1.csv'))

validation_df = {'Loss': model.validation_losses, 'Error': model.validation_metrics}
validation_df  = pd.DataFrame(validation_df)
validation_df.to_csv(os.path.join(results_folder, 'ValidationResults1.csv'))
