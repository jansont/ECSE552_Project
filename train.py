from pytorch_lightning import Trainer
from pytorch_lightning import utilities

from dataloaders import get_iterators
from models.GRU import GRU
import pandas as pd

AVAIL_GPUS = 0
SEED = 0

EPOCHS = 100   

'''
Test:
- GCN with full data
- 
'''

historical_len = 1
batch_size = 16
pred_len = 1

input_dim = 108      #input to GRU 9 x (num node feat )
HIDDEN_DIM = 128
output_dim = 9     #output to GRU (number of nodes)
n_layers = 2


utilities.seed.seed_everything(seed=SEED)

train_dl, val_dl, edge_idx = get_iterators(
    historical_len = historical_len,
    pred_len = pred_len,
    batch_size = batch_size,
)

model = GRU(
    input_dim=input_dim, hidden_dim=HIDDEN_DIM, 
    output_dim=output_dim, n_layers=n_layers,
    edge_idx=edge_idx
    )

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=EPOCHS
)

# for x in val_dl: 
#     break
# print(len(x[0]))
# print(x[0][0])
# assert False

trainer.fit(model, train_dl, val_dl)

training_df = {'Loss': model.training_losses, 'Error': model.training_metrics}
training_df  = pd.DataFrame(training_df)
training_df.to_csv('/Users/alixdanglejan-chatillon/ECSE552_Project/Results/TrainingResults1.csv')

validation_df = {'Loss': model.validation_losses, 'Error': model.validation_metrics}
validation_df  = pd.DataFrame(validation_df)
validation_df.to_csv('/Users/alixdanglejan-chatillon/ECSE552_Project/Results/ValidationResults1.csv')

