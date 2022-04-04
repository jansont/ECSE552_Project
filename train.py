from pytorch_lightning import Trainer
from pytorch_lightning import utilities

from dataloaders import get_iterators
from models.GRU import GRU

AVAIL_GPUS = 0
SEED = 0

EPOCHS = 100

historical_len = 7
batch_size = 16
pred_len = 1

input_dim = 54
HIDDEN_DIM = 128
output_dim = 9
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

trainer.fit(model, train_dl, val_dl)
