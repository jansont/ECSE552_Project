from pytorch_lightning import Trainer
from pytorch_lightning import utilities

from dataloaders import get_iterators
from models.GRU import GRU

AVAIL_GPUS = 0
SEED = 0

EPOCHS = 2

historical_len = 7
batch_size = 16
pred_len = 1

input_dim=13
HIDDEN_DIM = 16
output_dim=1
n_layers = 2

model = GRU(
    hidden_dim=HIDDEN_DIM, input_dim=input_dim,
    output_dim=output_dim, n_layers=n_layers
            )

utilities.seed.seed_everything(seed=SEED)

train_dl, val_dl = get_iterators(
    historical_len = historical_len,
    pred_len = pred_len,
    batch_size = batch_size,
)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=EPOCHS
)

trainer.fit(model, train_dl, val_dl)
