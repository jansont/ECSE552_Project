import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule


class GRU(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.rnncell = nn.GRUCell(input_dim, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.loss_func = nn.MSELoss()

    def forward(self, X, fc=True):
        """
        X : (node_features, edge_index, edge_features, labels_x)
        fc: To return the fully connected result or the hidden states
        res_total has shape (Seq_len, Batch size, hidden dims/output dims
        """
        print(X[0].shape)
        seq_len = X[0].shape[1]
        batch_size = X[0].shape[0]

        h = torch.zeros(batch_size, self.hidden_dim)
        out_total, h_total = [], []

        for i in range(seq_len):
            h = self.rnncell(X[0][:, i], h)
            out = self.fc(h)

            out_total.append(out)
            h_total.append(h)

        out_total = torch.stack(out_total)
        h_total = torch.stack(h_total)

        res_total = out_total if fc else h_total

        return res_total[:-1]


    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return y, pred

    def step(self, batch, batch_idx):
        y, pred = self.predict_step(batch, batch_idx)
        loss = self.loss_func(y, pred)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
