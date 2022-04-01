import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import Sequential, Linear, Sigmoid
from torch.nn import functional as F
from torch_scatter import scatter_add

class GraphGNN(nn.Module):
    def __init__(self, edge_index, in_dim, out_dim):
        super(GraphGNN, self).__init__()
        self.edge_index = torch.LongTensor(edge_index)

        e_h = 32
        e_out = 30
        n_out = out_dim

        self.edge_mlp = Sequential(Linear(in_dim * 2 + 2 + 1, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        """
        takes in edge_weights, edge_features,         
        """
        edge_features, edge_weight = x
        edge_index = self.edge_index

        edge_src, edge_target = edge_index
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]

        out = torch.cat([node_src, node_target, edge_weight], dim=-1)
        out = self.edge_mlp(out)

        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1))  # For higher version of PyG.

        out = out_add + out_sub
        out = self.node_mlp(out)

        return out


class GRU(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, edge_idx):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.edge_index = edge_idx

        self.gnn_in = gnn_in
        self.gnn_out = 13

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc = nn.Linear(self.hid_dim, self.out_dim)

        self.graph_gnn = GraphGNN(edge_idx, self.gnn_in, self.gnn_out)
        self.rnncell = nn.GRUCell(input_dim, hidden_dim, n_layers)

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
