import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import Sequential, Linear, Sigmoid
from torch.nn import functional as F
from torch_scatter import scatter_add
from torchmetrics import MeanSquaredError

class GraphGNN(nn.Module):
    def __init__(self, edge_index, in_dim, out_dim):
        super(GraphGNN, self).__init__()
        self.edge_index = torch.LongTensor(edge_index)

        e_h = 32
        e_out = 30
        n_out = out_dim

        self.edge_mlp = Sequential(Linear(in_dim, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        """
        x = (node_features, edge_features)
        """
        node_features, edge_weight = x
        edge_src, edge_target = self.edge_index
        # print(node_features.shape)
        # print(edge_src.shape)
        # print(self.edge_index.shape)
        # print(edge_weight.shpe)
        node_src = node_features[:, edge_src]
        node_target = node_features[:, edge_target]
        # print(node_src.shape)
        # print(node_target.shape)

        out = torch.cat([node_src, node_target, edge_weight.unsqueeze(-1)], dim=-1).float()
        # print(out.shape)
        out = self.edge_mlp(out)
        # print(out.shape)


        out_add = scatter_add(out, edge_target, dim=1, dim_size=node_features.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=node_features.size(1))  # For higher version of PyG.

        out = out_add + out_sub
        out = self.node_mlp(out)

        return out


class GRU(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, edge_idx):
        super().__init__()

        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = output_dim
        self.edge_index = edge_idx

        self.gnn_in = 11
        self.gnn_out = 1

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.graph_gnn = GraphGNN(edge_idx, self.gnn_in, self.gnn_out)
        self.rnncell = nn.GRUCell(input_dim, hidden_dim, n_layers)

        self.loss_func = nn.MSELoss()
        self.metric = MeanSquaredError(squared=False)

    def forward(self, X):
        """
        X : (node_features, edge_features, labels_x)
        fc: To return the fully connected result or the hidden states
        res_total has shape (Seq_len, Batch size, hidden dims/output dims
        """
        (node_features, edge_features, labels_x, lengths) = X
        seq_len = node_features.shape[1]
        batch_size = node_features.shape[0]

        h = torch.zeros(batch_size, self.hid_dim)
        out_total= []
        for i in range(seq_len):
            graph_input = (node_features[:, i], edge_features[:, i])
            graph_out = self.graph_gnn(graph_input)
            # x = torch.cat((labels_x[:], feature[:, self.hist_len + i]), dim=-1)
            # print(graph_out.shape, node_features[:, i].shape)

            x = torch.cat([graph_out, node_features[:, i]], dim=-1)
            x = torch.flatten(x, 1, -1)
            h = self.rnncell(x.float(), h)
            out = self.fc_out(h)
            out_total.append(out)

        out_total = torch.stack(out_total)
        out_total = torch.stack([out_total[lengths[j]-1, j]
                        for j in range(batch_size)])

        return out_total.squeeze()


    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return y.squeeze().float(), pred

    def step(self, batch, batch_idx):
        y, pred = self.predict_step(batch, batch_idx)
        loss = self.loss_func(y, pred)
        metric = self.metric(pred, y)
        self.log('RMSE', self.metric)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
