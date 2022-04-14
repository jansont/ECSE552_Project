from sklearn import metrics
import torch.nn as nn
import torch.optim
from pytorch_lightning import LightningModule
from torch.nn import Sequential, Linear, Sigmoid
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch.nn import ReLU
from torch_geometric.nn import GCNConv

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

        

class GCN(torch.nn.Module):
    def __init__(self, in_channels, dropout):
        super(GCN, self).__init__()
        self.h1 = 16
        self.h2 = 10
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GCNConv(in_channels, self.h1)
        self.conv2 = GCNConv(self.h1, self.h1)
        self.conv3 = GCNConv(self.h1, 1)


    def forward(self, x, edge_index) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = nn.ReLU()
        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = nn.ReLU(x)
        x = self.conv3(x, edge_index)
        return x

class EdgeGNN(nn.Module):
    def __init__(self, edge_index, in_dim, dropout):
        super(EdgeGNN, self).__init__()
        self.edge_index = torch.LongTensor(edge_index)

        e_h = 16
        e_out = 16
        n_out = 1
        e_h2 = 16
        n_h = 12

        self.edge_mlp = Sequential(Linear(in_dim, e_h),
                                   nn.Dropout(dropout),                                   
                                   ReLU(),
                                   Linear(e_h, e_h2),
                                   nn.Dropout(dropout),                                  
                                   ReLU(),
                                   Linear(e_h2, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_h),
                                    nn.Dropout(dropout),       
                                   ReLU(),
                                   Linear(n_h, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        """
        x = (node_features, edge_features)
        """
        node_features, edge_weight = x
        edge_src, edge_target = self.edge_index

        node_src = node_features[:, edge_src]
        node_target = node_features[:, edge_target]

        out = torch.cat([node_src, node_target, edge_weight.unsqueeze(-1)], dim=-1).float()

        out = self.edge_mlp(out)
        out_add = scatter_add(out, edge_target, dim=1, dim_size=node_features.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=node_features.size(1))  # For higher version of PyG.

        out = out_add + out_sub
        out = self.node_mlp(out)
        return out


class GRU(LightningModule):
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                n_layers,
                edge_idx,
                graph_model,
                criterion, 
                learning_rate, 
                weight_decay, 
                metrics):

        super().__init__()

        self.in_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = output_dim
        self.edge_index = torch.LongTensor(edge_idx)

        self.graph_model = graph_model

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.rnncell = nn.GRUCell(input_dim, hidden_dim, n_layers)

        self.loss_func = criterion
        self.metric = metrics
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.training_losses = []
        self.validation_losses = []
        self.training_metrics = []
        self.validation_metrics = []

    def forward(self, X):
        """
        X : (node_features, edge_features, labels_x)
        """
        (node_features, edge_features, labels_x, lengths) = X
        seq_len = node_features.shape[1]
        batch_size = node_features.shape[0]
    

        h = torch.zeros(batch_size, self.hid_dim)
        out_total= []
        for i in range(seq_len):

            if isinstance(self.graph_model, GCN):
                graph_input = (node_features[:, i])
                graph_out = self.graph_model(graph_input.float(), self.edge_index)
            elif isinstance(self.graph_model, EdgeGNN):
                graph_input = (node_features[:, i], edge_features[:, i])
                graph_out = self.graph_model(graph_input)

            graph_out = graph_out.squeeze()

            pseudo_index = torch.isnan(labels_x[i])
            pseudo_labels = torch.zeros(labels_x[i].shape)
            pseudo_labels[pseudo_index] = graph_out[pseudo_index]

            pseudo_labels[~pseudo_index] = labels_x[i][~pseudo_index]
            # total = graph_out[labels_x.shape[-1]:].reshape([batch_size, labels_x.shape[-1], 4])
            
            pseudo_labels = pseudo_labels.unsqueeze(dim = -1)

            graph_out = graph_out.unsqueeze(-1)

            x = torch.cat([pseudo_labels, node_features[:, i], graph_out ], dim=-1)
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
        self.log('RMSE', metric)
        return loss, metric

    def training_step(self, batch, batch_idx):
        training_loss, training_metric = self.step(batch, batch_idx)
        self.training_losses.append(training_loss.detach().numpy())
        self.training_metrics.append(training_metric.detach().numpy())
        return training_loss

    def validation_step(self, batch, batch_idx):
        validation_loss, validation_metric = self.step(batch, batch_idx)
        self.validation_losses.append(validation_loss.detach().numpy())
        self.validation_metrics.append(validation_metric.detach().numpy())
        return validation_loss

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def configure_optimizers(self):
        return  torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

