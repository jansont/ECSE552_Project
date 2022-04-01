from tabnanny import check
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import date, timedelta
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, Dataset
# from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error


def data_to_numpy(weather_data, edge_names, node_names, station_ids, date_range): 
    checkpt = 'checkpt'
    if not os.path.exists(checkpt):
        print('Checkpt doesnt exist, making it')
        os.makedirs(checkpt)

        graph_node_features = np.empty((len(date_range), len(stations), len(node_cols)))
        graph_edge_features = np.empty((len(date_range), len(stations), len(edge_cols)))
        graph_labels = np.empty((len(date_range), len(stations)))

        for day_idx in range(len(date_range)): 
            for station_idx in range(len(stations)): 
                    d = date_range[day_idx]             #get date from index
                    station = stations[station_idx]     #get station number from index
                    vals = weather_data[weather_data['DATE'] == d]  #get data date using date
                    vals = vals[vals['STATION'] == station]         #get data of station on date
                    pm = vals['pm25'].values
                    edge = vals[edge_cols]
                    edge_vals = np.array(edge.values.tolist()).flatten()  #edge feature as array
                    node_vals = vals[node_cols]
                    node_vals = np.array(node_vals.values.tolist()).flatten() #node features as array
                    if len(node_vals) == 0:                           #if there is no weather data on date, set to all zeros
                        node_vals = np.zeros(len(node_cols))
                    graph_node_features[day_idx, station_idx] = node_vals 
                    if len(pm) == 0:     #if no pm label on date set to 0
                        pm = np.zeros(1)
                    graph_labels[day_idx, station_idx] = pm
                    if len(edge_vals) == 0:  #if no edge feature, set to 0
                        edge_vals = np.zeros(len(edge_cols))
                    graph_edge_features[day_idx, station_idx] = edge_vals
        np.save(os.path.join(checkpt, 'graph_node_features'), graph_node_features)
        np.save(os.path.join(checkpt, 'graph_edge_features'), graph_edge_features)
        np.save(os.path.join(checkpt, 'graph_labels'), graph_labels)

    else:
        print('Found Checkpoint, loading')
        graph_node_features = np.load(os.path.join(checkpt, 'graph_node_features.npy') )
        graph_edge_features = np.load(os.path.join(checkpt, 'graph_edge_features.npy') )
        graph_labels = np.load(os.path.join(checkpt, 'graph_labels.npy') )

    return graph_node_features, graph_edge_features, graph_labels

def generate_nodes(metadata):
    stations = metadata['STATION'].unique()
    stations.sort()
    nodes = OrderedDict()
    for id in stations:
        row = metadata[metadata['STATION'] == id]
        lat = row['Latitudes'].values[0]
        lon = row['Longitudes'].values[0]
        # elev = row['Elevetation']
        nodes.update({id:{'Latitude':lat,'Longitude':lon}})
    return nodes

def get_node_features(nodes):
    altitudes = [nodes[id]['altitude'] for id in nodes]
    node_features = np.array(altitudes)
    return node_features


def geo_distance(first_node, second_node):
        '''Haversine formula for geodesic distance'''
        lat1, long1 = first_node
        lat2, long2 = second_node
        R = 6371e3 #m
        lat1 = lat1 * np.pi/180; #rad
        lat2 = lat2 * np.pi/180; #rad
        delta_lat = (lat2 - lat1) * np.pi/180;
        delta_long = (long1 - long2) * np.pi/180;
        a = (np.sin(delta_lat/2))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_long/2) * np.sin(delta_long/2)
        c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
        d = R * c #m
        return d

def generate_node_distances(coordinates):
    distance_matrix = np.zeros((len(coordinates), len(coordinates)))
    for i in range(len(coordinates)): 
        coord1 = coordinates[i]
        for j in range(len(coordinates)): 
            coord2 = coordinates[j]
            distance = geo_distance(coord1, coord2)
            distance_matrix[i][j] = distance
    return distance_matrix




def sparse_adjacency(adj): 
    """Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0), edge_attr


class Graph():
    def __init__(self,
                 graph_metadata,
                 edge_data, 
                 distance_threshold):
        self.graph_metadata = graph_metadata
        self.distance_threshold = distance_threshold
        self.nodes = generate_nodes(graph_metadata)
        self.size = len(self.nodes)
        self.edge_data = edge_data
        self.edge_index, self.edge_attr = self.generate_edges()
        self.edge_attr = self.edge_attr.transpose()
        self.adjacency = self.edge_list_sequence_to_adj()

    def generate_edges(self):
        nodes = self.nodes             
        node_list = list(self.nodes)        #get list of node ids
        coordinates = list(zip(self.graph_metadata['Latitudes'], self.graph_metadata['Longitudes']))  #zip latitude and longitude of all nodes
        distance_matrix = generate_node_distances(coordinates)  #square matrix of distance to all other nodes

        adj_matrix = np.identity(self.size)
        adj_matrix[distance_matrix < self.distance_threshold] = 1 #in adj matrix, set entry to 1 if distance between nodes below threshold

        distance_matrix = distance_matrix * adj_matrix

        edge_idx, edge_dist = sparse_adjacency(torch.tensor(distance_matrix)) #edge_idx : shape (2 * number of connected nodes (dis < threshold)) 
        edge_idx, edge_dist = edge_idx.numpy(), edge_dist.numpy()             #edge_dist: same shape as above, distance values between those node indices
        edge_vectors = []#other edge attributes?

        for i in range(edge_idx.shape[1]):
            #get index of non-zero edges
            source_idx = edge_idx[0, i]
            dest_idx = edge_idx[1, i]
            #get lat lon for the nodes at ends of non zero edge
            key0 = node_list[source_idx]
            lat0 = nodes[key0]['Latitude']
            long0 = nodes[key0]['Longitude']
            key1 = node_list[dest_idx]
            lat1 = nodes[key1]['Latitude']
            long1 = nodes[key1]['Longitude']
            distance_x = geo_distance((lat0, 0), (lat1, 0))
            distance_y = geo_distance((0, long1), (0, long1))
            edge_vect_x = distance_x 
            #get wind along x and y vectors from both source and edge
            wind_source_x = self.edge_data[:, source_idx, 0]
            wind_dest_x = self.edge_data[:, dest_idx, 0]
            wind_source_y = self.edge_data[:, source_idx, 1]
            wind_dest_y = self.edge_data[:, dest_idx, 1]
            #average source and edge wind components to get net wind 
            wind_x = (wind_source_x + wind_dest_x)/2
            wind_y = (wind_source_y + wind_dest_y)/2

            diffusion_x = distance_x * wind_x 
            diffusion_y = distance_y * wind_y

            dist_dir = np.tan(distance_y / (distance_x))
            wind_dir = np.tan(wind_y / (wind_x+1e-12))
            wind_str = wind_x**2 + wind_y**2
            distance = distance_x**2 + distance_y**2
            delta_angle = np.abs(wind_dir - dist_dir)
            edge_weight = (wind_str * np.cos(delta_angle) / distance**0.5)
            #normalize edge weights
            mean_edge_weight = np.mean(edge_weight)
            std_edge_weight = np.std(edge_weight)
            normalized_edge_weight = (edge_weight - mean_edge_weight) / std_edge_weight

            edge_vectors.append(edge_weight)

        edge_vectors = np.stack(edge_vectors)

        return edge_idx, edge_vectors

    def edge_list_to_adj(self):
        adj = np.identity(self.edge_index.size)
        for k,(i,j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
            adj[i,j] = self.edge_attr[j]
        return adj

    def edge_list_sequence_to_adj(self):
        adjacencies = []
        for i in range(self.edge_attr.shape[0]):
            adj = np.identity(self.size)
            for k,(i,j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
                adj[i,j] = self.edge_attr[i][k]
            adjacencies.append(adj)
        return np.array(adjacencies)

def normalize(vect, mean, std):
    norm_v = (vect - mean) / std
    return norm_v

class WeatherData(Dataset):
    def __init__(self, 
                 labels,   
                 node_features,
                 edge_features, 
                 historical_len, 
                 prediction_len
                 ):
        
        self.historical_len = historical_len
        self.pred_len = prediction_len
        self.seq_len = historical_len + prediction_len

        self.edge_features = edge_features

        label_mean = labels.mean()
        label_sdev = labels.std()
        self.labels = normalize(labels, label_mean, label_sdev)

        feat_mean = node_features.mean(axis=1)
        feat_sdev = node_features.std(axis=1)
        for i in range(self.features.shape[1]):
            self.features[:, i] = (self.features[:, i] - feat_mean[i])/feat_sdev(i)

    def __len__(self):
        return len(self.labels - self.historical_len)

    def __getitem__(self, idx):
        features = self.features[idx: idx + self.historical_len]
        edge_features = self.edge_features[idx:idx + self.historical_len]
        labels_x = self.labels[idx: idx + self.historical_len]
        labels_y = self.labels[idx + self.historical_len: idx + self.seq_len]
        return (features, edge_features, labels_x), labels_y


def get_iterators(historical_len, pred_len, batch_size):
    path = './data/'
    file1 = 'la_weather_with_pm_per_day.csv'
    file2 = 'metadata_with_station.csv'

    weather_data = pd.read_csv(path+file1)
    metadata = pd.read_csv(path+file2)
    metadata = metadata[metadata['location'] == 'Los Angeles (SoCAB)'].reset_index(drop = True)

    start = date(2018,2,1)
    end = date(2021,1,1)
    date_range = pd.date_range(start,end-timedelta(days=1))
    date_range = [str(x)[:10] for x in date_range]

    node_cols = ['ceiling', 'visibility', 'dew', 'precipitation_duration', 'precipitation_depth']
    edge_cols = ['wind_x', 'wind_y']
    stations = metadata['STATION'].unique()
    stations.sort()

    graph_node_features, graph_edge_features, graph_labels = data_to_numpy(weather_data, edge_cols, node_cols, stations, date_range)
    graph_node_features = np.nan_to_num(graph_node_features)

    print(graph_node_features.shape)
    print(graph_edge_features.shape)
    print(graph_labels.shape)

    graph = Graph(metadata, graph_edge_features, distance_threshold = 30e3)
    split = int(graph_labels.shape[0]*0.8)

    train_dataset = WeatherData(edge_features = graph.edge_attr[:split], 
                        labels = graph_labels[:split],
                        node_features = graph_node_features[:split],
                        historical_len = historical_len,
                        prediction_len = pred_len
                        )


    val_dataset = WeatherData(edge_features = graph.edge_attr[split:], 
                        labels = graph_labels[split:],
                        node_features = graph_node_features[split:],
                        historical_len = historical_len,
                        prediction_len = pred_len
                        )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, graph.edge_index

# def __test__dl__():
#     for batch in train_dl:
#         (node_features, edge_index, edge_features, labels_x), labels_y = batch
#         print(node_features.shape)
#         print(edge_index.shape)
#         print(edge_features.shape)
#         print(labels_x.shape)
#         print(labels_y.shape)
#         # print(labels_x[0], labels_y[0])
#         break
