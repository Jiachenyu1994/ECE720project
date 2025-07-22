import re
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch_geometric.nn import NNConv
from torch_geometric.utils import degree
import os
import random
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm
import networkx as nx
import pandas as pd


NESTLIST_FOLDER = './test_ext'  # Path to the folder containing nestlist files

def extract_info_netlist(spice_file):
    transistors=[]
    with open(spice_file,'r') as f:
        spice_data=f.read()
        lines=spice_data.splitlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('device msubckt'):
                features=line.split(" ")
                # print(features)
                transistor_info={
                    'model': features[2],
                    "x_location" : (float(features[3])+float(features[5]))/2*0.005,
                    "y_location" : (float(features[4])+float(features[6]))/2*0.005,
                    'width': float(features[8].split('=')[1])*.005,
                    'length': float(features[7].split('=')[1])*.005,
                    "source": features[16],
                    'gate': features[10],
                    'drain': features[13],
                    'bulk': features[9],
                }
                transistors.append(transistor_info)
        if not transistors:
            print(f"{spice_file}")
        # else:
        #     print(f"Extracted {len(transistors)} transistors from {spice_file}")
        return transistors

def edges_info(transistors):
    edges=[]
    for i in range(len(transistors)):
        for j in range(i+1,len(transistors)):
            transistor_i=transistors[i]
            nets_i=[ transistor_i["source"],transistor_i["gate"],transistor_i["drain"],transistor_i["bulk"]]
            transistor_j=transistors[j]
            nets_j=[ transistor_j["source"],transistor_j["gate"],transistor_j["drain"],transistor_j["bulk"]]
            if set(nets_i) & set(nets_j):
                edge=(i,j,{"shared_nets":list(set(nets_i) & set(nets_j))})
            edges.append(edge)

    return edges

def build_nodes(transistors):
    model_map = {'sky130_fd_pr__nfet_01v8': 0, 
                 'sky130_fd_pr__nfet_01v8_hvt': 1,
                 'sky130_fd_pr__nfet_01v8_lvt': 2,
                 'sky130_fd_pr__pfet_01v8': 3,
                 'sky130_fd_pr__pfet_01v8_hvt': 4,
                 'sky130_fd_pr__pfet_01v8_lvt': 5,
                 'sky130_fd_pr__special_nfet_01v8': 6,
                 'sky130_fd_pr__special_pfet_01v8': 7,
                 'sky130_fd_pr__res_generic_pos': 8,
                 'sky130_fd_pr__res_generic_neg': 9,
                 'sky130_fd_pr__res_generic_pos_hvt': 10,
                 'sky130_fd_pr__res_generic_neg_hvt': 11,
                 'sky130_fd_pr__res_generic_pos_lvt': 12,
                 'sky130_fd_pr__res_generic_neg_lvt': 13,
                 'sky130_fd_pr__special_pfet_01v8_hvt': 14,
                 'sky130_fd_pr__special_pfet_01v8_lvt': 15
                 }

    nodes = []
    xylocation = []
    for t in transistors:
        width = t["width"]
        length = t["length"]
        x_location = t["x_location"]
        y_location = t["y_location"]
        model_vec = [0] * len(model_map)
        model_vec[model_map[t["model"]]] = 1
        node_feature = [width, length] + model_vec
        nodes.append(node_feature)
        xylocation.append((x_location, y_location))

    x = torch.tensor(nodes, dtype=torch.float)
    xylocation = torch.tensor(xylocation, dtype=torch.float)
    return x, xylocation

def build_edges(edge_list):
    edge_index = []
    for src, dst, _ in edge_list:  # or edge_list_with_attrs
        edge_index.append([src, dst])
        edge_index.append([dst, src])

    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    edge_attr = []
    for src, dst, attr in edge_list:
        shared_count = len(attr["shared_nets"])
        edge_attr.append([shared_count])
        edge_attr.append([shared_count]) 

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index,edge_attr


def generate_structural_features(edge_index):
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    degree = torch.tensor([val for _, val in G.degree()], dtype=torch.float).unsqueeze(1)
    clustering = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float).unsqueeze(1)
    eig_centrality = torch.tensor(list(nx.eigenvector_centrality(G).values()), dtype=torch.float).unsqueeze(1)
    betweenness = torch.tensor(list(nx.betweenness_centrality(G).values()), dtype=torch.float).unsqueeze(1)
    closeness = torch.tensor(list(nx.closeness_centrality(G).values()), dtype=torch.float).unsqueeze(1)
    pagerank = torch.tensor(list(nx.pagerank(G).values()), dtype=torch.float).unsqueeze(1)
    core_number = torch.tensor(list(nx.core_number(G).values()), dtype=torch.float).unsqueeze(1)

    features = torch.cat([degree, clustering, eig_centrality, betweenness, closeness, pagerank, core_number], dim=1)
    return features
def min_max_normalize(coords):
    
    min_vals = coords.min(dim=0, keepdim=True)[0]
    max_vals = coords.max(dim=0, keepdim=True)[0]
    return (coords - min_vals) / (max_vals - min_vals + 1e-8)


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels, out_channels):
        super().__init__()

        # Edge network for conv1
        self.edge_net1 = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_channels * in_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels * in_channels, hidden_channels * in_channels)
        )

        self.conv1 = NNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            nn=self.edge_net1,
            aggr='mean'
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Edge network for conv2
        self.edge_net2 = nn.Sequential(
            nn.Linear(edge_attr_dim, (hidden_channels // 2) * hidden_channels),
            nn.ReLU(),
            nn.Linear((hidden_channels // 2) * hidden_channels, (hidden_channels // 2) * hidden_channels)
        )

        self.conv2 = NNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // 2,
            nn=self.edge_net2,
            aggr='mean'
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)

         # Edge network for conv3
        self.edge_net3 = nn.Sequential(
            nn.Linear(edge_attr_dim, (out_channels) * (hidden_channels // 2)),
            nn.ReLU(),
            nn.Linear((out_channels) * (hidden_channels // 2), (out_channels) * (hidden_channels // 2))
        )

        self.conv3 = NNConv(
            in_channels=hidden_channels // 2,
            out_channels=out_channels,
            nn=self.edge_net3,
            aggr='mean'
        )
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.dropout(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = self.dropout(x)

        return x

def dataset_prepare(filenames):
    dataset = []
    results = []
    for filename in filenames:
        nodes, xylocation = build_nodes(extract_info_netlist(os.path.join(NESTLIST_FOLDER, filename + '.ext')))
        print(f"{filename}: {xylocation}")
        for idx, coord in enumerate(xylocation.numpy()):
            results.append({
                'graph_id': filename,
                'node_index': idx,
                'x': coord[0],
                'y': coord[1],
            })
        # xylocation = min_max_normalize(xylocation)
        edge_list = edges_info(extract_info_netlist(os.path.join(NESTLIST_FOLDER, filename + '.ext')))
        edge_index, edge_attr = build_edges(edge_list)
        structural_features = generate_structural_features(edge_index)
        nodes = torch.cat([nodes, structural_features], dim=1)
        data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=xylocation)
        data = data.to(device)
        data.name = filename  # Store the filename in the data object
        dataset.append(data)
    df = pd.DataFrame(results)
    df.to_csv('./predicted_locations/original_coordinates.csv', index=False)
    print("Saved predictions to ./predicted_locations/original_coordinates.csv")
    return dataset

def denormalize(normalized_coords, min_vals, max_vals):
    """
    normalized_coords: Tensor of shape [N, 2] (normalized xy predictions)
    min_vals: Tensor of shape [1, 2] (min x and y used for normalization)
    max_vals: Tensor of shape [1, 2] (max x and y used for normalization)
    
    Returns:
    Denormalized coordinates in original scale
    """
   
    return normalized_coords * (max_vals - min_vals + 1e-8) + min_vals

dataset_filenames=[]
for filename in tqdm(os.listdir("./test_ext"), desc="reading nestlist files"):
    if filename.endswith('.ext'):
        dataset_filenames.append(filename.replace('.ext', ''))  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNEncoder(
    in_channels=25,
    edge_attr_dim=1,
    hidden_channels=64,
    out_channels=2
).to(device)
model.load_state_dict(torch.load('model_weights.pth'))
test_dataset = dataset_prepare(dataset_filenames)

model.eval()
os.makedirs('predicted_locations', exist_ok=True)

results = []
for data in tqdm(test_dataset, desc="Predicting locations"):
    data = data.to(device)
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr)
        predicted_coords = output[:, -2:]  # Last two columns are the x and y coordinates
        # min_vals = data.x[:, -2:].min(dim=0, keepdim=True)[0]
        # max_vals = data.x[:, -2:].max(dim=0, keepdim=True)[0]   
        # denormalized_coords = denormalize(predicted_coords, min_vals, max_vals)
        loss = F.mse_loss(predicted_coords, data.y)
        print(f"Predicted coordinates: {predicted_coords.cpu().numpy()}")
        for idx, coord in enumerate(predicted_coords.cpu().numpy()):
            results.append({
                'graph_id': data.name if hasattr(data, 'name') else 'unknown',
                'node_index': idx,
                'x': coord[0],
                'y': coord[1],
                'loss': loss.item()
            })

df = pd.DataFrame(results)
df.to_csv('./predicted_locations/predicted_coordinates.csv', index=False)
print("Saved predictions to ./predicted_locations/predicted_coordinates.csv")
