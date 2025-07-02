import re
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import tqdm
import gdsfactory as gf
import torch.nn as nn
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# check if pytorch and cuda avaliable
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())



def extract_info_spice(spice_file):
    transistors=[]
    with open(spice_file,'r') as f:
        spice_data=f.read()
        lines=spice_data.splitlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('X'):
                features=line.split(" ")
                transistor_info={
                    "name":features[0],
                    "source": features[1],
                    'gate': features[2],
                    'drain': features[3],
                    'bulk': features[4],
                    'model': features[5],
                    'width': float(features[6].split('=')[1].replace('u', ''))*1e-6,
                    'length': float(features[7].split('=')[1].replace('u', ''))*1e-6,
                }
                transistors.append(transistor_info)
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
                edge=(transistor_i["name"],transistor_j["name"],{"shared_nets":list(set(nets_i) & set(nets_j))})
            edges.append(edge)

    return edges

def build_nodes(transistors):
    model_map = {'sky130_fd_pr__nfet_01v8': 0, 
                'sky130_fd_pr__pfet_01v8': 1, 
                'sky130_fd_pr__pfet_01v8_hvt': 2, 
                'sky130_fd_pr__nfet_01v8_hvt': 3, 
                'sky130_fd_pr__pfet_01v8_lvt': 4, 
                'sky130_fd_pr__nfet_01v8_lvt': 5, 
                'sky130_fd_pr__nfet_g5v0d10v5': 6, 
                'sky130_fd_pr__pfet_g5v0d10v5': 7}
    special_nets = {
    "power": {"VPWR", "VGND", "VPB", "VNB"},
    "inputs": {"A", "B", "CIN"},
    "outputs": {"SUM", "COUT"},
    }   
    nodes = []
    def is_power(net):
        return net in special_nets["power"]
    def is_input(net):
        return net in special_nets["inputs"]
    def is_output(net):
        return net in special_nets["outputs"]
    for t in transistors:
        width = t["width"]
        length = t["length"]
        source_is_power = is_power(t["source"])
        gate_is_input = is_input(t["gate"])
        drain_is_output = is_output(t["drain"])
        logic_features = [int(source_is_power), int(gate_is_input), int(drain_is_output)]
        model_vec = [0] * 8
        model_vec[model_map[t["model"]]] = 1
        node_feature = [width, length] +logic_features+ model_vec
        nodes.append(node_feature)

    x = torch.tensor(nodes, dtype=torch.float)
    return x

def build_edges(edge_list):
    edge_index = []
    for src, dst, _ in edge_list:  # or edge_list_with_attrs
        i = int(src.replace("X",""))
        j = int(dst.replace("X",""))
        edge_index.append([i, j])
        edge_index.append([j, i])  # if undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    edge_attr = []
    for src, dst, attr in edge_list:
        shared_count = len(attr["shared_nets"])
        edge_attr.append([shared_count])
        edge_attr.append([shared_count])  # undirected pair

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index,edge_attr

def add_noise_to_node_features(x, noise_level=0.1,seed=42):
    torch.manual_seed(seed)
    noise = torch.randn_like(x) * noise_level
    x_noisy = x + noise
    return x_noisy

def perturb_edges(edge_index,edge_attr, drop_prob=0.1,seed=42):
    torch.manual_seed(seed)
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_prob
    
    return edge_index[:, keep_mask],edge_attr[keep_mask]

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
    
def denoising_loss(z_clean, z_noisy):
    return F.mse_loss(z_noisy, z_clean)

def contrastive_loss(z_clean, z_noisy, temperature=0.1):
    z_clean = F.normalize(z_clean, dim=1)
    z_noisy = F.normalize(z_noisy, dim=1)

    sim = torch.mm(z_clean, z_noisy.T) / temperature  # shape: [N, N]
    exp_sim = torch.exp(sim)

    pos_sim = torch.diag(exp_sim)  # numerator: exp(sim(h_i, h_i'))
    denom = exp_sim.sum(dim=1)     # denominator: sum over all exp(sim(h_i, h_j'))

    loss = -torch.log(pos_sim / denom)
    return loss.mean()

def loss_function(z_clean, z_noisy,lamda_denoise,lamda_contrastive,temperature=0.1):
    denoise_loss = denoising_loss(z_clean, z_noisy)
    contrastive = contrastive_loss(z_clean, z_noisy, temperature)
    return lamda_denoise*denoise_loss + lamda_contrastive*contrastive


def train_model(model,device, x, edge_index, edge_attr, optimizer, epochs=30000, noise_level=0.2,
                lamda_denoise=0.5, lamda_contrastive=1.0, temperature=0.1):
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    model.to(device)

    pbar = tqdm(range(epochs), desc="Training", dynamic_ncols=True)
    for itr in pbar:
        model.train()

        # Generate noisy view
        x_noisy = add_noise_to_node_features(x, noise_level=noise_level).to(device)
        edge_index_noisy, edge_attr_noisy = perturb_edges(edge_index, edge_attr)
        edge_index_noisy = edge_index_noisy.to(device)
        edge_attr_noisy = edge_attr_noisy.to(device)

        # Create clean and noisy data
        data_clean = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_noisy = Data(x=x_noisy, edge_index=edge_index_noisy, edge_attr=edge_attr_noisy)

        # Forward pass
        z_clean = model(data_clean.x, data_clean.edge_index)
        z_noisy = model(data_noisy.x, data_noisy.edge_index)
        # Compute loss
        optimizer.zero_grad()
        loss = loss_function(z_clean, z_noisy, lamda_denoise, lamda_contrastive, temperature)
        loss.backward()
        optimizer.step()
        # Print progress
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})


transistors=extract_info_spice("sky130_fd_sc_hd__fa_1.spice")
edge_list=edges_info(transistors)
# print('\n'.join(f"{transistor}" for transistor in transistors  ))
# print('\n'.join(f'{edge}' for edge in edges_info(transistors)))
edge_index, edge_attr = build_edges(edge_list)
x = build_nodes(transistors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= GNNEncoder(in_channels=x.shape[1], hidden_channels=128, out_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model,device, x, edge_index, edge_attr, optimizer, epochs=30000, noise_level=0.2,
            lamda_denoise=0.5, lamda_contrastive=1.0, temperature=0.1)

model.eval()
with torch.no_grad():
    z = model(x.to(device), edge_index.to(device))

num_groups = 4  # you can tune this
kmeans = KMeans(n_clusters=num_groups, random_state=0)
labels = kmeans.fit_predict(z.cpu().numpy())  # shape: [num_nodes]

for i, label in enumerate(labels):
    print(f"Transistor X{i} → Group {label}")

with open("transistor_groups.txt", "w") as f:
    for i, label in enumerate(labels):
        f.write(f"X{i} {label}\n")







