import re
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import tqdm
import gdsfactory as gf



# check if pytorch and cuda avaliable
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x  # [num_nodes, out_channels] = embedding per transistor


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
    nodes = []
    for t in transistors:
        width = t["width"]
        length = t["length"]
        model_vec = [0] * 8
        model_vec[model_map[t["model"]]] = 1
        node_feature = [width, length] + model_vec
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


def group_transistors(embedding, k):
    """Cluster transistor embeddings into k groups."""
    embedding_np = embedding.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(embedding_np)
    return labels  # [num_nodes] cluster ID per transistor


def simulate_layout_area(transistors, labels):
    """
    Replace this function with actual layout using GDSFactory
    based on groupings, and return the bounding box area.
    """
    # For now, simulate area as the spread of each group
    group_spread = 0
    for group_id in set(labels):
        group = [t for t, lbl in zip(transistors, labels) if lbl == group_id]
        widths = [t['width'] for t in group]
        group_spread += max(widths) * len(group)  # fake "diffusion row length"
    return torch.tensor(group_spread, dtype=torch.float)


def train_GNN(data):
    model = GNN(10, 32, 8)  # 10 input features → 8-D embedding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # Step 1: Get embeddings
        embedding = model(data.x, data.edge_index, data.edge_attr)

        # Step 2: Grouping
        k = 8  # or some dynamic estimate
        labels = group_transistors(embedding, k)

        # Step 3: Layout simulation (you’ll replace this later)
        layout_area = simulate_layout_area(transistors, labels)

        # Step 4: Loss = area
        loss = layout_area
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Area Loss: {loss.item():.2f}")



def assign_dummy_labels(transistors):
    # Fake GNN grouping: group every 4 transistors together
    for i, t in enumerate(transistors):
        t["group"] = i // 4
    return transistors

def layout_from_groupings(transistors):
    layout = gf.Component("gnn_layout")

    # Group by label
    group_dict = {}
    for t in transistors:
        group_dict.setdefault(t["group"], []).append(t)

    x_offset = 0
    for group_id, group in sorted(group_dict.items()):
        y_offset = 0
        for t in group:
            if "pfet" in t["model"]:
                cell = pdk.cells["sky130_fd_pr__nfet_01v8"](w=t["w"], l=t["l"], nf=t["nf"])
            else:
                cell = pdk.cells["sky130_fd_pr__nfet_01v8"](w=t["w"], l=t["l"], nf=t["nf"])
            ref = layout.add_ref(cell)
            ref.move((x_offset, y_offset))
            y_offset += 40  # space between transistors
        x_offset += 100  # space between groups

    return layout


transistors=extract_info_spice("sky130_fd_sc_hd__fa_1.spice")
edge_list=edges_info(transistors)
print('\n'.join(f"{transistor}" for transistor in transistors  ))
print('\n'.join(f'{edge}' for edge in edges_info(transistors)))
edge_index, edge_attr = build_edges(edge_list)
data = Data(
    x=build_nodes(transistors),                     
    edge_index=edge_index,   
    edge_attr=edge_attr  

)
print(data)

#having issues....
# train_GNN(data)
# print(list(pdk.cells.keys()))

# spice_file = "sky130_fd_sc_hd__fa_1.spice"
# transistors = assign_dummy_labels(transistors)  # replace with GNN output

# layout = layout_from_groupings(transistors)
# layout.show()
# layout.write_gds("gnn_layout.gds")

