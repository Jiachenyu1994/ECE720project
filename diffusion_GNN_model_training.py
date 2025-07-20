import re
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch_geometric.nn import NNConv

import os
import random
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader as GeoDataLoader
from PIL import Image
from torch.utils.data import Dataset
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch.optim as optim
from torchvision.utils import save_image

NESTLIST_FOLDER = './ext_output'  # Path to the folder containing nestlist files
CLEAN_IMAGES_FOLDER = './smaller_layout'  # Path to save clean images



class GNNEncoder(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels, out_channels, pooling='mean'):
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

        self.dropout = nn.Dropout(p=0.3)

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

        # # Pooling method
        # if pooling == 'mean':
        #     self.pool = global_mean_pool
        # elif pooling == 'max':
        #     from torch_geometric.nn import global_max_pool
        #     self.pool = global_max_pool
        # elif pooling == 'add':
        #     from torch_geometric.nn import global_add_pool
        #     self.pool = global_add_pool
        # else:
        #     raise ValueError(f"Unsupported pooling method: {pooling}")

        # self.fc = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = self.dropout(x)

        # graph_embedding = self.pool(x, batch)
        # return self.fc(graph_embedding)
        return x  # Return node embeddings instead of graph embedding

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
    for t in transistors:
        width = t["width"]
        length = t["length"]
        x_location = t["x_location"]
        y_location = t["y_location"]
        model_vec = [0] * len(model_map)
        model_vec[model_map[t["model"]]] = 1
        node_feature = [width, length,x_location,y_location] + model_vec
        nodes.append(node_feature)

    x = torch.tensor(nodes, dtype=torch.float)
    return x

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

    
def make_graphs_from_filenames(filenames):
    train_graphs = []

    for filename in filenames:
        spice_file = os.path.join(NESTLIST_FOLDER, f'{filename}.ext')
        transistors = extract_info_netlist(spice_file)
        edges = edges_info(transistors)
        nodes = build_nodes(transistors)
        edge_index, edge_attr = build_edges(edges)
        data= Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)  # Dummy target
        train_graphs.append(data)
    return train_graphs



class LayoutGraphDataset(Dataset):
    def __init__(self, filenames, image_folder, netlist_folder, gnn_encoder, device='cpu'):
        self.filenames = filenames
        self.image_folder = image_folder
        self.netlist_folder = netlist_folder
        self.gnn_encoder = gnn_encoder
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load clean image
        img_path = os.path.join(self.image_folder, f'{filename}.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # shape [3, 2000, 800]

        # Process corresponding netlist
        spice_file = os.path.join(self.netlist_folder, f'{filename}.ext')
        transistors = extract_info_netlist(spice_file)
        edges = edges_info(transistors)
        nodes = build_nodes(transistors).to(self.device)
        edge_index, edge_attr = build_edges(edges)
        batch = torch.zeros(nodes.size(0), dtype=torch.long).to(self.device)

        # Get GNN embedding
        gnn_embedding = self.gnn_encoder(nodes, edge_index.to(self.device), edge_attr.to(self.device), batch)
        return image, gnn_embedding
    


def save_image_tensor(tensor, filename):
    # Tensor shape: (B, C, H, W)

        save_image(tensor.cpu().clamp(0, 1), os.path.join("./validation_3_epochs/", filename))

def validate(val_loader, diffusion_model, gnn_encoder, noise_scheduler, device):
    diffusion_model.eval()
    gnn_encoder.eval()

    with torch.no_grad():
        for idx, (images, gnn_embeddings) in enumerate(val_loader):
            images = images.to(device)
            gnn_embeddings = gnn_embeddings.to(device)

            # Sample random timestep
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (images.size(0),),
                device=device
            )

            # Add noise
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Predict noise
            noise_pred = diffusion_model(
                sample=noisy_images,
                timestep=timesteps,
                encoder_hidden_states=gnn_embeddings
            ).sample

            # Reconstruct images by denoising (single-step approximation)
            reconstructed = noisy_images - noise_pred

            # Save images
            save_image_tensor(noisy_images, f'val_noisy_{idx}.png')
            save_image_tensor(reconstructed, f'val_reconstructed_{idx}.png')
            save_image_tensor(images, f'val_groundtruth_{idx}.png')

            print(f"Saved validation sample {idx}")










# dataset preparation
dataset_filenames=[]
for filename in tqdm(os.listdir(NESTLIST_FOLDER), desc="reading nestlist files"):
    if filename.endswith('.ext'):
        dataset_filenames.append(filename.replace('.ext', ''))

random.shuffle(dataset_filenames)
print(f"Total dataset size: {len(dataset_filenames)}")

train_filenames = dataset_filenames[:int(len(dataset_filenames)*0.8)]
val_filenames = dataset_filenames[int(len(dataset_filenames)*0.8):int(len(dataset_filenames))]




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device ="cpu"

gnn_encoder = GNNEncoder(
    in_channels=20,
    edge_attr_dim=1,
    hidden_channels=64,
    out_channels=32
).to(device)

dataset = LayoutGraphDataset(train_filenames, CLEAN_IMAGES_FOLDER, NESTLIST_FOLDER, gnn_encoder, device=device)
train_loader = GeoDataLoader(dataset, batch_size=1, shuffle=True)


# Define diffusion model
diffusion_model = UNet2DConditionModel(
    sample_size=(160, 400),  # image resolution
    in_channels=3,           # RGB image
    out_channels=3,          # RGB image output
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    cross_attention_dim=32,  # match GNN embedding size
    attention_head_dim=8
).to(device)

# Define noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)

optimizer = optim.AdamW(
    list(diffusion_model.parameters()) + list(gnn_encoder.parameters()),
    lr=1e-4,      # learning rate
    weight_decay=1e-4  # optional, prevents overfitting
)
# print(f"Diffusion Model Sample Size: {diffusion_model.config.sample_size}")

training_losses = []

gnn_encoder.train()
diffusion_model.train()
# Forward step: predict noise residual
for step in tqdm(range(2), desc="Training Epochs"):  # 2 epochs
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training Loop")
    for step, (images, gnn_embeddings) in enumerate(progress_bar):
        
        # print(f"gnn_embeddings shape: {gnn_embeddings.shape}")
        # print(f"images shape: {images.shape}")

        images = images.to(device)            # shape: [B, 3, 2000, 800]
        gnn_embeddings = gnn_embeddings.to(device)  # shape: [B, 32]

        # Sample random timesteps for each sample in the batch
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (images.size(0),),  # batch size
            device=device
        )
        # print(f"Timesteps: {timesteps}")

        # Generate random noise for each image
        noise = torch.randn(images.shape, device=images.device)

        # Add noise to the clean images at the chosen timesteps
        noisy_images = noise_scheduler.add_noise(
            images,
            noise,
            timesteps
        )

        #  Predict the noise residual
        # print("Shapes after noise addition")
        # print(f"noisy_images: {noisy_images.shape}, noise: {noise.shape}, timesteps: {timesteps.shape}")

    
        noise_pred = diffusion_model(
            sample=noisy_images,
            timestep=timesteps,
            encoder_hidden_states=gnn_embeddings
        ).sample
        # print(f"pass")
        #  Compute loss between predicted noise and true noise
        loss = F.mse_loss(noise_pred, noise)

        #  Backpropagation step
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix({'train_loss': loss.item()})
    training_losses.append(total_loss / len(train_loader))

torch.save({
    'gnn_encoder_state_dict': gnn_encoder.state_dict(),
    'diffusion_model_state_dict': diffusion_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # Optional
}, 'gnn_diffusion_model.pth')

print("Model saved to gnn_diffusion_model.pth")

os.makedirs('./validation_3_epochs', exist_ok=True)

validate(
    val_loader=GeoDataLoader(LayoutGraphDataset(val_filenames, CLEAN_IMAGES_FOLDER, NESTLIST_FOLDER, gnn_encoder, device=device), batch_size=1),
    diffusion_model=diffusion_model,
    gnn_encoder=gnn_encoder,
    noise_scheduler=noise_scheduler,
    device=device
)

plt.figure(figsize=(8,4))
plt.plot(training_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('diffusion_training_loss_curve_3_epochs.png')