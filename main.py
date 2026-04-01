import copy
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.segmentation import slic
from skimage import graph
from scipy.ndimage import mean as ndimage_mean
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.loader import DataLoader


def encode_with_superpixels(img, n_segments, compactness):
    """Convert an image tensor into a superpixel graph.

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W).
        n_segments (int): Number of superpixels.
        compactness (float): SLIC compactness parameter.

    Returns:
        x (torch.Tensor): Node features (num_nodes, 5).
        edge_index (torch.Tensor): Edge indices (2, num_edges).
        edge_attr (torch.Tensor): Edge weights (num_edges, 1).
    """

    # Convert tensor (C,H,W) -> numpy (H,W,C)
    img_np = img.permute(1, 2, 0).numpy()
    h, w = img_np.shape[:2]

    # Generate superpixels using SLIC
    segments = slic(img_np, n_segments=n_segments, compactness=compactness, start_label=0)
    num_nodes = segments.max() + 1
    index = np.arange(num_nodes)

    # Initialize node feature matrix: [R, G, B, row, col]
    x = np.zeros((num_nodes, 5), dtype=np.float32)

    # Compute mean RGB per superpixel
    for c in range(3):
        x[:, c] = ndimage_mean(img_np[:, :, c], labels=segments, index=index)

    # Create coordinate grids
    row_grid, col_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Compute normalized centroid positions
    x[:, 3] = ndimage_mean(row_grid, labels=segments, index=index) / h
    x[:, 4] = ndimage_mean(col_grid, labels=segments, index=index) / w

    # Build region adjacency graph (RAG)
    rag = graph.rag_mean_color(img_np, segments)

    edges, edge_weights = [], []

    # Extract edges and weights (undirected)
    for i, j, data in rag.edges(data=True):
        weight = data.get("weight", 1.0)
        edges.extend([[i, j], [j, i]])
        edge_weights.extend([weight, weight])

    # Handle case with no edges
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    x = torch.tensor(x, dtype=torch.float)

    return x, edge_index, edge_attr


def build_superpixel_dataset(root="./data", n_segments=None, compactness=None):
    """Load CIFAR-10 and convert images into superpixel graphs.

    Args:
        root (str): Dataset directory.
        n_segments (int): Number of superpixels.
        compactness (float): SLIC compactness parameter.

    Returns:
        train_graphs (list[Data]): Training graph dataset.
        test_graphs (list[Data]): Test graph dataset.
    """

    transform = transforms.ToTensor()

    # Download/load CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    def build(dataset, split):
        """Convert dataset split into list of graph objects.

        Args:
            dataset: Torchvision dataset.
            split (str): Split name.

        Returns:
            list[Data]: List of graph objects.
        """
        graph_list = []

        for img, label in tqdm(dataset, desc=f"Encoding {split}", unit="img"):
            # Encode image into graph
            x, edge_index, edge_attr = encode_with_superpixels(img, n_segments, compactness)

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.long)
            )

            graph_list.append(data)

        return graph_list

    train_graphs = build(trainset, "train")
    test_graphs = build(testset, "test")

    return train_graphs, test_graphs


class SuperpixelGCN(nn.Module):
    """GCN model for classifying superpixel graphs.

    Args:
        in_channels (int): Input node feature size.
        edge_dim (int): Edge feature size.
        hidden_dim (int): Hidden layer size.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_channels=5, edge_dim=1, hidden_dim=64, num_classes=10, dropout=0.5):
        super().__init__()

        # Edge encoders map edge features to match node feature dimensions
        self.edge_encoder1 = Linear(edge_dim, in_channels)
        self.edge_encoder2 = Linear(edge_dim, hidden_dim)
        self.edge_encoder3 = Linear(edge_dim, hidden_dim)

        # Graph convolution layers
        self.conv1 = GINEConv(Sequential(Linear(in_channels, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))
        self.conv2 = GINEConv(Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))
        self.conv3 = GINEConv(Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))

        self.dropout = nn.Dropout(dropout)

        # Final classifier (after pooling)
        self.classifier = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass on batched graph.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            edge_attr (Tensor): Edge features.
            batch (Tensor): Batch assignment vector.

        Returns:
            Tensor: Class logits per graph.
        """

        # Apply 3 GNN layers with edge encoding and dropout
        x = self.dropout(self.conv1(x, edge_index, self.edge_encoder1(edge_attr)))
        x = self.dropout(self.conv2(x, edge_index, self.edge_encoder2(edge_attr)))
        x = self.dropout(self.conv3(x, edge_index, self.edge_encoder3(edge_attr)))

        # Global pooling (mean + max)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        return self.classifier(x)


def train(model, train_loader, test_loader, epochs=100, lr=0.001, weight_decay=1e-4, device="cuda"):
    """Train model and return best-performing weights.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization.
        device (str): "cuda" or "cpu".

    Returns:
        nn.Module: Trained model with best weights.
    """

    model = model.to(device)

    # Optimizer, scheduler, and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):

        # ---- Training phase ----
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward + loss
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- Evaluation phase ----
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                preds = out.argmax(dim=1)

                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)

        acc = correct / total
        scheduler.step(acc)

        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

    # Restore best weights
    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")

    return model


def main():
    """Run full pipeline: data prep, training, and saving.

    Returns:
        None
    """

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build datasets
    print("Building superpixel datasets...")
    train_graphs, test_graphs = build_superpixel_dataset(n_segments=50, compactness=10)

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    print(f"Dataset ready — {len(train_graphs)} train, {len(test_graphs)} test")

    # Initialize model
    model = SuperpixelGCN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    trained_model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=50,
        lr=0.002,
        weight_decay=1e-4,
        device=device,
    )

    # Save best model
    save_path = "superpixel_gcn_best.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Saved to '{save_path}'")


if __name__ == "__main__":
    main()