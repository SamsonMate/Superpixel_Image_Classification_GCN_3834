import copy
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.segmentation import slic
from skimage import graph
from skimage.color import rgb2hsv
from scipy.ndimage import mean as ndimage_mean, variance as ndimage_variance
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.loader import DataLoader


def encode_with_superpixels(img, n_segments, compactness):
    """Convert an image tensor into a superpixel graph with enriched features.

    Node features (12-dimensional):
        - Mean RGB (3)
        - Normalized centroid row, col (2)
        - RGB color variance per channel (3)
        - Normalized superpixel area (1)
        - Mean HSV (3)

    Edge features (2-dimensional):
        - RAG mean color difference weight (1)
        - Normalized Euclidean distance between centroids (1)

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W).
        n_segments (int): Target number of superpixels.
        compactness (float): SLIC compactness parameter.

    Returns:
        x (torch.Tensor): Node features (num_nodes, 12).
        edge_index (torch.Tensor): Edge indices (2, num_edges).
        edge_attr (torch.Tensor): Edge attributes (num_edges, 2).
    """

    img_np = img.permute(1, 2, 0).numpy()
    h, w = img_np.shape[:2]

    segments = slic(img_np, n_segments=n_segments, compactness=compactness, start_label=0)
    num_nodes = segments.max() + 1
    index = np.arange(num_nodes)

    # Initialize node feature matrix: 12 features per node
    x = np.zeros((num_nodes, 12), dtype=np.float32)

    # Mean RGB (features 0-2)
    for c in range(3):
        x[:, c] = ndimage_mean(img_np[:, :, c], labels=segments, index=index)

    # Coordinate grids for centroid computation
    row_grid, col_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Normalized centroid positions (features 3-4)
    centroid_row = ndimage_mean(row_grid, labels=segments, index=index)
    centroid_col = ndimage_mean(col_grid, labels=segments, index=index)
    x[:, 3] = centroid_row / h
    x[:, 4] = centroid_col / w

    # RGB color variance per channel (features 5-7)
    for c in range(3):
        x[:, 5 + c] = ndimage_variance(img_np[:, :, c], labels=segments, index=index)

    # Normalized superpixel area as fraction of total pixels (feature 8)
    area = np.bincount(segments.ravel(), minlength=num_nodes).astype(np.float32)
    x[:, 8] = area / (h * w)

    # Mean HSV values (features 9-11)
    img_hsv = rgb2hsv(img_np)
    for c in range(3):
        x[:, 9 + c] = ndimage_mean(img_hsv[:, :, c], labels=segments, index=index)

    # Build region adjacency graph
    rag = graph.rag_mean_color(img_np, segments)

    edges, color_weights, dist_weights = [], [], []

    for i, j, data in rag.edges(data=True):
        color_weight = data.get("weight", 1.0)

        # Euclidean distance between normalized centroids (features already in [0,1])
        dist = np.sqrt(
            (centroid_row[i] / h - centroid_row[j] / h) ** 2 +
            (centroid_col[i] / w - centroid_col[j] / w) ** 2
        )

        edges.extend([[i, j], [j, i]])
        color_weights.extend([color_weight, color_weight])
        dist_weights.extend([dist, dist])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(
            np.column_stack([color_weights, dist_weights]), dtype=torch.float
        )

    x = torch.tensor(x, dtype=torch.float)

    return x, edge_index, edge_attr


def build_superpixel_dataset(dataset_class, root="./data", n_segments=50, compactness=10):
    """Load a torchvision dataset and convert images into superpixel graphs.

    Hyperparameter tuning note:
        n_segments and compactness significantly affect graph structure and
        model performance. Recommended sweep ranges:
            n_segments:  [25, 50, 75, 100]
            compactness: [5, 10, 20, 30]

    Args:
        dataset_class: Torchvision dataset class to load (e.g. torchvision.datasets.CIFAR10).
        root (str): Dataset directory.
        n_segments (int): Target number of superpixels per image.
        compactness (float): SLIC compactness — higher values produce more
            spatially regular superpixels at the cost of color homogeneity.

    Returns:
        train_graphs (list[Data]): Training graph dataset.
        test_graphs (list[Data]): Test graph dataset.
    """

    transform = transforms.ToTensor()

    trainset = dataset_class(root=root, train=True,  download=True, transform=transform)
    testset  = dataset_class(root=root, train=False, download=True, transform=transform)

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
            x, edge_index, edge_attr = encode_with_superpixels(img, n_segments, compactness)

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
    """GINE-based GNN with residual connections for classifying superpixel graphs.

    Args:
        in_channels (int): Input node feature size (12 with enriched features).
        edge_dim (int): Edge feature size (2 with centroid distance added).
        hidden_dim (int): Hidden layer size.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_channels=12, edge_dim=2, hidden_dim=64, num_classes=10, dropout=0.5):
        super().__init__()

        # Edge encoders map edge features to match node feature dimensions
        self.edge_encoder1 = Linear(edge_dim, in_channels)
        self.edge_encoder2 = Linear(edge_dim, hidden_dim)
        self.edge_encoder3 = Linear(edge_dim, hidden_dim)

        # Graph convolution layers
        self.conv1 = GINEConv(Sequential(Linear(in_channels, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))
        self.conv2 = GINEConv(Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))
        self.conv3 = GINEConv(Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))

        # Residual projection for layer 1 — required since in_channels != hidden_dim
        self.residual_proj = Linear(in_channels, hidden_dim)

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

        # Layer 1: project residual to hidden_dim before adding (in_channels != hidden_dim)
        x = self.dropout(self.conv1(x, edge_index, self.edge_encoder1(edge_attr)) + self.residual_proj(x))

        # Layers 2-3: direct residual connections (dimensions match)
        x = self.dropout(self.conv2(x, edge_index, self.edge_encoder2(edge_attr)) + x)
        x = self.dropout(self.conv3(x, edge_index, self.edge_encoder3(edge_attr)) + x)

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

            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)

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

        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")

    return model


def main():
    """Run full pipeline: data prep, training, and saving.

    Returns:
        None
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =============================================================================
    # HYPERPARAMETERS
    # =============================================================================

    # -- Dataset --
    # Swap dataset_class to use a different torchvision dataset (e.g. torchvision.datasets.CIFAR100)
    dataset_class = torchvision.datasets.CIFAR10
    data_root     = "./data" # This is where the data will be stored/downloaded

    # -- SLIC (graph encoding) --
    # Recommended sweep: n_segments in [25, 50, 75, 100], compactness in [5, 10, 20, 30]
    n_segments  = 50
    compactness = 10

    # -- Model architecture --
    hidden_dim  = 64
    dropout     = 0.5

    # -- Training --
    batch_size   = 64
    epochs       = 50
    lr           = 0.002
    weight_decay = 1e-4

    # =============================================================================

    print("Building superpixel datasets...")
    train_graphs, test_graphs = build_superpixel_dataset(
        dataset_class=dataset_class,
        root=data_root,
        n_segments=n_segments,
        compactness=compactness
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    print(f"Dataset ready — {len(train_graphs)} train, {len(test_graphs)} test")

    model = SuperpixelGCN(hidden_dim=hidden_dim, dropout=dropout)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trained_model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    save_path = "superpixel_gcn_best.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Saved to '{save_path}'")


if __name__ == "__main__":
    main()