"""
Please view README.md for details
"""
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

#
# Import our models dataset and segment it accordingly
#

def encode_with_superpixels(img, n_segments=50, compactness=10):
    """Encodes a single image as a graph using SLIC superpixel segmentation.

Converts a tensor image into a graph representation where each node
corresponds to a superpixel. Node features consist of mean RGB color
and normalized centroid coordinates. Edges are derived from a Region
Adjacency Graph (RAG) built on mean color, with edge weights reflecting
color similarity between adjacent superpixels.

Args:
    img (torch.Tensor): Input image tensor of shape (C, H, W) with values
        in [0, 1], as returned by torchvision.transforms.ToTensor().
    n_segments (int, optional): Target number of superpixels for SLIC
        segmentation. The actual number may vary slightly. Defaults to 50.
    compactness (int, optional): Balances color proximity and space
        proximity in SLIC. Higher values give more square superpixels,
        lower values follow image boundaries more closely. Defaults to 10.

Returns:
    x (torch.Tensor): Node feature matrix of shape (num_nodes, 5), where
        each row contains [R, G, B, norm_row, norm_col] for a superpixel.
        RGB values are in [0, 1] and centroid coordinates are normalized
        by image height.
    edge_index (torch.Tensor): Graph connectivity in COO format of shape
        (2, num_edges). Edges are undirected, so each pair (i, j) appears
        as both (i, j) and (j, i). Shape is (2, 0) if no edges are found.
    edge_attr (torch.Tensor): Edge weight matrix of shape (num_edges, 1),
        where each value is the RAG color similarity weight between two
        adjacent superpixels. Shape is (0, 1) if no edges are found.
"""
    img_np = img.permute(1, 2, 0).numpy()  # (H,W,C)
    h = img_np.shape[0]
    w = img_np.shape[1]

    segments = slic(img_np, n_segments=n_segments, compactness=compactness, start_label=0)
    num_nodes = segments.max() + 1
    index = np.arange(num_nodes)

    # --- node features (vectorized) ---
    x = np.zeros((num_nodes, 5), dtype=np.float32)

    for c in range(3):
        x[:, c] = ndimage_mean(img_np[:, :, c], labels=segments, index=index)

    row_grid, col_grid = np.meshgrid(
        np.arange(img_np.shape[0]),
        np.arange(img_np.shape[1]),
        indexing='ij'
    )

    x[:, 3] = ndimage_mean(row_grid, labels=segments, index=index) / h
    x[:, 4] = ndimage_mean(col_grid, labels=segments, index=index) / w

    # --- adjacency + edge weights from RAG ---
    rag = graph.rag_mean_color(img_np, segments)

    edges = []
    edge_weights = []

    for i, j, data in rag.edges(data=True):
        weight = data.get("weight", 1.0)
        edges.append([i, j])
        edges.append([j, i])
        edge_weights.extend([weight, weight])

    # Guard against empty edge list
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    x = torch.tensor(x, dtype=torch.float)

    return x, edge_index, edge_attr

def build_superpixel_dataset(root="./data", n_segments=50, compactness=10):
    """Downloads CIFAR-10 and builds superpixel graph datasets for training and testing.

Iterates over the CIFAR-10 train and test splits, encoding each image as
a graph using encode_with_superpixels(). Each graph contains superpixel
node features, a RAG-based edge structure with color similarity weights,
and the original class label. Returns two lists of PyTorch Geometric Data
objects ready for use with a graph DataLoader.

Args:
    root (str, optional): Directory where CIFAR-10 will be downloaded and
        cached. Created automatically if it does not exist. Defaults to "./data".
    n_segments (int, optional): Target number of superpixels per image,
        passed directly to encode_with_superpixels(). Defaults to 50.
    compactness (int, optional): SLIC compactness parameter controlling the
        trade-off between color and spatial proximity in segmentation,
        passed directly to encode_with_superpixels(). Defaults to 10.

Returns:
    train_graphs (list of torch_geometric.data.Data): Graph representations
        of the 50,000 CIFAR-10 training images. Each Data object contains:
        - x (torch.Tensor): Node features of shape (num_nodes, 5).
        - edge_index (torch.Tensor): Edge connectivity of shape (2, num_edges).
        - edge_attr (torch.Tensor): Edge weights of shape (num_edges, 1).
        - y (torch.Tensor): Scalar class label in range [0, 9].
    test_graphs (list of torch_geometric.data.Data): Graph representations
        of the 10,000 CIFAR-10 test images, in the same format as train_graphs.
"""
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    def build(dataset, split):
        graph_list = []

        for img, label in tqdm(dataset, desc=f"Encoding {split}", unit="img"):
            x, edge_index, edge_attr = encode_with_superpixels(
                img, n_segments=n_segments, compactness=compactness
            )

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
    """Graph Convolutional Network for image classification using superpixel graphs.

A Graph Convolutional Network (GCN) is a neural network that operates directly
on graph-structured data. Unlike standard CNNs that process regular pixel grids,
a GCN learns by passing messages between connected nodes, allowing each node to
accumulate information from its local neighborhood. Stacking multiple convolutional
layers allows information to propagate further across the graph, enabling the
network to capture increasingly global structure with each layer.

Edge encoders are small learnable networks that project edge attributes into the
same vector space as node features. This is necessary because message passing
operations combine node and edge representations mathematically, so they must
share the same dimensionality. Without edge encoders, raw edge attributes such
as color similarity weights could not be incorporated into the message passing
process and would be discarded entirely.

After message passing, each node holds an embedding that summarizes its local
graph neighborhood. Since graphs vary in size, a pooling operation is used to
collapse all node embeddings into a single fixed-size vector that represents the
entire graph. This graph-level vector is then passed to a classifier, which is
a fully connected network that maps the learned graph representation to a
probability distribution over the target classes, producing a final prediction.

Args:
    in_channels (int, optional): Number of input node features per superpixel.
        Corresponds to the feature size produced by encode_with_superpixels(),
        which outputs [R, G, B, norm_row, norm_col]. Defaults to 5.
    edge_dim (int, optional): Dimensionality of input edge attributes. Corresponds
        to the single RAG color similarity weight produced by
        encode_with_superpixels(). Defaults to 1.
    hidden_dim (int, optional): Number of hidden units in each GINEConv layer.
        Also determines the size of the first classifier layer. Defaults to 64.
    num_classes (int, optional): Number of output classes for classification.
        Set to 10 for CIFAR-10. Defaults to 10.
    dropout (float, optional): Dropout probability applied after each GINEConv
        layer and within the classifier head. Defaults to 0.5.
"""
    def __init__(self, in_channels=5, edge_dim=1, hidden_dim=64, num_classes=10, dropout=0.5):
        super().__init__()

        # Edge encoders — GINEConv requires edge_attr to match node feature dim at each layer
        self.edge_encoder1 = Linear(edge_dim, in_channels)   # 1 -> 5
        self.edge_encoder2 = Linear(edge_dim, hidden_dim)    # 1 -> 64
        self.edge_encoder3 = Linear(edge_dim, hidden_dim)    # 1 -> 64

        # GINEConv layers — BN and ReLU are inside each layer's nn
        self.conv1 = GINEConv(
            Sequential(Linear(in_channels, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        )
        self.conv2 = GINEConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        )
        self.conv3 = GINEConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        )

        self.dropout = nn.Dropout(dropout)

        # Classifier — mean+max concatenated doubles the input size
        self.classifier = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Layer 1
        x = self.conv1(x, edge_index, self.edge_encoder1(edge_attr))
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index, self.edge_encoder2(edge_attr))
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index, self.edge_encoder3(edge_attr))
        x = self.dropout(x)

        # Global pooling — mean + max concatenated into one vector per graph
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        return self.classifier(x)

def train(model, train_loader, test_loader, epochs=100, lr=0.001, weight_decay=1e-4, device="cuda"):
    """Trains a SuperpixelGCN model on superpixel graph data and evaluates it on a test set.

In graph-based learning, a training loop must handle batched graphs rather than
batched tensors. Each batch produced by a PyTorch Geometric DataLoader is a single
large disconnected graph containing multiple images, where a batch vector tracks
which nodes belong to which image. The model processes this batched graph in one
forward pass, making training efficient regardless of varying graph sizes across
images.

L2 regularization is applied through weight decay in the Adam optimizer, which
adds a penalty proportional to the magnitude of the model's weights to the loss
function. This discourages the model from relying too heavily on any individual
parameter and helps prevent overfitting, which is particularly relevant for graph
models trained on small or noisy node features.

A learning rate scheduler monitors validation accuracy after each epoch and reduces
the learning rate when improvement stalls. This allows the optimizer to take smaller,
more precise steps as training progresses, often recovering additional accuracy
that a fixed learning rate would miss.

Per-class accuracy is tracked separately for each of the 10 CIFAR-10 classes
throughout evaluation. This provides a more diagnostic view of model performance
than overall accuracy alone, making it easier to identify which classes the model
consistently confuses.

Args:
    model (torch.nn.Module): The GCN model to train. Expected to accept
        (x, edge_index, edge_attr, batch) as forward pass arguments.
    train_loader (DataLoader): PyTorch Geometric DataLoader wrapping the
        training graph dataset.
    test_loader (DataLoader): PyTorch Geometric DataLoader wrapping the
        test graph dataset.
    epochs (int, optional): Number of full passes over the training dataset.
        Defaults to 100.
    lr (float, optional): Initial learning rate for the Adam optimizer.
        Defaults to 0.001.
    weight_decay (float, optional): L2 regularization coefficient applied
        through Adam's weight decay parameter. Defaults to 1e-4.
    device (str, optional): Device to run training on. Accepts "cuda" or
        "cpu". Defaults to "cuda".

Returns:
    model (torch.nn.Module): The trained model restored to the checkpoint
        with the highest validation accuracy achieved during training.
"""
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)
    criterion = torch.nn.CrossEntropyLoss()

    CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"]

    best_acc = 0.0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):

        # --- Training ---
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

        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        class_correct = torch.zeros(10)
        class_total = torch.zeros(10)

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                preds = out.argmax(dim=1)

                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)

                # Per-class accuracy tracking
                for c in range(10):
                    mask = batch.y == c
                    class_correct[c] += (preds[mask] == batch.y[mask]).sum().item()
                    class_total[c] += mask.sum().item()

        acc = correct / total
        scheduler.step(acc)

        # --- Checkpoint best model ---
        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

        # Per-class breakdown
        for c in range(10):
            class_acc = class_correct[c] / class_total[c]
            print(f"  {CIFAR10_CLASSES[c]:<12} {class_acc:.4f}")

    # Restore best model at the end
    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")

    return model

def main():
    # Select device — fall back to CPU if CUDA is unavailable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Import, segment, then build the dataset using superpixel image segmentation.
    print("Building superpixel datasets (this may take a while on first run)...")
    train_graphs, test_graphs = build_superpixel_dataset()
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)
    print(f"Dataset ready — {len(train_graphs)} train graphs, {len(test_graphs)} test graphs.")

    # Instantiate the model
    model = SuperpixelGCN(
        in_channels=5,
        edge_dim=1,
        hidden_dim=64,
        num_classes=10,
        dropout=0.5,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model for num_epochs
    num_epochs = 100
    trained_model = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=num_epochs,
        lr=0.001,
        weight_decay=1e-4,
        device=device,
    )

    # Save the best checkpoint
    save_path = "superpixel_gcn_best.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Best model weights saved to '{save_path}'.")


if __name__ == "__main__":
    main()