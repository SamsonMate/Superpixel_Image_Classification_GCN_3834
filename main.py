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
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_scatter import scatter
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
    """GCN-based model with MLP node encoder, edge feature augmentation,
    and residual connections for classifying superpixel graphs.

    Edge attributes are incorporated via pre-convolution node feature augmentation
    (mean and max aggregation of incident edge weights) since GCNConv only supports
    scalar edge weights natively.

    Args:
        in_channels (int): Input node feature size (12 with enriched features).
        hidden_dim (int): Hidden layer size.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_channels=12, hidden_dim=64, num_classes=10, dropout=0.5):
        super().__init__()

        # in_channels + 4: edge_attr has 2 columns, scatter produces 2 values per
        # reduction (mean and max), so 4 values are appended in total
        augmented_channels = in_channels + 4

        # MLP node encoder to project augmented features into hidden space
        # BatchNorm is intentionally omitted here as GCN layers already apply
        # BatchNorm after each convolution, avoiding redundant normalisation
        self.node_encoder = Sequential(
            Linear(augmented_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Post-convolution batch norm and activation
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)

        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)

        # Final classifier (after pooling)
        self.classifier = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )

    @staticmethod
    def augment_node_features(x, edge_index, edge_attr):
        """Aggregate incident edge attributes per node and append to node features.

        Args:
            x (Tensor): Node features (num_nodes, in_channels).
            edge_index (Tensor): Edge indices (2, num_edges).
            edge_attr (Tensor): Edge attributes (num_edges, 2).

        Returns:
            Tensor: Augmented node features (num_nodes, in_channels + 4).
        """

        row = edge_index[0]
        num_nodes = x.size(0)

        # Mean and max aggregation of incident edge attributes per node
        # edge_attr has 2 columns so each reduction produces a (num_nodes, 2) tensor,
        # giving 4 appended values in total
        edge_mean = scatter(edge_attr, row, dim=0, dim_size=num_nodes, reduce='mean')

        # Clamp max to 0 to guard against -inf for isolated nodes (no incident edges)
        edge_max  = scatter(edge_attr, row, dim=0, dim_size=num_nodes, reduce='max').clamp(min=0)

        # Concatenate aggregated edge statistics onto node features
        return torch.cat([x, edge_mean, edge_max], dim=1)

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

        # Augment node features with aggregated edge statistics
        x = self.augment_node_features(x, edge_index, edge_attr)

        # Project augmented node features into hidden space via MLP
        x = self.node_encoder(x)

        # Squeeze color-difference weight to scalar for GCNConv
        edge_weight = edge_attr[:, 0]

        # Apply 3 GCN layers with residual connections, batch norm, and dropout
        x = self.dropout(self.relu(self.bn1(self.conv1(x, edge_index, edge_weight))) + x)
        x = self.dropout(self.relu(self.bn2(self.conv2(x, edge_index, edge_weight))) + x)
        x = self.dropout(self.relu(self.bn3(self.conv3(x, edge_index, edge_weight))) + x)

        # Global pooling (mean + max)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        return self.classifier(x)


def evaluate(model, loader, num_classes, device):
    """Run inference over a data loader and collect predictions and labels.

    Args:
        model (nn.Module): Trained model in eval mode.
        loader (DataLoader): Data loader to evaluate.
        num_classes (int): Total number of classes.
        device (str): "cuda" or "cpu".

    Returns:
        acc (float): Overall accuracy.
        per_class_acc (np.ndarray): Per-class accuracy array of shape (num_classes,).
        confusion (np.ndarray): Confusion matrix of shape (num_classes, num_classes),
            where confusion[true, pred] is the count of samples with true label
            ``true`` predicted as ``pred``.
    """

    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = out.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Confusion matrix: rows = true class, cols = predicted class
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1

    # Per-class accuracy: diagonal / row sum (guard against empty classes)
    row_sums      = confusion.sum(axis=1)
    per_class_acc = np.where(row_sums > 0, confusion.diagonal() / row_sums, 0.0)

    acc = confusion.diagonal().sum() / max(len(all_labels), 1)

    return acc, per_class_acc, confusion


def train(model, train_loader, test_loader, epochs=100, lr=0.001, weight_decay=1e-4,
          device="cuda", num_classes=10, class_names=None):
    """Train model and return best-performing weights, per-class accuracy, and confusion matrix.

    A final evaluation pass is performed after restoring the best checkpoint so that
    the returned metrics always correspond to the best model state.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization.
        device (str): "cuda" or "cpu".
        num_classes (int): Number of output classes.
        class_names (list[str] | None): Optional class name labels for display.

    Returns:
        model (nn.Module): Trained model with best weights loaded.
        confusion (np.ndarray): Confusion matrix (num_classes, num_classes).
        per_class_acc (np.ndarray): Per-class accuracy (num_classes,).
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

            out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- Evaluation phase ----
        acc, _, _ = evaluate(model, test_loader, num_classes, device)
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"\nEpoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

    # Restore best checkpoint and run a final clean evaluation pass for metrics
    model.load_state_dict(best_model_state)
    best_acc, per_class_acc, confusion = evaluate(model, test_loader, num_classes, device)

    # ---- Shared formatting helpers ----
    labels     = [class_names[i] if class_names else str(i) for i in range(num_classes)]
    max_digits = max(len(str(confusion.max())), 4)   # minimum 4 chars for readability
    name_w     = max(max(len(l) for l in labels), 10) # minimum 10 chars for label column

    def hline(left, mid, right, fill, widths):
        """Build a horizontal rule from box-drawing characters."""
        return left + mid.join(fill * w for w in widths) + right

    # ---- Per-class accuracy summary table ----
    col_w = [name_w, max_digits + 2, max_digits + 2, 8]  # Class | Correct | Total | Acc
    heads = ["Class", "Correct", "Total", "Acc"]

    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")
    print("\nPer-class accuracy")
    print(hline("┌", "┬", "┐", "─", [w + 2 for w in col_w]))
    print("│ " + " │ ".join(h.center(col_w[i]) for i, h in enumerate(heads)) + " │")
    print(hline("├", "┼", "┤", "─", [w + 2 for w in col_w]))

    for cls_idx in range(num_classes):
        tp    = confusion[cls_idx, cls_idx]
        total = int(confusion[cls_idx].sum())
        acc   = per_class_acc[cls_idx]
        cells = [
            labels[cls_idx].ljust(col_w[0]),
            str(tp).rjust(col_w[1]),
            str(total).rjust(col_w[2]),
            f"{acc:.4f}".rjust(col_w[3]),
        ]
        print("│ " + " │ ".join(cells) + " │")

    print(hline("└", "┴", "┘", "─", [w + 2 for w in col_w]))

    # ---- Per-class binary confusion matrices ----
    # Each matrix shows TP/FN/FP/TN for one class treated as the positive class.
    cell_w  = max(max_digits + 4, 10)   # wide enough for "TP: NNNN" labels
    label_w = max(name_w, 10)

    print("\nPer-class binary confusion matrices")

    for cls_idx in range(num_classes):
        name = labels[cls_idx]

        tp = int(confusion[cls_idx, cls_idx])
        fn = int(confusion[cls_idx].sum()) - tp          # true class, predicted other
        fp = int(confusion[:, cls_idx].sum()) - tp       # other class, predicted this
        tn = int(confusion.sum()) - tp - fn - fp

        col_heads = [f"Pred: {name}", "Pred: other"]
        row_heads = [f"True: {name}", "True: other"]

        # Dynamic column widths: fit label text and counts
        c0 = max(label_w, max(len(h) for h in row_heads))
        c1 = max(cell_w,  len(col_heads[0]))
        c2 = max(cell_w,  len(col_heads[1]))

        top  = hline("┌", "┬", "┐", "─", [c0 + 2, c1 + 2, c2 + 2])
        mid  = hline("├", "┼", "┤", "─", [c0 + 2, c1 + 2, c2 + 2])
        bot  = hline("└", "┴", "┘", "─", [c0 + 2, c1 + 2, c2 + 2])

        def row(label, v1_label, v1, v2_label, v2):
            c1_text = f"{v1_label}: {v1}".center(c1)
            c2_text = f"{v2_label}: {v2}".center(c2)
            return f"│ {label:<{c0}} │ {c1_text} │ {c2_text} │"

        print(f"\n  {name}")
        print(top)
        # Header row (col labels, no values)
        print(f"│ {'':^{c0}} │ {col_heads[0]:^{c1}} │ {col_heads[1]:^{c2}} │")
        print(mid)
        print(row(row_heads[0], "TP", tp, "FN", fn))
        print(mid)
        print(row(row_heads[1], "FP", fp, "TN", tn))
        print(bot)

    # ---- Total confusion matrix ----
    # Cell width: fit the largest count and the column header (class name)
    tot_cell_w = max(max_digits, max(len(l) for l in labels)) + 2
    row_label_w = name_w

    print("\nTotal confusion matrix  (rows = true class, cols = predicted class)")
    # Top border
    print(hline("┌", "┬", "┐", "─", [row_label_w + 2] + [tot_cell_w + 2] * num_classes))
    # Column header row
    header_cells = "".join(f" {l:^{tot_cell_w}} │" for l in labels)
    print(f"│ {'':^{row_label_w}} │{header_cells}")
    print(hline("├", "┼", "┤", "─", [row_label_w + 2] + [tot_cell_w + 2] * num_classes))
    # Data rows
    for i, label in enumerate(labels):
        vals = "".join(f" {confusion[i, j]:^{tot_cell_w}} │" for j in range(num_classes))
        # Highlight diagonal (true positive) with a marker
        diag_vals = []
        for j in range(num_classes):
            val_str = str(confusion[i, j])
            if i == j:
                val_str = f"[{val_str}]"
            diag_vals.append(f" {val_str:^{tot_cell_w}} │")
        print(f"│ {label:<{row_label_w}} │{''.join(diag_vals)}")
        if i < num_classes - 1:
            print(hline("├", "┼", "┤", "─", [row_label_w + 2] + [tot_cell_w + 2] * num_classes))
    print(hline("└", "┴", "┘", "─", [row_label_w + 2] + [tot_cell_w + 2] * num_classes))
    print("  Diagonal values in [brackets] are correct predictions (TP per class).\n")

    return model, confusion, per_class_acc


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
    data_root     = "./data"

    # -- SLIC (superpixel image graph encoding method) --
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

    # CIFAR-10 class names; update if switching to a different dataset
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    model = SuperpixelGCN(hidden_dim=hidden_dim, dropout=dropout)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trained_model, confusion, per_class_acc = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        num_classes=len(class_names),
        class_names=class_names,
    )

    # Save confusion matrix and per-class accuracy alongside model weights
    save_path = "superpixel_gcn_best.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Saved model to '{save_path}'")

    np.save("confusion_matrix.npy", confusion)
    np.save("per_class_acc.npy", per_class_acc)
    print("Saved confusion_matrix.npy and per_class_acc.npy")


if __name__ == "__main__":
    main()