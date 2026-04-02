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
        val_graphs (list[Data]): Validation graph dataset (10% of train split).
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

    all_train_graphs = build(trainset, "train")
    test_graphs      = build(testset,  "test")

    # Partition validation set (10%) from the training split using a fixed seed
    # for reproducibility. The validation set drives scheduler stepping and
    # best-model checkpointing; the test set is never touched during training.
    rng     = torch.Generator().manual_seed(42)
    n_total = len(all_train_graphs)
    n_val   = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    indices      = torch.randperm(n_total, generator=rng).tolist()
    train_graphs = [all_train_graphs[i] for i in indices[:n_train]]
    val_graphs   = [all_train_graphs[i] for i in indices[n_train:]]

    return train_graphs, val_graphs, test_graphs


class SuperpixelGCN(nn.Module):
    """GCN-based model with MLP node encoder, learnable edge weight projection,
    learnable pooling gate, and residual connections for classifying superpixel graphs.

    Edge attributes are incorporated in two ways:
      1. Pre-convolution node feature augmentation (mean/max scatter of incident edges).
      2. A learnable linear projection (edge_weight_proj) maps the 2-dim edge_attr to
         a scalar edge weight passed into each GCNConv layer, replacing the previous
         hardcoded selection of edge_attr[:, 0]. This lets the model learn which
         combination of color-difference and centroid-distance is most informative
         for message passing.

    A learnable pooling gate (pool_gate) replaces the fixed mean+max concatenation.
    It is a 2-element softmax-normalized parameter that interpolates between global
    mean pooling and global max pooling before the classifier, giving the model
    control over how graph-level statistics are aggregated. The classifier input
    width is preserved at hidden_dim * 2 (gated_mean || gated_max).

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

        # ------------------------------------------------------------------
        # Learnable edge weight projection
        # ------------------------------------------------------------------
        # Projects the 2-dim edge_attr (color difference, centroid distance)
        # to a non-negative scalar used as the edge weight in each GCNConv call.
        # Sigmoid activation ensures weights remain in (0, 1).
        # Previously edge_attr[:, 0] was hardcoded; this lets the model learn the
        # optimal linear combination of both edge features during training.
        self.edge_weight_proj = nn.Sequential(
            Linear(2, 1),
            nn.Sigmoid()
        )

        # ------------------------------------------------------------------
        # Learnable pooling gate
        # ------------------------------------------------------------------
        # A 2-element parameter softmax-normalised at forward time to produce
        # convex combination weights [w_mean, w_max] for global mean and max pooling.
        # Initialised to zeros (equal weighting after softmax), deviating from the
        # previous fixed concatenation. The classifier input size remains hidden_dim * 2:
        # out = cat(w_mean * mean_pool, w_max * max_pool).
        self.pool_gate = nn.Parameter(torch.zeros(2))  # raw logits; softmax applied in forward

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
        # Input is still hidden_dim * 2: gated_mean (hidden_dim) || gated_max (hidden_dim)
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

        # ------------------------------------------------------------------
        # Learnable edge weight projection
        # ------------------------------------------------------------------
        # edge_weight_proj maps (num_edges, 2) → (num_edges, 1) then squeeze to
        # (num_edges,) as required by GCNConv. Sigmoid keeps values in (0, 1).
        edge_weight = self.edge_weight_proj(edge_attr).squeeze(-1)

        # Apply 3 GCN layers with residual connections, batch norm, and dropout
        x = self.dropout(self.relu(self.bn1(self.conv1(x, edge_index, edge_weight))) + x)
        x = self.dropout(self.relu(self.bn2(self.conv2(x, edge_index, edge_weight))) + x)
        x = self.dropout(self.relu(self.bn3(self.conv3(x, edge_index, edge_weight))) + x)

        # ------------------------------------------------------------------
        # Learnable pooling gate
        # ------------------------------------------------------------------
        # Softmax over the 2 raw logits gives convex combination weights.
        # w[0] scales global mean pooling, w[1] scales global max pooling.
        # Scaled tensors are concatenated to preserve the hidden_dim * 2
        # input width expected by the classifier.
        gate   = torch.softmax(self.pool_gate, dim=0)
        pooled = torch.cat([
            gate[0] * global_mean_pool(x, batch),
            gate[1] * global_max_pool(x, batch),
        ], dim=1)

        return self.classifier(pooled)


def evaluate(model, loader, num_classes, device):
    """Run inference over a data loader and collect predictions and labels.

    Precision and recall are computed in the standard multi-class sense:
        precision[i] = TP_i / (TP_i + FP_i)  — how often class-i predictions are correct
        recall[i]    = TP_i / (TP_i + FN_i)  — how often class-i samples are found
        f1[i]        = 2 * P_i * R_i / (P_i + R_i)

    Macro averages are the unweighted mean across classes.
    Weighted averages weight each class by its true sample count (support).
    All per-class metrics default to 0.0 for classes with no support or no predictions.

    Args:
        model (nn.Module): Trained model in eval mode.
        loader (DataLoader): Data loader to evaluate.
        num_classes (int): Total number of classes.
        device (str): "cuda" or "cpu".

    Returns:
        acc (float): Overall accuracy.
        per_class_acc (np.ndarray): Per-class accuracy (num_classes,).
        confusion (np.ndarray): Confusion matrix (num_classes, num_classes),
            where confusion[true, pred] is the count of samples with true label
            ``true`` predicted as ``pred``.
        metrics (dict): Dictionary containing:
            per_class_precision (np.ndarray), per_class_recall (np.ndarray),
            per_class_f1 (np.ndarray), macro_precision (float),
            macro_recall (float), macro_f1 (float), weighted_precision (float),
            weighted_recall (float), weighted_f1 (float).
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

    diag     = confusion.diagonal()
    row_sums = confusion.sum(axis=1)   # true positives + false negatives per class
    col_sums = confusion.sum(axis=0)   # true positives + false positives per class

    # Per-class accuracy = recall (TP / true support); guard against empty classes
    per_class_acc = np.where(row_sums > 0, diag / row_sums, 0.0)

    # Per-class precision: TP / (TP + FP); guard against zero predicted support
    per_class_precision = np.where(col_sums > 0, diag / col_sums, 0.0)

    # Per-class recall (identical to per_class_acc by definition)
    per_class_recall = per_class_acc

    # Per-class F1: harmonic mean of precision and recall
    pr_sum      = per_class_precision + per_class_recall
    per_class_f1 = np.where(pr_sum > 0, 2 * per_class_precision * per_class_recall / pr_sum, 0.0)

    # Macro averages: unweighted mean across all classes
    macro_precision = float(per_class_precision.mean())
    macro_recall    = float(per_class_recall.mean())
    macro_f1        = float(per_class_f1.mean())

    # Weighted averages: weight by true class support (row_sums)
    total = max(int(row_sums.sum()), 1)
    weighted_precision = float((per_class_precision * row_sums).sum() / total)
    weighted_recall    = float((per_class_recall    * row_sums).sum() / total)
    weighted_f1        = float((per_class_f1        * row_sums).sum() / total)

    acc = int(diag.sum()) / max(len(all_labels), 1)

    metrics = dict(
        per_class_precision = per_class_precision,
        per_class_recall    = per_class_recall,
        per_class_f1        = per_class_f1,
        macro_precision     = macro_precision,
        macro_recall        = macro_recall,
        macro_f1            = macro_f1,
        weighted_precision  = weighted_precision,
        weighted_recall     = weighted_recall,
        weighted_f1         = weighted_f1,
    )

    return acc, per_class_acc, confusion, metrics


def train(model, train_loader, val_loader, test_loader, epochs=100, lr=0.001, weight_decay=1e-4,
          device="cuda", num_classes=10, class_names=None):
    """Train model and return best-performing weights, per-class accuracy, and confusion matrix.

    The validation set (val_loader) is used for learning-rate scheduling and
    best-model checkpointing. The test set (test_loader) is evaluated once per
    epoch for reporting only and has no influence on model selection.

    A final evaluation pass is performed after restoring the best checkpoint so that
    the returned metrics always correspond to the best model state.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader (used for checkpointing).
        test_loader (DataLoader): Test data loader (reporting only).
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
        metrics (dict): Precision, recall, and F1 scores (per-class, macro, weighted).
    """

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Scheduler monitors validation accuracy rather than test accuracy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
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
        model.eval()
        val_acc,  _, _, val_metrics  = evaluate(model, val_loader,  num_classes, device)
        test_acc, _, _, _            = evaluate(model, test_loader, num_classes, device)

        # Scheduler steps on validation accuracy; test set is untouched
        scheduler.step(val_acc)

        # Checkpoint on best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        # Log learnable pool gate weights each epoch for transparency
        gate = torch.softmax(model.pool_gate.detach().cpu(), dim=0)

        print(
            f"\nEpoch {epoch+1:03d} | Loss: {avg_loss:.4f} "
            f"| Train Acc: {val_acc:.4f} (best: {best_val_acc:.4f}) "
            f"| F1: {val_metrics['macro_f1']:.4f} "
            f"| Test Acc: {test_acc:.4f} "
        )

    # Restore best checkpoint and run a final clean evaluation pass for metrics
    model.load_state_dict(best_model_state)
    best_val_acc, per_class_acc, confusion, metrics = evaluate(model, val_loader, num_classes, device)
    _, _, _, test_metrics = evaluate(model, test_loader, num_classes, device)

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

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
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

    # ---- Per-class precision / recall / F1 table ----
    # Displayed for the test split so the table reflects held-out performance.
    prf_col_w = [name_w, 9, 9, 9, max_digits + 2]  # Class | Precision | Recall | F1 | Support
    prf_heads = ["Class", "Precision", "Recall", "F1", "Support"]

    t_prec    = test_metrics["per_class_precision"]
    t_rec     = test_metrics["per_class_recall"]
    t_f1      = test_metrics["per_class_f1"]
    _, _, test_confusion, _ = evaluate(model, test_loader, num_classes, device)
    t_support = test_confusion.sum(axis=1)

    print("\nPer-class precision / recall / F1  (test set)")
    print(hline("┌", "┬", "┐", "─", [w + 2 for w in prf_col_w]))
    print("│ " + " │ ".join(h.center(prf_col_w[i]) for i, h in enumerate(prf_heads)) + " │")
    print(hline("├", "┼", "┤", "─", [w + 2 for w in prf_col_w]))

    for cls_idx in range(num_classes):
        cells = [
            labels[cls_idx].ljust(prf_col_w[0]),
            f"{t_prec[cls_idx]:.4f}".rjust(prf_col_w[1]),
            f"{t_rec[cls_idx]:.4f}".rjust(prf_col_w[2]),
            f"{t_f1[cls_idx]:.4f}".rjust(prf_col_w[3]),
            str(t_support[cls_idx]).rjust(prf_col_w[4]),
        ]
        print("│ " + " │ ".join(cells) + " │")

    print(hline("├", "┼", "┤", "─", [w + 2 for w in prf_col_w]))

    # Macro average row
    macro_cells = [
        "macro avg".ljust(prf_col_w[0]),
        f"{test_metrics['macro_precision']:.4f}".rjust(prf_col_w[1]),
        f"{test_metrics['macro_recall']:.4f}".rjust(prf_col_w[2]),
        f"{test_metrics['macro_f1']:.4f}".rjust(prf_col_w[3]),
        "".rjust(prf_col_w[4]),
    ]
    print("│ " + " │ ".join(macro_cells) + " │")

    # Weighted average row
    weighted_cells = [
        "weighted avg".ljust(prf_col_w[0]),
        f"{test_metrics['weighted_precision']:.4f}".rjust(prf_col_w[1]),
        f"{test_metrics['weighted_recall']:.4f}".rjust(prf_col_w[2]),
        f"{test_metrics['weighted_f1']:.4f}".rjust(prf_col_w[3]),
        str(int(t_support.sum())).rjust(prf_col_w[4]),
    ]
    print("│ " + " │ ".join(weighted_cells) + " │")

    print(hline("└", "┴", "┘", "─", [w + 2 for w in prf_col_w]))

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

    return model, confusion, per_class_acc, metrics


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
    train_graphs, val_graphs, test_graphs = build_superpixel_dataset(
        dataset_class=dataset_class,
        root=data_root,
        n_segments=n_segments,
        compactness=compactness
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False)

    print(
        f"Dataset ready — {len(train_graphs)} train, "
        f"{len(val_graphs)} val, {len(test_graphs)} test"
    )

    # CIFAR-10 class names; update if switching to a different dataset
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    model = SuperpixelGCN(hidden_dim=hidden_dim, dropout=dropout)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trained_model, confusion, per_class_acc, metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        num_classes=len(class_names),
        class_names=class_names,
    )

    # Save model weights and evaluation artefacts
    save_path = "superpixel_gcn_best.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Saved model to '{save_path}'")

    np.save("confusion_matrix.npy", confusion)
    np.save("per_class_acc.npy", per_class_acc)
    np.save("per_class_precision.npy", metrics["per_class_precision"])
    np.save("per_class_recall.npy",    metrics["per_class_recall"])
    np.save("per_class_f1.npy",        metrics["per_class_f1"])
    print("Saved confusion_matrix.npy, per_class_acc.npy, per_class_precision.npy, "
          "per_class_recall.npy, per_class_f1.npy")


if __name__ == "__main__":
    main()