import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import torchvision

from main import build_superpixel_dataset, SuperpixelGCN
from torch_geometric.nn import global_mean_pool, global_max_pool


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def pca_torch(X, n_components=2):
    """
    PCA using PyTorch only, so sklearn is not required.
    """
    X = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(X, q=n_components)
    return X @ V[:, :n_components]


def get_graph_embedding(model, batch):
    """
    Reproduce the pooled graph embedding from main.py without modifying main.py.
    This is the graph-level representation right before the final classifier.
    """
    x = model.augment_node_features(batch.x, batch.edge_index, batch.edge_attr)
    x = model.node_encoder(x)

    edge_weight = model.edge_weight_proj(batch.edge_attr).squeeze(-1)

    x = model.dropout(model.relu(model.bn1(model.conv1(x, batch.edge_index, edge_weight))) + x)
    x = model.dropout(model.relu(model.bn2(model.conv2(x, batch.edge_index, edge_weight))) + x)
    x = model.dropout(model.relu(model.bn3(model.conv3(x, batch.edge_index, edge_weight))) + x)

    gate = torch.softmax(model.pool_gate, dim=0)
    pooled = torch.cat([
        gate[0] * global_mean_pool(x, batch.batch),
        gate[1] * global_max_pool(x, batch.batch),
    ], dim=1)

    return pooled


def collect_embeddings(model, loader, device, max_samples=1000):
    model.eval()
    zs = []
    ys = []
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = get_graph_embedding(model, batch)
            y = batch.y

            zs.append(z.cpu())
            ys.append(y.cpu())
            total += y.size(0)

            if total >= max_samples:
                break

    embeddings = torch.cat(zs, dim=0)[:max_samples]
    labels = torch.cat(ys, dim=0)[:max_samples].numpy()
    return embeddings, labels


def reduce_embeddings(embeddings, method="pca", random_state=42):
    method = method.lower()

    if method == "pca":
        points_2d = pca_torch(embeddings, n_components=2).numpy()
        title = "PCA of Superpixel Graph Embeddings"

    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError(
                "t-SNE requires scikit-learn. Install it with: pip install scikit-learn"
            )

        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=30,
            init="pca"
        )
        points_2d = reducer.fit_transform(embeddings.numpy())
        title = "t-SNE of Superpixel Graph Embeddings"

    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    return points_2d, title


def plot_embeddings(points_2d, labels, class_names=None, save_path=None, title="Embedding Space"):
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        name = class_names[label] if class_names is not None else str(label)
        plt.scatter(points_2d[idx, 0], points_2d[idx, 1], s=22, alpha=0.75, label=name)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_embeddings_side_by_side(
    points_before,
    points_after,
    labels_before,
    labels_after,
    class_names=None,
    save_path=None,
    title_before="Before Training",
    title_after="After Training"
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    unique_before = np.unique(labels_before)
    unique_after = np.unique(labels_after)

    for label in unique_before:
        idx = labels_before == label
        name = class_names[label] if class_names is not None else str(label)
        axes[0].scatter(
            points_before[idx, 0],
            points_before[idx, 1],
            s=22,
            alpha=0.75,
            label=name
        )

    axes[0].set_title(title_before)
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")

    for label in unique_after:
        idx = labels_after == label
        name = class_names[label] if class_names is not None else str(label)
        axes[1].scatter(
            points_after[idx, 0],
            points_after[idx, 1],
            s=22,
            alpha=0.75,
            label=name
        )

    axes[1].set_title(title_after)
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, bbox_to_anchor=(1.02, 0.98), loc="upper left", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding space for superpixel GCN")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights")
    parser.add_argument("--compare", action="store_true", help="Show before vs after training side-by-side")
    parser.add_argument("--method", type=str, default="pca", choices=["tsne", "pca"], help="Dimensionality reduction method")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of graph embeddings to plot")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding extraction")
    parser.add_argument("--n-segments", type=int, default=50, help="Number of SLIC superpixels")
    parser.add_argument("--compactness", type=float, default=10, help="SLIC compactness")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Model dropout")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root")
    parser.add_argument("--save-path", type=str, default="embedding_space.png", help="Path to save the plot")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_class = torchvision.datasets.CIFAR10
    print("Building superpixel datasets...")
    train_graphs, val_graphs, test_graphs = build_superpixel_dataset(
        dataset_class=dataset_class,
        root=args.data_root,
        n_segments=args.n_segments,
        compactness=args.compactness,
    )

    split_map = {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }
    graphs = split_map[args.split]
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    model = SuperpixelGCN(hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model.to(device)

    print(f"Collecting embeddings from {args.split} split...")

    if args.compare:
        if args.weights is None:
            raise ValueError("--compare requires --weights <path_to_trained_model>")

        print(f"Visualizing {args.method.upper()} embeddings — Before vs After Training")

        # BEFORE training
        print("Using random initialized model for BEFORE training plot...")
        embeddings_before, labels_before = collect_embeddings(
            model, loader, device, max_samples=args.max_samples
        )
        points_before, title_before = reduce_embeddings(embeddings_before, method=args.method)
        title_before = f"{title_before} (Before Training)"

        # AFTER training
        print("Loading trained weights for AFTER training plot...")
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state)

        embeddings_after, labels_after = collect_embeddings(
            model, loader, device, max_samples=args.max_samples
        )
        points_after, title_after = reduce_embeddings(embeddings_after, method=args.method)
        title_after = f"{title_after} (After Training)"

        plot_embeddings_side_by_side(
            points_before,
            points_after,
            labels_before,
            labels_after,
            class_names=CLASS_NAMES,
            save_path=args.save_path,
            title_before=title_before,
            title_after=title_after,
        )

    else:
        if args.weights is not None:
            print("Loading trained weights...")
            state = torch.load(args.weights, map_location=device)
            model.load_state_dict(state)
            title_suffix = "After Training"
        else:
            print("No weights provided — using random initialized model (Before Training)")
            title_suffix = "Before Training"

        print(f"Visualizing {args.method.upper()} embeddings — {title_suffix}")

        embeddings, labels = collect_embeddings(
            model, loader, device, max_samples=args.max_samples
        )
        points_2d, title = reduce_embeddings(embeddings, method=args.method)
        title = f"{title} ({title_suffix})"

        plot_embeddings(
            points_2d,
            labels,
            class_names=CLASS_NAMES,
            save_path=args.save_path,
            title=title
        )


if __name__ == "__main__":
    main()