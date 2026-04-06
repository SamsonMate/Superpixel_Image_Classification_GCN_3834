import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import slic, mark_boundaries
from skimage import graph
from scipy.ndimage import mean as ndimage_mean


def visualize_individual_superpixel_graph(img, n_segments=50, compactness=10, save_path=None):
    """
    Visualize:
      1. Original image
      2. Superpixel segmentation
      3. Superpixel graph overlay
    """

    img_np = img.permute(1, 2, 0).numpy()
    h, w = img_np.shape[:2]

    # --- Superpixels ---
    segments = slic(img_np, n_segments=n_segments, compactness=compactness, start_label=0)
    num_nodes = segments.max() + 1
    index = np.arange(num_nodes)

    # --- Centroids ---
    row_grid, col_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    centroid_row = ndimage_mean(row_grid, labels=segments, index=index)
    centroid_col = ndimage_mean(col_grid, labels=segments, index=index)

    # --- Graph (RAG) ---
    rag = graph.rag_mean_color(img_np, segments)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Superpixels
    axes[1].imshow(mark_boundaries(img_np, segments))
    axes[1].set_title(f"SLIC Superpixels (n={num_nodes})")
    axes[1].axis("off")

    # 3. Graph overlay
    axes[2].imshow(img_np)
    axes[2].set_title("Superpixel Graph Overlay")
    axes[2].axis("off")

    # --- Draw edges ---
    for i, j, data in rag.edges(data=True):
        x1, y1 = centroid_col[i], centroid_row[i]
        x2, y2 = centroid_col[j], centroid_row[j]

        weight = data.get("weight", 1.0)
        line_width = max(0.5, 3.0 / (1.0 + weight))

        axes[2].plot([x1, x2], [y1, y2], linewidth=line_width)

    # --- Node colors (mean RGB) ---
    node_colors = np.zeros((num_nodes, 3), dtype=np.float32)
    for c in range(3):
        node_colors[:, c] = ndimage_mean(img_np[:, :, c], labels=segments, index=index)

    axes[2].scatter(
        centroid_col,
        centroid_row,
        s=30,
        c=node_colors,
        edgecolors="black"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------
# 🔹 RUN THIS FILE DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    img, label = dataset[0]

    print(f"Visualizing CIFAR-10 sample (label={label})")

    visualize_individual_superpixel_graph(
        img,
        n_segments=50,
        compactness=10,
        save_path="example_superpixel_graph.png"
    )