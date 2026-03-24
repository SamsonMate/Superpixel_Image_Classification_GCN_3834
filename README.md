# Superpixel Image Classification GCN

A Graph Convolutional Network (GCN) based model that classifies images from the CIFAR-10 dataset. Images are first encoded into superpixel graphs using SLIC segmentation, where each superpixel becomes a node with RGB and spatial features, and edges are derived from a Region Adjacency Graph (RAG) weighted by colour similarity. The GCN then learns over this graph structure to produce a class prediction across 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

Made for a university term project.

---

## Dependencies

- **PyTorch** — core deep learning framework
- **PyTorch Geometric** — graph neural network layers and data utilities (`GINEConv`, `DataLoader`, pooling)
- **Torchvision** — CIFAR-10 dataset download and transforms
- **Scikit-Image** — SLIC superpixel segmentation and RAG construction
- **NumPy** — array operations during graph encoding
- **SciPy** — per-superpixel mean feature computation (`scipy.ndimage.mean`)
- **tqdm** — progress bars for dataset encoding and training

---

## Installation

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install PyTorch first by following the official instructions for your platform and CUDA version at https://pytorch.org/get-started/locally/

Then install the remaining dependencies:

```bash
pip install torch-geometric torchvision scikit-image scipy tqdm
```

---

## Usage

```bash
python main.py
```

On first run, CIFAR-10 will be downloaded automatically to `./data`. All 60,000 images will then be encoded into superpixel graphs before training begins — this step is CPU-bound and may take 20–60 minutes depending on your hardware. Progress bars are shown for both encoding and training.

The best model checkpoint (by validation accuracy) is saved to `superpixel_gcn_best.pt` at the end of training.

---

## References

**Dataset:**
Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. Chapter 3 describes the CIFAR-10 dataset and its collection methodology.
https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
