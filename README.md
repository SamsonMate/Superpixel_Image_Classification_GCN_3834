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
- **Torch Scatter** - for edge aggregation for feature augmentation

---

## Installation

It is recommended to use a virtual environment:

```bash
cd Superpixel_Image_Classification_GCN_3834
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install PyTorch first by following the official instructions for your platform and CUDA version at https://pytorch.org/get-started/locally/

Seperate PyTorch extension used for performance increase:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA}.html
```

Then install the remaining dependencies:

```bash
pip install torch-geometric scikit-image scipy tqdm numpy
```

---

## If it says device is CPU but you have a CUDA enabled GPU:

Problem: using cpu despite having CUDA enabled GPU.
Solution: Reinstall pytorch completely with correct sources.

Sometimes if you've installed pytorch before especially in a rush you may have accidentally installed
the CPU version of pytorch and it's subsidiaries. Below are the commands needed to reinstall

0) 
Not necessary but good practice to install/update official NVIDIA drivers and CUDA toolkit:
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

1) 
```bash
# To uninstall torch
pip uninstall torch torchvision torchaudio -y
pip cache purge
```

2) 
Go to https://pytorch.org/get-started/locally/
Under "compute platform" select "CUDA <version>"
Ensure everything else is correct in regards to your system (OS, package, language, etc...)
Then copy the provided url

3) 
```bash
# To reinstall torch literally just paste the command in your terminal
```

---

## Usage

Note: Running this in a regular terminal will yield best performance. VSCode terminal is laggy and throttled.

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
