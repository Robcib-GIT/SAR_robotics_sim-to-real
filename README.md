# High-Fidelity Synthetic RGB-D Datasets for SAR Perception Tasks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16587242.svg)](https://doi.org/10.5281/zenodo.16587242)  
📦 **Download the dataset on [Zenodo](https://doi.org/10.5281/zenodo.16587242)**

**📌 Reproducible pipeline for generating annotated datasets in post-disaster environments using NVIDIA Isaac Sim.**


<img width="1827" height="544" alt="git_car" src="https://github.com/user-attachments/assets/789c36e1-e433-4bd5-881a-a58bce6b5e0b" />

## 🛠️ Features

- Procedural generation of photorealistic disaster environments.
- Automatic pixel-wise labeling of terrain textures.
- Configurable RGB-D sensor simulation with noise models.
- Training pipeline for CNN and BLNN terrain classifiers.
- Evaluation scripts for domain transfer to real-world datasets.
- Ready-to-use dataset samples.

<img width="1707" height="492" alt="git_segment" src="https://github.com/user-attachments/assets/16abc001-a051-479a-ad8a-2d76ab12eadf" />

## 🚀 Usage

### 1. Run Simulation and Auto-labeling in Isaac Sim

Launch Isaac Sim and execute the script below to simulate the quadruped robot moving through a disaster environment while collecting RGB-D images and semantic labels automatically.

```bash
python3 go1_car_christyan.py
```


📌 This script:

Spawns a quadruped robot in a procedurally generated terrain.

Simulates movement with onboard RGB-D sensing.

Automatically records and labels each frame for terrain classification tasks.

### 2. Convert Raw Data into Standard Dataset Format

Once the simulation has completed and data is collected, convert the raw images and semantic labels into a structured dataset (e.g., Pascal VOC) using the following script:

```bash
python3 dataset_conversion_isaac_a_cnn_pascal.py
```

📌 This script:

Organizes RGB images, depth maps, and annotations.

Converts Isaac Sim outputs into formats ready for CNN training (Pascal VOC, etc.).

Optionally filters or resizes the dataset.
