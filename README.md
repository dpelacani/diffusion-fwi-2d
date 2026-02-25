# Diffusion-FWI-2D

This repository implements an interleaved scheme of diffusion models with Full Waveform Inversion (FWI) in 2D. The code leverages the `stride` library for ultrasound simulation and integrates deep learning models for regularisation of inverse problems. See the `scripts` folder for main entry points.

## Installation

1. **Install stride** following the instructions in [here](https://github.com/trustimaging/stride)

2. **Clone this repository**:
   ```bash
   git clone git@github.com:dpelacani/diffusion-fwi-2d.git
   cd diffusion-fwi-2d
   ```

3. *(Optional but recommended)* Uninstall PyTorch if already present from Stride's installation:
   ```bash
   conda uninstall pytorch
   ```

4. **Install dependencies in Stride's environment**:
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter memory errors:
   ```bash
   mkdir ~/tmp # or any other tmp folder in a disk with storage
   TMPDIR=~/tmp pip install -r requirements.txt
   ```

5. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## Entry Points

Scripts in the `scripts` folder:
- `training.py`: Trains the diffusion model for FWI.
- `forward.py`: Runs forward modeling to generate synthetic ultrasound data.
- `inverse.py`: Performs inversion to reconstruct velocity models from data with optional integration of the interleaved diffusion scheme.
- `run.sh`: example SLURM batch script for running `slurm` jobs on HPC clusters.



## Folder Structure

- `scripts/`: Main entry point scripts for forward modeling, inversion, training, and HPC job submission.
- `src/diffusionfwi/`: Core Python package with submodules:
  - `dataset/`: Data loading and preprocessing.
  - `diffusion/`: Diffusion process pipeline implementations.
  - `fwi/`: FWI operators, pipelines, and utilities (visualisation, metrics, etc.).
  - `loops/`: Training and validation loops.
  - `models/`: Model architectures (UNet, Attention, etc.).
  - `utils/`: Metrics, visualization, and helper functions.
- `requirements.txt`: Python dependencies.
- `environment.yml`: Conda environment specification.