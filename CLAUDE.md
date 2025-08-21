# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nicheformer is a foundation model for single-cell and spatial omics data. It's built using PyTorch Lightning and provides pre-training, fine-tuning, and embedding extraction capabilities for biological data analysis.

## Installation and Setup

The project uses conda for environment management:

```bash
# Use this environment that has been pre-created.
conda init
conda activate niche_env

```
## Source WANDB API KEY from the .env file before executing the training script

## Key Commands

### Training and Fine-tuning
- `python downstream_fine_tune.py` - Comprehensive fine-tuning script with command-line interface, focus all commands to work with this script.


### Code Quality
- `pytest` - Run tests (when available)

### Data Processing
- Scripts in `data/spatialcorpus-110M/` for downloading and preprocessing datasets
- Notebooks in `notebooks/tokenization/` for data tokenization examples

## Architecture

### Core Components

**Models** (`src/nicheformer/models/`):
- `Nicheformer` - Main transformer-based foundation model
- `NicheformerFineTune` - Fine-tuning wrapper with task-specific heads
- Supports classification, regression, and distribution prediction tasks

**Data** (`src/nicheformer/data/`):
- `NicheformerDataset` - PyTorch dataset for handling single-cell/spatial omics data
- `datamodules.py` - PyTorch Lightning data modules
- Handles AnnData format, technology-specific normalization, and train/val/test splits

**Configuration** (`src/nicheformer/config_files/`):
- `_config_train.py` - Pre-training configuration
- `_config_fine_tune.py` - Fine-tuning configuration
- `_config_embeddings.py` - Embedding extraction configuration

### Key Features

- **Multi-modal support**: Handles different spatial omics technologies (Xenium, MERFISH, CosMX, ISS)
- **Flexible fine-tuning**: Supports various downstream tasks (cell type annotation, niche classification, etc.)
- **Technology-aware normalization**: Uses technology-specific mean vectors for normalization
- **Configurable architecture**: Learnable positional encodings, species/assay/modality tokens

### Data Flow

1. Raw spatial omics data (H5AD format) → `NicheformerDataset`
2. Tokenization and chunking with technology-specific normalization
3. Model training/fine-tuning with PyTorch Lightning
4. Embedding extraction or prediction outputs stored back to AnnData

## Working with the Codebase

- Pre-trained model weights should be downloaded from Mendeley Data and placed in the repository root
- Technology mean files are stored in `data/model_means/`
- Output checkpoints and logs go to `output/` directory
- Configuration parameters are primarily handled through the config files, but `downstream_fine_tune.py` provides a command-line interface

## Important Notes

- No existing test suite - testing should be done functionally through training/fine-tuning scripts
- The codebase assumes CUDA availability for training but falls back to CPU
- Memory requirements can be significant for large spatial datasets
- Uses W&B (wandb) for experiment tracking in some scripts

# Interface Style
The cli tool lyra is inspired by the constellation Lyra, the Greek myth of Orpheus, and the musical instrument the lyre. Pepper all outputs with stylistic flair and emojis pertaining to these themes.