# Molecular Graph Captioning

This project implements a deep learning model for generating captions from molecular graphs. It was developed in the context of the **[Molecular Graph Captioning](https://www.kaggle.com/competitions/molecular-graph-captioning/overview)** Kaggle competition.

## Prerequisites

Before starting, ensure that the following data files are placed in the `src/data` directory:

- `test_graphs.pkl`
- `train_graphs.pkl`
- `validation_graphs.pkl`

## Installation

To set up the environment, run the following command:

```bash
make env
```

Once the installation is complete, activate the virtual environment:

```bash
source .venv/bin/activate
```

## Usage

Follow the steps below to run the pipeline in order:

### 1. Generate Embeddings

Generate the necessary embeddings for training:

```bash
make embeddings
```

### 2. Train the Model

Train the model using the default configuration:

```bash
make train
```

**Note:** You can modify the training hyperparameters, such as the number of epochs (default: 100) and batch size (default: 128), by editing the variables in `src/train/train.py`.

### 3. Inference

Run inference to generate captions for the test set:

```bash
make infer
```

This will generate the predicted captions in the following file:
`src/inference/predicted_captions.csv`

## Authors

- Maxence Debes
- Sinoué Gad