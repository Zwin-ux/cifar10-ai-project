# Modular AI Training Lab

This repository is a modular AI training lab designed for experimenting with various models across different domains like Computer Vision (CV) and Natural Language Processing (NLP).

## Project Structure

```
.
├── config/          # Experiment configs (YAML)
│   └── experiments/
│       ├── cifar10_simplecnn.yaml
│       └── sst2_bert.yaml
├── datasets/        # Modular dataset loading scripts
│   ├── __init__.py
│   ├── cv_datasets.py
│   └── nlp_datasets.py
├── models/          # All model architectures
│   ├── __init__.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── bert_classifier.py
│   └── vision/
│       ├── __init__.py
│       └── simple_cnn.py
├── notebooks/       # Experiment analysis (Jupyter notebooks)
│   └── .keep         # Placeholder
├── trainers/        # Specialized trainers (CV, NLP, RL)
│   ├── __init__.py
│   ├── cv_trainer.py
│   └── nlp_trainer.py
├── utils/           # Logging, metrics, plotting, saving tools
│   ├── __init__.py
│   ├── plotting.py
│   └── saving.py
├── .gitignore
├── README.md
├── requirements.txt
└── train.py         # Entrypoint: model + dataset CLI & config
```

## Features

*   **Modular Design:** Easily add new models, datasets, and trainers.
*   **Configuration-Driven:** Use YAML files in `config/experiments` to define and manage experiments.
*   **CLI Overrides:** Override YAML configurations directly from the command line.
*   **Multi-Domain:** Supports CV (e.g., SimpleCNN on CIFAR-10) and NLP (e.g., BERT on SST-2) out-of-the-box.
*   **Extensible:** Planned support for more models, datasets, and domains like Reinforcement Learning and Generative AI.

## Prerequisites

Ensure you have Python 3.8+ installed.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your system and whether you want to use a GPU, you might need a specific version of PyTorch. Please refer to the [official PyTorch installation guide](https.pytorch.org/get-started/locally/) for instructions.*

## Running Experiments

The main script for running experiments is `train.py`. You can run an experiment by specifying a configuration file or by providing command-line arguments.

### Using a Configuration File

The recommended way to run experiments is by using a YAML configuration file from the `config/experiments/` directory.

**Example (CIFAR-10 with SimpleCNN):**
```bash
python train.py --config_file config/experiments/cifar10_simplecnn.yaml
```

**Example (SST-2 with BERT):**
```bash
python train.py --config_file config/experiments/sst2_bert.yaml
```

### Using Command-Line Arguments

You can also specify parameters directly via the command line. CLI arguments will override any values set in a configuration file if both are provided.

**Example (Override epochs for CIFAR-10 experiment):**
```bash
python train.py --config_file config/experiments/cifar10_simplecnn.yaml --epochs 10
```

**Example (Run CIFAR-10 without a config file, specifying key parameters):**
```bash
python train.py --model_type simplecnn --dataset_type cv --model_name simplecnn --dataset_name cifar10 --epochs 5 --batch_size 32 --lr 0.01 --save_path artifacts/my_cifar_run.pth
```

### Key Command-Line Arguments

*   `--config_file`: Path to the YAML experiment configuration file.
*   `--model_type`: Type of the model (e.g., `simplecnn`, `bert_classifier`).
*   `--dataset_type`: Type of the dataset (e.g., `cv`, `nlp`).
*   `--model_name`: Specific model name (e.g., `simplecnn`, `bert-base-uncased`).
*   `--dataset_name`: Specific dataset name (e.g., `cifar10`, `sst2`).
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Batch size for training and evaluation.
*   `--lr`: Learning rate.
*   `--save_path`: Path to save the best model.
*   `--use_cuda`: ( `true` | `false` ) Whether to use CUDA if available.
*   `--plot_curves`: ( `true` | `false` ) Whether to plot training curves (currently basic for CV).

Refer to `python train.py --help` for a full list of available arguments and their descriptions.

## Adding New Components

### Models
1.  Create your model file in the appropriate `models/<domain>/` directory (e.g., `models/vision/my_new_model.py`).
2.  Ensure your model loading function or class is imported in `models/<domain>/__init__.py`.
3.  Update `train.py` to recognize and instantiate your new model based on `--model_name` or a new `--model_type`.
4.  Add a corresponding YAML configuration in `config/experiments/`.

### Datasets
1.  Create your dataset loading script in `datasets/` (e.g., `datasets/my_new_dataset_loader.py`).
2.  Import your loading function in `datasets/__init__.py`.
3.  Update `train.py` to use your new dataset loader based on `--dataset_name`.
4.  Add a corresponding YAML configuration.

### Trainers
1.  If your new domain or model requires a significantly different training loop, create a new trainer class in `trainers/` (e.g., `trainers/my_new_trainer.py`).
2.  Import it in `trainers/__init__.py`.
3.  Update `train.py` to select and use your new trainer.

## Future Work / Contributions

This lab is continuously evolving. Future enhancements include:
*   Support for more model architectures (ViT, GANs, etc.).
*   Wider range of datasets for CV, NLP, and other domains.
*   Dedicated Reinforcement Learning (RL) and Generative Model pipelines.
*   Advanced logging with TensorBoard or Weights & Biases.
*   Hyperparameter sweeping capabilities.
*   More sophisticated evaluation and comparison tools.
*   Notebook examples for analysis and custom experiments.

Contributions are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.
