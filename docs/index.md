# CIFAR-10 AI Project

Welcome to the CIFAR-10 AI project! This site provides a quick overview, setup instructions, and links to the code.

## Contents

- Overview: What this repository contains and what it does
- Usage: How to install dependencies, train the model, and run the demo
- Model: Architecture details and design choices
- Results: How to evaluate and view metrics

## Quick links

- Source repository: see project on GitHub
- Training script: `cifar10_train.py`
- Inference model: `cifar10_model.py`
- Local demo: `streamlit_app.py`

## What is CIFAR-10?

CIFAR-10 is a dataset with 60,000 32x32 color images in 10 classes. More info: <https://www.cs.toronto.edu/~kriz/cifar.html>

## Highlights

- Simple CNN with 3 conv blocks and 2 FC layers (`cifar10_model.py`)
- Training with validation, early stopping, scheduler, and saved best weights (`cifar10_train.py`)
- Local interactive demo via Streamlit (`streamlit_app.py`)

## Continue to

- [Usage](usage.md)
- [Model](model.md)
- [Results](results.md)

## Live Demo

This project includes a local Streamlit app (`streamlit_app.py`). To host it online, deploy to Streamlit Community Cloud (free for small apps):

1. Push this repo to GitHub
2. Go to Streamlit Community Cloud and create a new app from your repo
3. Set the entrypoint to `streamlit_app.py`

Once deployed, add your app URL here:

- Live demo: <https://your-app.streamlit.app>
