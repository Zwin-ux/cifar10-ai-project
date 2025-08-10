# CIFAR-10 AI Project

Simple CNN baseline for CIFAR-10 with a training pipeline, saved best checkpoint, and an interactive Streamlit demo. Includes a GitHub Pages docs site under `docs/`.

## What is CIFAR-10?
The CIFAR-10 dataset contains 60,000 32x32 color images across 10 classes. More info: <https://www.cs.toronto.edu/~kriz/cifar.html>

## Features
- Simple CNN with 3 conv blocks and 2 FC layers (`cifar10_model.py`)
- Training with validation split, LR scheduler, early stopping, and best-checkpoint saving (`cifar10_train.py`)
- Local demo via Streamlit to classify test or uploaded images (`streamlit_app.py`)
- Static documentation site in `docs/` for GitHub Pages

## Quickstart
1. Create environment and install dependencies

```bash
pip install -r requirements.txt
```

2. Train

```bash
python cifar10_train.py
```

This downloads CIFAR-10 to `./data/` and saves the best model to `cifar10_best_model.pth`.

3. Run demo (local)

```bash
python -m streamlit run streamlit_app.py
```

Load a random CIFAR-10 test image or upload your own (auto-resized to 32x32). The app loads `cifar10_best_model.pth`.

4. Optional desktop GUI (Tkinter)

```bash
python tkinter_gui.py
```

## GitHub Pages (Docs)
This repo includes a simple docs site in `docs/` using a GitHub Pages theme.

To publish:
- Push this repository to GitHub
- In your GitHub repo: Settings → Pages → Source = "Deploy from a branch"
- Branch: `main` (or your default), Folder: `/docs`
- Save. Your site will build and be available at `https://<username>.github.io/<repo>/`

Docs entrypoint: `docs/index.md` with theme config in `docs/_config.yml`.

## Project structure
```text
.
├─ cifar10_model.py        # Model and load_trained_model()
├─ cifar10_train.py        # Training loop with early stopping, scheduler, plots
├─ streamlit_app.py        # Interactive local demo
├─ tkinter_gui.py          # Optional desktop GUI
├─ requirements.txt        # Dependencies
└─ docs/                   # Static docs for GitHub Pages
   ├─ index.md
   ├─ usage.md
   ├─ model.md
   ├─ results.md
   └─ _config.yml
```

## Dependencies
- torch, torchvision, numpy, matplotlib, Pillow, streamlit

## Acknowledgments
- CIFAR-10: <https://www.cs.toronto.edu/~kriz/cifar.html>
