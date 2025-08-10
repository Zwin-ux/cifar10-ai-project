# Usage


## Setup

 
- Python 3.9+
- Install deps:

```bash
pip install -r requirements.txt
```

 
## Train

 
```bash
python cifar10_train.py
```

 
- Downloads CIFAR-10 automatically to `./data/`
- Trains with train/val split, LR scheduler, early stopping
- Saves best weights to `cifar10_best_model.pth`

 
## Test locally (Streamlit)

 
```bash
python -m streamlit run streamlit_app.py
```

 
- Try a random CIFAR-10 test image
- Upload your own image (auto-resized to 32x32)

 
## Project layout

 
- `cifar10_model.py`: Inference model and loader
- `cifar10_train.py`: Training loop, early stopping, metrics and plots
- `streamlit_app.py`: Interactive local demo
- `requirements.txt`: Dependencies
- `docs/`: Static docs for GitHub Pages
