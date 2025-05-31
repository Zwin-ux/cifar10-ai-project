import torch
import os

def save_model(model, path, filename="best_model.pth"):
    '''Saves the model state_dict to the specified path and filename.'''
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")

def load_model(model, path, filename="best_model.pth", device='cpu'):
    '''Loads the model state_dict from the specified path and filename.'''
    full_path = os.path.join(path, filename)
    if os.path.exists(full_path):
        model.load_state_dict(torch.load(full_path, map_location=torch.device(device)))
        model.eval() # Set to evaluation mode
        print(f"Model loaded from {full_path}")
        return model
    else:
        print(f"No model found at {full_path}")
        return None
