import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
from cifar10_model import Net, load_trained_model

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = load_trained_model('cifar10_best_model.pth')
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
    return CLASSES[pred_idx], probs[pred_idx], probs

def show_random():
    import torchvision
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    idx = np.random.randint(0, len(testset))
    img, label = testset[idx]
    display_image(img)
    pred, conf, probs = predict_image(img)
    result_var.set(f'Prediction: {pred} (Confidence: {conf:.2f})\nGround Truth: {CLASSES[label]}')

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])
    if file_path:
        img = Image.open(file_path).convert('RGB')
        display_image(img)
        pred, conf, probs = predict_image(img)
        result_var.set(f'Prediction: {pred} (Confidence: {conf:.2f})')

def display_image(img):
    img_disp = img.resize((128, 128))
    img_tk = ImageTk.PhotoImage(img_disp)
    panel.config(image=img_tk)
    panel.image = img_tk

root = tk.Tk()
root.title('CIFAR-10 Image Classifier')

panel = tk.Label(root)
panel.pack(pady=10)

btn_random = tk.Button(root, text='Show Random Test Image', command=show_random)
btn_random.pack(pady=5)

btn_upload = tk.Button(root, text='Upload Your Own Image', command=upload_image)
btn_upload.pack(pady=5)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=('Arial', 12))
result_label.pack(pady=10)

root.mainloop()
