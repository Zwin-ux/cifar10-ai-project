import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from cifar10_model import Net, load_trained_model

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def get_model():
    return load_trained_model('cifar10_model.pth')

@st.cache_data
def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def predict_image(img, model, transform):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
    return CLASSES[pred_idx], probs[pred_idx], probs

st.title('CIFAR-10 AI Tester')

model = get_model()
transform = get_transform()

st.header('Test a Random CIFAR-10 Image')
if st.button('Show Random Test Image'):
    import torchvision
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    idx = np.random.randint(0, len(testset))
    img, label = testset[idx]
    st.image(np.array(img), caption=f'Ground Truth: {CLASSES[label]}', width=128)
    pred, conf, probs = predict_image(img, model, transform)
    st.write(f'Prediction: **{pred}** (Confidence: {conf:.2f})')
    st.bar_chart(probs)

st.header('Upload Your Own Image')
uploaded = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded Image', width=128)
    pred, conf, probs = predict_image(img, model, transform)
    st.write(f'Prediction: **{pred}** (Confidence: {conf:.2f})')
    st.bar_chart(probs)
