import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from cifar10_model import Net, load_trained_model # Net is implicitly used by load_trained_model
import json
import os
import pandas as pd
import torchvision # For loading CIFAR-10 testset

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def get_model():
    """Loads the pre-trained CIFAR-10 model."""
    return load_trained_model('cifar10_model.pth')

@st.cache_data
def get_transform():
    """Returns the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def predict_image(img: Image.Image, model, transform) -> tuple[str, float, np.ndarray]:
    """
    Predicts the class and probabilities for a given image.

    Args:
        img (PIL.Image.Image): The input image.
        model: The trained PyTorch model.
        transform: The torchvision transform to apply to the image.

    Returns:
        tuple: A tuple containing:
            - str: The predicted class name.
            - float: The confidence score for the predicted class.
            - np.ndarray: An array of probabilities for all classes.
    """
    img = transform(img).unsqueeze(0) # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
    return CLASSES[pred_idx], probs[pred_idx], probs

def display_prediction_probabilities(probs: np.ndarray):
    """Displays the prediction probabilities in a table and a bar chart."""
    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame({'Class': CLASSES, 'Probability': probs})
    prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
    st.dataframe(prob_df)
    st.bar_chart(prob_df.set_index('Class'))

st.title('CIFAR-10 AI Tester')

# Sidebar for general information (optional, can be moved to main page if preferred)
st.sidebar.title("About")
st.sidebar.info(
    "This application allows you to test a pre-trained CIFAR-10 image classifier. "
    "You can see predictions on random test images or upload your own."
)

model = get_model()
transform = get_transform()

tab1, tab2, tab3 = st.tabs(["Image Tester", "Training Insights", "AI Concepts"])

with tab1:
    st.header("Test the CIFAR-10 Model")
    st.write("Interact with the trained model by testing it on random images from the CIFAR-10 dataset or by uploading your own.")

    st.subheader('Test a Random CIFAR-10 Image')
    if st.button('Show Random Test Image', help="Loads a random image from the CIFAR-10 test set and classifies it."):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        idx = np.random.randint(0, len(testset))
        img, label = testset[idx] # img is already a PIL Image
        st.image(np.array(img), caption=f'Ground Truth: {CLASSES[label]}', width=128)
        pred, conf, probs = predict_image(img, model, transform)
        st.write(f'Prediction: **{pred}** (Confidence: {conf:.2f})')
        display_prediction_probabilities(probs)

    st.divider()

    st.subheader('Upload Your Own Image')
    uploaded = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'], help="Upload your own image (PNG, JPG, JPEG) to see the model's prediction.")
    if uploaded is not None:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption='Uploaded Image', width=128)
        pred, conf, probs = predict_image(img, model, transform)
        st.write(f'Prediction: **{pred}** (Confidence: {conf:.2f})')
        display_prediction_probabilities(probs)

with tab2:
    st.header("Model Training and Dataset Information")
    st.write("Explore details about how the model was trained, its performance, and the dataset it was trained on.")

    with st.expander("Model Training Insights", expanded=True):
        st.subheader("Training Performance")

        loss_plot_path = 'loss_plot.png'
        accuracy_plot_path = 'accuracy_plot.png'
        history_path = 'training_history.json'

        if os.path.exists(loss_plot_path) and os.path.exists(accuracy_plot_path):
            st.image(loss_plot_path, caption="Training and Validation Loss Plot")
            st.image(accuracy_plot_path, caption="Training and Validation Accuracy Plot")
        else:
            st.write("Training plot images not found. Please run the training script to generate them.")

        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)

            if history:
                st.subheader("Training Progress Chart")
                epochs = [epoch_data['epoch'] for epoch_data in history]
                train_loss = [epoch_data['train_loss'] for epoch_data in history]
                val_loss = [epoch_data['val_loss'] for epoch_data in history]

                loss_data = {
                    "Epoch": epochs,
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss
                }
                st.line_chart(loss_data, x="Epoch")

                total_epochs_num = len(history)
                st.write(f"Total epochs trained: {total_epochs_num}")

                best_val_acc = 0
                best_epoch = 0
                for epoch_data in history:
                    if epoch_data['val_acc'] > best_val_acc:
                        best_val_acc = epoch_data['val_acc']
                        best_epoch = epoch_data['epoch']

                st.write(f"Best validation accuracy: {best_val_acc:.4f} (achieved at epoch {best_epoch})")

                final_train_acc = history[-1]['train_acc']
                st.write(f"Final training accuracy: {final_train_acc:.4f}")
            else:
                st.write("Training history is empty. Run the training script to generate data.")
        else:
            st.write("Training history JSON (`training_history.json`) not found. Please run the training script.")

        st.subheader("Understanding Key Terms")
        st.markdown("""
        - **Epoch:** One complete pass through the entire training dataset.
        - **Loss:** A measure of how far off your model's predictions are from the actual target values. Lower is generally better.
        - **Accuracy:** The percentage of correct predictions made by the model. Higher is generally better.
        - **Training Set:** The data used to teach the model.
        - **Validation Set:** A separate portion of data used to evaluate the model's performance during training and tune hyperparameters, helping to prevent overfitting.
        - **Overfitting:** When a model learns the training data too well, including its noise, and performs poorly on new, unseen data.
        """)

    with st.expander("About the Model and Dataset", expanded=False):
        st.subheader("CIFAR-10 Dataset")
        st.write("The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.")
        st.write("More info: [CIFAR-10 Dataset Website](https://www.cs.toronto.edu/~kriz/cifar.html)")

        st.subheader("Convolutional Neural Network (CNN) Model")
        st.write("The model is a Convolutional Neural Network (CNN) with 3 convolutional layers followed by max pooling, and 2 fully connected layers. It's trained to classify images into one of the 10 CIFAR-10 categories.")

with tab3:
    st.header("Deeper Dive into AI Concepts")
    st.write("Learn more about how AI models like this one work and some of their current limitations.")

    with st.expander("How AI 'Sees' Images (Interpretability)", expanded=False):
        st.markdown("""
        Convolutional Neural Networks (CNNs) like the one used here learn to identify images by recognizing patterns hierarchically.
        - Initially, they might learn to detect simple features like **edges and corners**.
        - Subsequent layers combine these to recognize more complex **textures and parts of objects** (e.g., a wheel, an ear).
        - Finally, deeper layers assemble these parts to identify **whole objects** (e.g., an automobile, a dog).

        Understanding exactly *why* a model makes a specific decision for a particular image is a challenging but important area of research known as **interpretability** or **Explainable AI (XAI)**.

        Researchers have developed various techniques to peek inside the "black box" of these models. Some examples include:
        - **Saliency Maps:** Highlight the pixels in an image that were most influential for the model's prediction.
        - **Class Activation Maps (CAM):** Show which regions of an image are most indicative of a particular class.
        - **SHAP (SHapley Additive exPlanations) Values:** Assign an importance value to each feature for a given prediction.

        These are advanced topics, but they are crucial for building trust and understanding in AI systems.
        """)

    with st.expander("Model Limitations: Robustness and Adversarial Examples", expanded=False):
        st.markdown("""
        AI models, including powerful image classifiers, can sometimes be surprisingly fragile.

        - **Sensitivity to Small Changes:** Models can be sensitive to tiny changes in input data that are often imperceptible to humans.
        - **Adversarial Examples:** These are specially crafted inputs designed to fool a model. For instance, adding a very subtle layer of noise to an image might cause the model to misclassify it with high confidence (e.g., classifying a picture of a cat as an airplane).

        The existence of adversarial examples highlights important challenges in AI:
        - **Robustness:** How well does a model maintain its performance when faced with unexpected or slightly altered inputs?
        - **Security:** Adversarial attacks could potentially be used to compromise AI systems in critical applications.

        Research into making models more robust to such manipulations is a key focus in ensuring AI safety and reliability.

        To learn more, you can check out resources like:
        - [OpenAI on Adversarial Examples](https://openai.com/blog/adversarial-examples/)
        - [Google AI Blog: Understanding Adversarial Examples](https://ai.googleblog.com/search/label/Adversarial%20Examples)
        (Note: You may need to search for specific articles on the Google AI blog if the direct link changes.)
        """)
