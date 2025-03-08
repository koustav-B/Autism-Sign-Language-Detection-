import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tempfile

# ====== CONFIGURATION ======
IMAGE_SIZE = (224, 224)  # Input size for MobileNetV2
TFLITE_MODEL_PATH = "sign_language_model.tflite"  # Your trained model

# Class Labels (Update as per your dataset)
CLASS_LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

# ====== LOAD THE TFLITE MODEL ======
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    """Loads an image, resizes it, and normalizes it for prediction."""
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Error: Unable to load image at {image_path}")
        return None

    image = cv2.resize(image, IMAGE_SIZE)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    return image.astype(np.float32)

def plot_probabilities(probabilities):
    """Plots the probability distribution as a bar graph."""
    plt.figure(figsize=(10, 5))
    labels = [CLASS_LABELS[i] for i in range(len(probabilities))]
    plt.bar(labels, probabilities * 100, color='blue')
    plt.xlabel("Alphabets")
    plt.ylabel("Probability (%)")
    plt.title("Sign Language Prediction Confidence")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Set y-axis limit to 100%

    # Highlight the highest probability in red
    max_index = np.argmax(probabilities)
    plt.bar(labels[max_index], probabilities[max_index] * 100, color='red', label="Predicted Alphabet")
    
    plt.legend()
    st.pyplot(plt)  # Show the plot in Streamlit

def predict_sign(image_path):
    """Runs inference and predicts the sign language alphabet with probabilities."""
    image = preprocess_image(image_path)
    if image is None:
        return None, None

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get top prediction
    predicted_index = np.argmax(output_data)
    predicted_label = CLASS_LABELS.get(predicted_index, "Unknown")
    confidence = output_data[0][predicted_index] * 100  # Convert to percentage

    return predicted_label, output_data[0]

# ====== STREAMLIT UI ======
st.title("🤟 Sign Language Alphabet Prediction")
st.write("Upload an image of a hand sign to predict the corresponding alphabet.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    predicted_label, probabilities = predict_sign(temp_path)

    if predicted_label:
        st.success(f"🅿️ **Predicted Alphabet: {predicted_label}**")
        plot_probabilities(probabilities)
