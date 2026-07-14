import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Pneumonia Detection | Medical Imaging AI",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E1E1E; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #808495; margin-bottom: 2rem; }
    .prediction-box { border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0; }
    .prediction-normal { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 2px solid #28a745; }
    .prediction-pneumonia { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 2px solid #dc3545; }
    .confidence-bar { height: 8px; border-radius: 4px; background: #e9ecef; overflow: hidden; }
    .confidence-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224
REPO_ID = "ananaya01/pneumonia-detector"

# ============================================================
# LOAD MODEL (only Custom CNN - smallest model)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Downloading model... This may take a minute."):
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="custom_cnn.h5",
            repo_type="model"
        )
    return tf.keras.models.load_model(model_path)


# ============================================================
# GRAD-CAM
# ============================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    last_conv_layer = model.get_layer(last_conv_layer_name)

    @tf.function
    def compute_gradcam(inputs):
        with tf.GradientTape() as tape:
            x = inputs
            conv_output = None
            for layer in model.layers:
                x = layer(x)
                if layer == last_conv_layer:
                    tape.watch(x)
                    conv_output = x
            loss = x[0][0]
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
        return heatmap

    return compute_gradcam(img_array).numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)


def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img_array


# ============================================================
# MAIN APP
# ============================================================
st.markdown('<div class="main-header">🫁 Pneumonia Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered chest X-ray analysis using Custom CNN with Grad-CAM visualization</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    alpha = st.slider("Heatmap overlay intensity", 0.0, 1.0, 0.4)
    st.markdown("---")
    st.markdown("**Model:** Custom CNN")
    st.markdown("- 4 conv blocks (32→64→128→256)")
    st.markdown("- Trained on Chest X-Ray Pneumonia dataset")
    st.info("Model loaded from Hugging Face Hub")

st.subheader("Upload Chest X-Ray")
uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("**Original X-Ray**")
        st.image(image, use_container_width=True)
        st.caption(f"Format: {image.format} | Size: {image.size[0]}x{image.size[1]}")

    img_batch, img_array = preprocess_image(image)

    with st.spinner("Analyzing..."):
        try:
            model = load_model()
            prediction = model.predict(img_batch, verbose=0)[0][0]
            pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            # Grad-CAM
            heatmap = make_gradcam_heatmap(img_batch, model, "conv2d_3")
            overlay = overlay_heatmap(img_array, heatmap, alpha=alpha)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Analysis Result")

                is_pneumonia = pred_class == "PNEUMONIA"
                box_class = "prediction-pneumonia" if is_pneumonia else "prediction-normal"
                icon = "⚠️" if is_pneumonia else "✅"
                color = "#dc3545" if is_pneumonia else "#28a745"

                html = f'<div class="prediction-box {box_class}">'
                html += f'<div style="font-size: 1.5rem; font-weight: 700; color: {color};">{icon} {pred_class}</div>'
                html += '<div style="margin-top: 1rem;">'
                html += '<div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">'
                html += f'<span style="font-size: 0.9rem;">Confidence</span><span style="font-weight: 600;">{confidence:.1%}</span></div>'
                html += '<div class="confidence-bar">'
                html += f'<div class="confidence-fill" style="width: {confidence*100}%; background: {color};"></div></div></div></div>'

                st.markdown(html, unsafe_allow_html=True)
                st.caption(f"Raw score: {prediction:.4f}")

            # Grad-CAM
            st.markdown("---")
            st.subheader("🔥 Grad-CAM Visualization")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Heatmap**")
                st.image(heatmap, use_container_width=True, clamp=True)
            with c2:
                st.markdown("**Overlay**")
                st.image(overlay_rgb, use_container_width=True)

            st.info("Red/yellow areas show where the AI focused its attention.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please try refreshing the page. If the error persists, the model may still be downloading.")

    st.markdown("---")
    st.warning("⚠️ Medical Disclaimer: This tool is for educational purposes only. NOT a substitute for professional medical diagnosis.")

else:
    st.info("👆 Upload a chest X-ray image above to begin analysis")
    st.markdown("---")
    st.subheader("How it works")
    step_cols = st.columns(3)
    steps = [("1", "Upload", "Upload X-ray image"), ("2", "Analyze", "AI model processes the image"), ("3", "Visualize", "See prediction + Grad-CAM heatmap")]
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(f"**Step {num}: {title}**")
            st.caption(desc)
