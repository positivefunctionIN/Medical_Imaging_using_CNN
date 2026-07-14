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
    layout="wide",
    initial_sidebar_state="expanded"
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
    .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e9ecef; text-align: center; color: #808495; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224
REPO_ID = "ananaya01/pneumonia-detector"

# ============================================================
# CUSTOM CBAMBlock LAYER
# ============================================================
class CBAMBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.dense1 = tf.keras.layers.Dense(channels // self.ratio, activation='relu')
        self.dense2 = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        self.multiply_channel = tf.keras.layers.Multiply()
        self.multiply_spatial = tf.keras.layers.Multiply()
        super(CBAMBlock, self).build(input_shape)

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)
        avg_out = self.dense2(self.dense1(avg_pool))
        max_out = self.dense2(self.dense1(max_pool))
        channel_attention = tf.keras.layers.Add()([avg_out, max_out])
        channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)
        refined_inputs = self.multiply_channel([inputs, channel_attention])
        avg_pool_spatial = tf.reduce_mean(refined_inputs, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(refined_inputs, axis=-1, keepdims=True)
        spatial_concat = self.concat([avg_pool_spatial, max_pool_spatial])
        spatial_attention = self.conv(spatial_concat)
        refined_output = self.multiply_spatial([refined_inputs, spatial_attention])
        return refined_output

    def get_config(self):
        config = super(CBAMBlock, self).get_config()
        config.update({"ratio": self.ratio})
        return config


# ============================================================
# DOWNLOAD MODELS FROM HUGGING FACE HUB
# ============================================================
@st.cache_resource(show_spinner=False)
def download_model(model_name):
    """Download model from Hugging Face Hub."""
    model_files = {
        "Custom CNN": "custom_cnn.h5",
        "Hybrid CNN (CBAM)": "hybrid_cnn_cbam.h5",
        "ResNet50": "resnet50.h5"
    }
    filename = model_files[model_name]

    with st.spinner(f"Downloading {model_name} model..."):
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="model"
        )
    return model_path


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    """Load the selected model."""
    custom_objects = {"CBAMBlock": CBAMBlock}
    model_path = download_model(model_name)

    if model_name == "Custom CNN":
        return tf.keras.models.load_model(model_path)
    elif model_name == "Hybrid CNN (CBAM)":
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    elif model_name == "ResNet50":
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")


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


def get_last_conv_layer(model_name):
    if model_name in ["Custom CNN", "Hybrid CNN (CBAM)"]:
        return "conv2d_3"
    elif model_name == "ResNet50":
        return "conv5_block3_out"
    return None


# ============================================================
# MAIN APP
# ============================================================
st.markdown('<div class="main-header">🫁 Pneumonia Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered chest X-ray analysis using three deep learning models with Grad-CAM visualization</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    selected_models = st.multiselect(
        "Select models to compare:",
        ["Custom CNN", "Hybrid CNN (CBAM)", "ResNet50"],
        default=["Custom CNN"]
    )
    alpha = st.slider("Heatmap overlay intensity", 0.0, 1.0, 0.4)
    st.markdown("---")
    st.markdown("**Model Details:**")
    st.markdown("- Custom CNN: 4 conv blocks")
    st.markdown("- Hybrid CNN (CBAM): + Attention")
    st.markdown("- ResNet50: Pre-trained, fine-tuned")
    st.info("Models loaded from Hugging Face Hub")

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
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, model_name in enumerate(selected_models):
        status_text.text(f"Analyzing with {model_name}... ({idx+1}/{len(selected_models)})")
        progress_bar.progress((idx + 1) / len(selected_models))
        try:
            model = load_model(model_name)
            prediction = model.predict(img_batch, verbose=0)[0][0]
            pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            last_conv = get_last_conv_layer(model_name)
            heatmap = make_gradcam_heatmap(img_batch, model, last_conv)
            overlay = overlay_heatmap(img_array, heatmap, alpha=alpha)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            results.append({
                "model": model_name,
                "prediction": pred_class,
                "confidence": confidence,
                "raw_score": prediction,
                "heatmap": heatmap,
                "overlay": overlay_rgb
            })
        except Exception as e:
            st.error(f"Error: {model_name} failed: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    with col2:
        st.subheader("Analysis Results")
        for res in results:
            is_pneumonia = res["prediction"] == "PNEUMONIA"
            box_class = "prediction-pneumonia" if is_pneumonia else "prediction-normal"
            icon = "WARNING" if is_pneumonia else "OK"
            color = "#dc3545" if is_pneumonia else "#28a745"

            html = f'<div class="prediction-box {box_class}">'
            html += f'<div style="font-size: 1.3rem; font-weight: 700; color: {color};">{icon} {res["prediction"]}</div>'
            html += f'<div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">{res["model"]}</div>'
            html += '<div style="margin-top: 1rem;">'
            html += '<div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">'
            html += f'<span style="font-size: 0.85rem;">Confidence</span><span style="font-weight: 600;">{res["confidence"]:.1%}</span></div>'
            html += '<div class="confidence-bar">'
            html += f'<div class="confidence-fill" style="width: {res["confidence"]*100}%; background: {color};"></div></div></div></div>'

            st.markdown(html, unsafe_allow_html=True)

    if results:
        st.markdown("---")
        st.subheader("Grad-CAM Visualizations")
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                st.markdown(f"**{res['model']}**")
                tab1, tab2 = st.tabs(["Overlay", "Heatmap"])
                with tab1:
                    st.image(res["overlay"], use_container_width=True)
                with tab2:
                    st.image(res["heatmap"], use_container_width=True, clamp=True)
                st.caption(f"Score: {res['raw_score']:.4f}")

        if len(results) >= 2:
            st.markdown("---")
            st.subheader("Model Consensus")
            pneumonia_votes = sum(1 for r in results if r["prediction"] == "PNEUMONIA")
            normal_votes = len(results) - pneumonia_votes
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Models Agree", f"{max(pneumonia_votes, normal_votes)}/{len(results)}")
            with c2:
                if pneumonia_votes > normal_votes:
                    st.error("PNEUMONIA detected by majority")
                elif normal_votes > pneumonia_votes:
                    st.success("NORMAL detected by majority")
                else:
                    st.warning("Split decision - no consensus")
            with c3:
                avg_confidence = np.mean([r["confidence"] for r in results])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    st.markdown("---")
    st.warning("Medical Disclaimer: This tool is for educational purposes only. NOT a substitute for professional medical diagnosis.")

else:
    st.info("Upload a chest X-ray image above to begin analysis")
    st.markdown("---")
    st.subheader("How it works")
    step_cols = st.columns(4)
    steps = [("1", "Upload", "Upload X-ray"), ("2", "Analyze", "AI models analyze"), ("3", "Visualize", "Grad-CAM heatmap"), ("4", "Compare", "Compare models")]
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(f"**Step {num}: {title}**")
            st.caption(desc)

st.markdown('<div class="footer"><p>Built with TensorFlow & Streamlit | Models hosted on Hugging Face Hub</p></div>', unsafe_allow_html=True)
