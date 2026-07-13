import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")

IMG_SIZE = 224

@st.cache_resource
def load_model(model_name):
    if model_name == "Custom CNN":
        return tf.keras.models.load_model("custom_cnn.h5")
    elif model_name == "Hybrid CNN (CBAM)":
        return tf.keras.models.load_model("hybrid_cnn_cbam.h5")
    elif model_name == "ResNet50":
        return tf.keras.models.load_model("resnet50.h5")

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
    if model_name == "Custom CNN":
        return "conv2d_3"
    elif model_name == "Hybrid CNN (CBAM)":
        return "conv2d_7"  
    elif model_name == "ResNet50":
        return "conv5_block3_out"
    return None


st.title("🫁 Pneumonia Detection — 3 Model Comparison")
st.write("Upload a chest X-ray and compare predictions across Custom CNN, CBAM, and ResNet50.")

st.sidebar.header("⚙️ Select Models")
selected_models = st.sidebar.multiselect(
    "Choose which models to run:",
    ["Custom CNN", "Hybrid CNN (CBAM)", "ResNet50"],
    default=["Custom CNN"]
)

alpha = st.sidebar.slider("Heatmap transparency", 0.0, 1.0, 0.4)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📷 Original")
        st.image(image, use_container_width=True)
    
    img_batch, img_array = preprocess_image(image)
    
    results = []
    for model_name in selected_models:
        with st.spinner(f"Running {model_name}..."):
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
                    "overlay": overlay_rgb
                })
            except Exception as e:
                st.error(f"{model_name} failed: {e}")
    
    with col2:
        st.subheader("📊 Predictions")
        for res in results:
            color = "🔴" if res["prediction"] == "PNEUMONIA" else "🟢"
            st.write(f"**{color} {res['model']}:** {res['prediction']} ({res['confidence']:.1%})")

    if results:
        st.subheader("🔥 Grad-CAM Comparison")
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                st.markdown(f"**{res['model']}**")
                st.image(res["overlay"], use_container_width=True)
      
        if len(results) >= 2:
            pneumonia_count = sum(1 for r in results if r["prediction"] == "PNEUMONIA")
            normal_count = len(results) - pneumonia_count
            st.subheader("🗳️ Consensus")
            if pneumonia_count > normal_count:
                st.error(f"Majority says **PNEUMONIA** ({pneumonia_count}/{len(results)})")
            elif normal_count > pneumonia_count:
                st.success(f"Majority says **NORMAL** ({normal_count}/{len(results)})")
            else:
                st.warning("Split decision!")
    
    st.warning("⚠️ For educational use only. Not a medical diagnosis.")
