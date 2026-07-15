import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(
    page_title="Pneumonia Detection with Grad-CAM",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray image to detect pneumonia with explainable AI (Grad-CAM)")

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    st.markdown("""
    **📊 Model Information:**
    - Model: Custom CNN
    - Task: Binary Classification
    - Classes: Normal / Pneumonia
    
    **🔍 Grad-CAM:**
    - Shows which regions the model focused on
    - Red = Important for prediction
    """)
    st.markdown("---")
    st.caption("Made with ❤️ for Internship Project")

if 'heatmap' not in st.session_state:
    st.session_state.heatmap = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("custom_cnn.h5")
    return model

def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_3"):
    # Create conv model
    conv_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(last_conv_layer_name).output
    )
    
    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        predictions = model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()[0]

def main():
    model = load_model()
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="Uploaded Chest X-Ray", use_column_width=True)
        
        image = Image.open(uploaded_file)
        img_batch, img_array = preprocess_image(image)
      
        prediction = model.predict(img_batch, verbose=0)[0][0]
        pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction
    
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Prediction",
                value=pred_class,
                delta=f"{confidence:.1%} confidence",
                delta_color="normal" if pred_class == "NORMAL" else "inverse"
            )
       
        with st.spinner("Generating Grad-CAM visualization..."):
            heatmap = make_gradcam_heatmap(img_batch, model)
    
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (224, 224))
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), 0.6, heatmap_colored, 0.4, 0)
   
        st.markdown("---")
        st.subheader("📊 Grad-CAM Visualization")
        st.markdown("*Red/Yellow regions show where the model focused to make its prediction*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image, caption="Original X-Ray", use_column_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(heatmap, cmap='jet')
            ax.axis('off')
            ax.set_title("Grad-CAM Heatmap")
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.image(overlay, caption=f"{pred_class} ({confidence:.1%})", use_column_width=True)
  
        st.markdown("---")
        with st.expander("ℹ️ How Grad-CAM Works"):
            st.markdown("""
            **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows which parts of the X-ray image the model considered most important for its prediction.
            
            - 🔴 **Red regions**: Highly important for the prediction
            - 🟡 **Yellow regions**: Moderately important
            - 🔵 **Blue regions**: Less important
            
            For pneumonia detection, the model typically focuses on:
            - **Pneumonia**: Hazy white patches (opacities) in the lungs
            - **Normal**: Diffuse patterns (no specific focus)
            """)
  
        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. 
        It is NOT approved for clinical diagnosis. Always consult a qualified medical professional.
        """)

if __name__ == "__main__":
    main()
