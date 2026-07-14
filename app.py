# ============================================================
# PneumoVision AI
# Multi Model Pneumonia Detection System
# ============================================================


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests

from PIL import Image
from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PneumoVision AI",
    page_icon="🫁",
    layout="wide"
)



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""
<style>

.main{
    background-color:#f8fafc;
}


h1{
    color:#0f172a;
}


.card{

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.08);

}


.result{

font-size:28px;
font-weight:bold;

}


</style>

""",
unsafe_allow_html=True
)



# ============================================================
# TITLE
# ============================================================


st.title("🫁 PneumoVision AI")

st.caption(
"Explainable Deep Learning System for Chest X-Ray Pneumonia Detection"
)



# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.title("⚙️ About System")


st.sidebar.info(
"""
Models Used:

🧠 Custom CNN

🧬 Hybrid CNN + CBAM

🚀 ResNet50


Framework:

TensorFlow

Streamlit

OpenCV


Purpose:

Educational and Research Demo

"""
)



# ============================================================
# HUGGING FACE MODEL URLS
# ============================================================


CUSTOM_URL = (
"https://huggingface.co/ananaya01/"
"pneumonia-detector/resolve/main/custom_cnn.h5"
)


HYBRID_URL = (
"https://huggingface.co/ananaya01/"
"pneumonia-detector/resolve/main/hybrid_cnn_cbam.h5"
)


RESNET_URL = (
"https://huggingface.co/ananaya01/"
"pneumonia-detector/resolve/main/resnet50.h5"
)



# ============================================================
# DOWNLOAD FUNCTION
# ============================================================


def download_model(url, filename):


    if not os.path.exists(filename):

        with st.spinner(
            f"Downloading {filename}..."
        ):

            response = requests.get(
                url
            )

            response.raise_for_status()


            with open(filename,"wb") as f:

                f.write(
                    response.content
                )


    return filename




# ============================================================
# LOAD MODELS
# ============================================================


@st.cache_resource
def load_models():


    custom_path = download_model(
        CUSTOM_URL,
        "custom_cnn.h5"
    )


    hybrid_path = download_model(
        HYBRID_URL,
        "hybrid_cnn_cbam.h5"
    )


    resnet_path = download_model(
        RESNET_URL,
        "resnet50.h5"
    )


    custom_model = tf.keras.models.load_model(
        custom_path,
        compile=False
    )


    hybrid_model = tf.keras.models.load_model(
        hybrid_path,
        compile=False
    )


    resnet_model = tf.keras.models.load_model(
        resnet_path,
        compile=False
    )


    return (
        custom_model,
        hybrid_model,
        resnet_model
    )



# ============================================================
# LOAD MODELS
# ============================================================


with st.spinner(
"Loading AI Models..."
):

    custom_model, hybrid_model, resnet_model = load_models()



st.success(
"✅ All AI Models Loaded Successfully"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================


IMG_SIZE = 224


def preprocess_image(image):

    img = image.convert("RGB")

    img = img.resize(
        (IMG_SIZE, IMG_SIZE)
    )


    img_array = np.array(img)


    img_array = img_array / 255.0


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    return img_array





# ============================================================
# PREDICTION FUNCTION
# ============================================================


def predict_model(model, img_array):

    prediction = model.predict(
        img_array,
        verbose=0
    )


    confidence = float(
        prediction[0][0]
    )


    if confidence >= 0.5:

        label = "PNEUMONIA"

        score = confidence


    else:

        label = "NORMAL"

        score = 1-confidence



    return label, score





# ============================================================
# FINAL ENSEMBLE PREDICTION
# ============================================================


def final_prediction(results):


    pneumonia_scores = []


    for item in results:


        if item["Prediction"] == "PNEUMONIA":

            pneumonia_scores.append(
                item["Confidence"]
            )

        else:

            pneumonia_scores.append(
                1-item["Confidence"]
            )



    avg_score = np.mean(
        pneumonia_scores
    )


    if avg_score >= 0.5:

        return (
            "PNEUMONIA",
            avg_score
        )


    else:

        return (
            "NORMAL",
            1-avg_score
        )





# ============================================================
# UPLOAD SECTION
# ============================================================


st.divider()


st.subheader(
"📤 Upload Chest X-Ray"
)



uploaded_file = st.file_uploader(
    "Choose X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)





if uploaded_file is not None:


    image = Image.open(
        uploaded_file
    )


    col1, col2 = st.columns(2)



    with col1:

        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )



    with col2:

        st.markdown(
        """
        ### Image Information

        AI will analyze:

        ✔ Lung patterns

        ✔ Infection signs

        ✔ Abnormal opacity

        ✔ Pneumonia probability

        """
        )



    img_array = preprocess_image(
        image
    )



    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================


    with st.spinner(
        "🧠 Analyzing X-Ray..."
    ):


        custom_label, custom_score = predict_model(
            custom_model,
            img_array
        )


        hybrid_label, hybrid_score = predict_model(
            hybrid_model,
            img_array
        )


        resnet_label, resnet_score = predict_model(
            resnet_model,
            img_array
        )



    results = [

        {
            "Model":"Custom CNN",
            "Prediction":custom_label,
            "Confidence":custom_score
        },


        {
            "Model":"Hybrid CNN + CBAM",
            "Prediction":hybrid_label,
            "Confidence":hybrid_score
        },


        {
            "Model":"ResNet50",
            "Prediction":resnet_label,
            "Confidence":resnet_score
        }

    ]



    final, final_confidence = final_prediction(
        results
    )





    # ========================================================
    # FINAL RESULT
    # ========================================================


    st.divider()

    st.subheader(
        "🩺 Final AI Diagnosis"
    )



    if final == "PNEUMONIA":

        st.error(
            f"""
            ## 🦠 Pneumonia Detected

            Confidence:
            {final_confidence*100:.2f}%
            """
        )


    else:

        st.success(
            f"""
            ## ✅ Normal X-Ray

            Confidence:
            {final_confidence*100:.2f}%
            """
        )





    # ========================================================
    # MODEL COMPARISON
    # ========================================================


    st.divider()


    st.subheader(
        "📊 Model Confidence Comparison"
    )


    for result in results:


        st.write(
            f"### {result['Model']}"
        )


        st.progress(
            float(result["Confidence"])
        )


        st.write(
            f"""
            Prediction:
            **{result['Prediction']}**

            Confidence:
            **{result['Confidence']*100:.2f}%**
            """
        )


        st.divider()

# ============================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================


def find_last_conv_layer(model):
    """
    Automatically finds the last Conv2D layer.
    Works with nested models like ResNet50.
    """

    conv_layers = []


    def search_layers(layer):

        if hasattr(layer, "layers"):

            for sub_layer in layer.layers:

                if isinstance(
                    sub_layer,
                    tf.keras.layers.Conv2D
                ):

                    conv_layers.append(
                        sub_layer
                    )


                elif hasattr(
                    sub_layer,
                    "layers"
                ):

                    search_layers(
                        sub_layer
                    )


    search_layers(model)


    if len(conv_layers) == 0:

        raise Exception(
            "No convolution layer found"
        )


    return conv_layers[-1]





def make_gradcam_heatmap(
        img_array,
        model
):


    last_conv_layer = find_last_conv_layer(
        model
    )


    grad_model = tf.keras.models.Model(

        inputs=model.inputs,

        outputs=[
            last_conv_layer.output,
            model.output
        ]

    )


    with tf.GradientTape() as tape:


        conv_outputs, predictions = grad_model(
            img_array
        )


        loss = predictions[:,0]



    gradients = tape.gradient(
        loss,
        conv_outputs
    )



    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0,1,2)
    )



    conv_outputs = conv_outputs[0]



    heatmap = conv_outputs @ pooled_gradients[...,None]



    heatmap = tf.squeeze(
        heatmap
    )



    heatmap = tf.maximum(
        heatmap,
        0
    )



    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        +
        1e-10
    )



    return heatmap.numpy()





def overlay_gradcam(
        image,
        heatmap
):


    img = np.array(
        image.convert("RGB")
    )


    heatmap = cv2.resize(

        heatmap,

        (
            img.shape[1],
            img.shape[0]
        )

    )



    heatmap = np.uint8(
        255 * heatmap
    )



    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )



    overlay = cv2.addWeighted(

        img,

        0.6,

        heatmap,

        0.4,

        0

    )


    return overlay





# ============================================================
# DISPLAY GRAD-CAM
# ============================================================


if uploaded_file is not None:


    st.divider()


    st.subheader(
        "🔥 Explainable AI - Grad-CAM"
    )


    st.write(
    """
    Grad-CAM highlights the regions
    of the X-ray that influenced
    the AI prediction.
    """
    )



    gradcam_models = [

        (
            "Custom CNN",
            custom_model
        ),

        (
            "Hybrid CNN + CBAM",
            hybrid_model
        ),

        (
            "ResNet50",
            resnet_model
        )

    ]



    cam_columns = st.columns(3)



    for index, (name, model) in enumerate(
        gradcam_models
    ):


        try:


            heatmap = make_gradcam_heatmap(

                img_array,

                model

            )



            cam_image = overlay_gradcam(

                image,

                heatmap

            )



            with cam_columns[index]:


                st.image(

                    cam_image,

                    caption=name,

                    use_container_width=True

                )



        except Exception as e:


            with cam_columns[index]:


                st.warning(

                    f"""
                    Grad-CAM unavailable

                    {name}

                    Error:
                    {e}
                    """

                )

# ============================================================
# PDF REPORT GENERATION
# ============================================================


from reportlab.pdfgen import canvas
from datetime import datetime



def create_pdf_report(
        diagnosis,
        confidence,
        results
):

    buffer = BytesIO()


    pdf = canvas.Canvas(
        buffer
    )


    pdf.setTitle(
        "PneumoVision AI Report"
    )


    pdf.setFont(
        "Helvetica-Bold",
        18
    )


    pdf.drawString(
        50,
        800,
        "PneumoVision AI"
    )


    pdf.setFont(
        "Helvetica",
        12
    )


    pdf.drawString(
        50,
        770,
        "Chest X-Ray Pneumonia Analysis Report"
    )


    pdf.drawString(
        50,
        740,
        f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )


    pdf.line(
        50,
        720,
        550,
        720
    )


    pdf.setFont(
        "Helvetica-Bold",
        14
    )


    pdf.drawString(
        50,
        690,
        "Final Diagnosis"
    )


    pdf.setFont(
        "Helvetica",
        12
    )


    pdf.drawString(
        50,
        665,
        diagnosis
    )


    pdf.drawString(
        50,
        640,
        f"Confidence: {confidence*100:.2f}%"
    )



    pdf.setFont(
        "Helvetica-Bold",
        14
    )


    pdf.drawString(
        50,
        600,
        "Model Predictions"
    )


    y = 570


    pdf.setFont(
        "Helvetica",
        12
    )


    for item in results:


        pdf.drawString(

            70,

            y,

            f"{item['Model']} : {item['Prediction']} ({item['Confidence']*100:.2f}%)"

        )


        y -= 25



    pdf.setFont(
        "Helvetica-Bold",
        14
    )


    pdf.drawString(
        50,
        y-20,
        "Disclaimer"
    )


    pdf.setFont(
        "Helvetica",
        10
    )


    pdf.drawString(
        50,
        y-45,
        "This AI system is developed for educational"
    )


    pdf.drawString(
        50,
        y-60,
        "and research purposes only."
    )


    pdf.drawString(
        50,
        y-75,
        "Consult a qualified healthcare professional"
    )


    pdf.drawString(
        50,
        y-90,
        "for medical decisions."
    )


    pdf.save()


    buffer.seek(0)


    return buffer





# ============================================================
# DOWNLOAD REPORT
# ============================================================


if uploaded_file is not None:


    st.divider()


    st.subheader(
        "📄 Generate Medical Report"
    )


    pdf_file = create_pdf_report(

        final,

        final_confidence,

        results

    )



    st.download_button(

        label="Download AI Report PDF",

        data=pdf_file,

        file_name="PneumoVision_Report.pdf",

        mime="application/pdf"

    )





# ============================================================
# DISEASE INFORMATION
# ============================================================


st.divider()


st.subheader(
    "📖 Understanding Pneumonia"
)



tab1, tab2, tab3 = st.tabs(
    [
        "About",
        "Symptoms",
        "Prevention"
    ]
)



with tab1:


    st.markdown(
    """
    ### What is Pneumonia?


    Pneumonia is an infection that affects
    the lungs and causes inflammation of
    air sacs called alveoli.


    The alveoli may fill with fluid,
    making breathing difficult.


    Common causes include:

    - Bacteria
    - Viruses
    - Fungi

    """
    )



with tab2:


    st.markdown(
    """
    ### Common Symptoms


    🟢 Fever

    🟢 Cough

    🟢 Chest pain

    🟢 Difficulty breathing

    🟢 Fatigue

    🟢 Chills

    """
    )



with tab3:


    st.markdown(
    """
    ### Prevention


    ✔ Maintain hygiene


    ✔ Avoid smoking


    ✔ Take recommended vaccines


    ✔ Exercise regularly


    ✔ Maintain healthy immunity


    """
    )





# ============================================================
# MEDICAL DISCLAIMER
# ============================================================


st.warning(
"""
⚠ Medical Disclaimer:

PneumoVision AI is an educational AI demonstration.
The prediction should not be considered a medical diagnosis.

Always consult qualified healthcare professionals.
"""
)





# ============================================================
# FOOTER
# ============================================================


st.divider()


st.markdown(
"""
<div style="text-align:center">


<h3>🫁 PneumoVision AI</h3>


TensorFlow • Streamlit • OpenCV • Deep Learning


<br><br>


Developed by <b>Nahid Kausar</b>


</div>
""",
unsafe_allow_html=True
)
