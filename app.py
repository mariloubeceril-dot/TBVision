import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from reportlab.pdfgen import canvas
from datetime import datetime
import tempfile

# ================= LOAD MODEL =================
import sys
import os

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "TBVision_DenseNet_Model.h5")

model = load_model(model_path, compile=False)

IMG_SIZE = (224, 224)

# ================= FIND LAST CONV LAYER =================
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

# ================= X-RAY VALIDATION =================
def is_xray_like(img):

    img_np = np.array(img)

    if len(img_np.shape) < 3:
        return False

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    color_variance = np.var(img_np[:,:,0] - img_np[:,:,1])
    contrast = gray.std()

    if color_variance > 500 or contrast < 20:
        return False

    return True


# ================= WINDOW LEVEL =================
def apply_window_level(img, window=150, level=120):

    img = np.array(img.convert("L"))

    lower = level - window//2
    upper = level + window//2

    img = np.clip(img, lower, upper)

    img = ((img - lower) / (upper - lower) * 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# ================= GRADCAM =================
def make_gradcam_heatmap(img_array):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        predictions = tf.convert_to_tensor(predictions)

        if len(predictions.shape) == 2:
            loss = predictions[:,0]
        else:
            loss = predictions

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap,0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap


# ================= OVERLAY HEATMAP =================
def overlay_heatmap(img, heatmap):

    img = np.array(img)

    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_bgr,0.6,heatmap_color,0.4,0)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay


# ================= PDF REPORT =================
def create_pdf(text, cam_img):

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    c = canvas.Canvas(tmp.name)

    c.drawString(80,750,"TBVision AI Screening Report")

    c.drawString(80,720,f"Date: {datetime.now()}")

    y = 690

    for line in text.split("\n"):
        c.drawString(80,y,line)
        y -= 20

    img_path = tmp.name.replace(".pdf",".png")

    cv2.imwrite(img_path,cam_img)

    c.drawImage(img_path,80,380,width=300,height=250)

    c.drawString(80,340,"Disclaimer:")
    c.drawString(80,320,"This AI tool does NOT provide medical diagnosis.")
    c.drawString(80,300,"Consult a licensed healthcare professional.")

    c.save()

    return tmp.name


# ================= PREDICTION =================
def predict_tbvision(img, window, level):

    try:

        if img is None:
            return "Please upload an image.",0,0,None,None,None

        img = img.convert("RGB")

        if not is_xray_like(img):
            return "Invalid Image: Please upload a chest X-ray.",0,0,None,None,None

        dicom_view = apply_window_level(img,window,level)

        img_resized = Image.fromarray(dicom_view).resize(IMG_SIZE)

        img_array = np.array(img_resized) / 255.0

        img_batch = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_batch)

        pred = np.array(pred).flatten()

        tb_prob = float(pred[0])

        tb_percent = round(tb_prob*100,2)
        normal_percent = round((1-tb_prob)*100,2)

        if tb_prob > 0.5:
            result = "⚠️ Possible Tuberculosis Detected"
        else:
            result = "✅ Normal (No Tuberculosis Detected)"

        heatmap = make_gradcam_heatmap(img_batch)

        cam_img = overlay_heatmap(dicom_view,heatmap)

        pdf_path = create_pdf(
            f"Result: {result}\nTB Likelihood: {tb_percent}%\nNormal Likelihood: {normal_percent}%",
            cam_img
        )

        return result,tb_percent,normal_percent,dicom_view,cam_img,pdf_path

    except Exception as e:

        return f"Error: {str(e)}",0,0,None,None,None


# ================= GRADIO UI =================
with gr.Blocks(theme=gr.themes.Soft(), title="TBVision") as demo:

    gr.Markdown(
    """
# 🩺 TBVision: AI Tuberculosis Screening System

Upload a **Chest X-ray Image** for AI-assisted tuberculosis screening.
"""
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(type="pil", label="Upload Chest X-ray")

            window = gr.Slider(50,255,value=150,label="Window")

            level = gr.Slider(50,255,value=120,label="Level")

            btn = gr.Button("Analyze X-ray")

        with gr.Column():

            result = gr.Textbox(label="Screening Result")

            tb_prob = gr.Slider(0,100,label="Tuberculosis Likelihood (%)")

            normal_prob = gr.Slider(0,100,label="Normal Likelihood (%)")

    with gr.Row():

        original_view = gr.Image(
            label="Processed X-ray",
            height=320
        )

        heatmap = gr.Image(
            label="AI Attention Heatmap (Grad-CAM)",
            height=320
        )

    gr.Markdown("""
### 🔎 AI Model Attention Scale (Grad-CAM)

🔴 **High Model Activation** – Areas strongly influencing the AI prediction  
🟡 **Moderate Activation** – Regions with moderate influence  
🔵 **Low Activation** – Minimal influence on prediction  

⚠️ Heatmap highlights **AI attention**, not confirmed disease location.
""")

    pdf_file = gr.File(label="Download PDF Report")

    gr.Markdown("""
### ⚠️ Medical Disclaimer

TBVision is an **AI-assisted screening support tool** developed for educational and research purposes.

• The system **does NOT provide medical diagnosis**  
• Results should **NOT replace professional medical evaluation**  
• Always consult a **licensed physician or radiologist**
""")

    gr.Markdown("""
### 🔐 Data Privacy Notice (RA 10173 – Data Privacy Act of 2012)

• Uploaded images are processed **only for AI analysis**  
• No personal data is stored or shared  
• Images are **not permanently saved on the system**
""")

    btn.click(
        fn=predict_tbvision,
        inputs=[image_input,window,level],
        outputs=[result,tb_prob,normal_prob,original_view,heatmap,pdf_file]
    )

demo.launch()