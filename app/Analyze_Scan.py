import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import pandas as pd
import time
from io import BytesIO
import openai
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ========== Key Validation Checker  ==========

#API exist and valid
def is_valid_key(key: str) -> bool:
    try:
        openai.api_key = key
        openai.models.list()  # Lightweight API call
        return True
    except Exception:
        return False

# Get current key from environment
current_key = os.environ.get("OPENAI_API_KEY")

# If no key or invalid key, prompt the user
if not current_key or not is_valid_key(current_key):
    st.set_page_config(
        page_title="API Key", 
        layout="centered",
        page_icon = "visualization/logoO.png"
    )
        
    st.title("OpenAI API Key Required")

    if current_key and not is_valid_key(current_key):
        st.error("The current API key is invalid. Please enter a valid one.")

    key_input = st.text_input("OpenAI API Key:", type="password")
    
    if key_input:
        if is_valid_key(key_input.strip()):
            os.environ["OPENAI_API_KEY"] = key_input.strip()
            st.session_state["api_key"] = key_input.strip()
            st.success("API Key is valid! Loading app...")
            st.rerun()

        else:
            st.error("Invalid API Key. Please try again.")

    st.stop()  # Block the rest of the app until a valid key is set


# ========== Model Check ==========
from models.model_checker import check_models
check_models()  # This will block the app if no models exist
# ========== Model Check ==========


#local packages 
from glcm.resnet_glcm import ResNetWithInternalGLCM
from inference.explainer import explain_prediction
from inference.checker import is_retinal_image_openai

# Page config
st.set_page_config(
    page_title="Retinal Anomaly Detector",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="visualization/logoO.png"
)

# ========== Helper Functions ==========

@st.cache_resource(show_spinner=False)
def load_model_and_labels(model_path: str, labels_csv: str, device):
    labels_df = pd.read_csv(labels_csv)
    labels = labels_df.columns[1:].tolist()
    
    model = ResNetWithInternalGLCM(num_classes=len(labels))
    state_dict = torch.load(model_path, map_location=device)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, labels

def preprocess_image(img: Image.Image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(img).unsqueeze(0)

def apply_normal_constraint(probs: np.ndarray, normal_idx: int):
    probs = probs.copy()
    if probs[normal_idx] > 0.9 and any( probs[i] > 0.9 for i in range(len(probs)) if i != normal_idx ):
        probs[normal_idx] /= 4.5

    elif probs[normal_idx] > 0.9 and not (any( probs[i] > 0.9 for i in range(len(probs)) if i != normal_idx )):
        probs[:normal_idx] = 0.0
        
    return probs

def generate_gradcam(model, input_tensor, device, original_image=None, target_layer_name="layer4"):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = getattr(model.base_model, target_layer_name)[-1].conv3
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    outputs = model(input_tensor.to(device))
    class_idx = outputs.argmax(dim=1).item()

    one_hot = torch.zeros_like(outputs)
    one_hot[0][class_idx] = 1
    outputs.backward(gradient=one_hot)

    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0][0].cpu().detach()

    for i in range(activation.size(0)):
        activation[i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activation, dim=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    heatmap = np.uint8(255 * heatmap.numpy())
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if original_image:
        heatmap = cv2.resize(heatmap, original_image.size)

    original_np = np.array(original_image.convert("RGB"))
    superimposed_img = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

    handle_fw.remove()
    handle_bw.remove()

    return superimposed_img, class_idx

def resize_for_display(image: Image.Image, max_width: int = 480):
    """Resize image for clean display preserving aspect ratio."""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_size = (max_width, int(height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image


# ========== Main App ==========

def main():
    st.title("Retinal Anomaly Detector")
    st.markdown(
        """
        Analyze retinal fundus images with AI-powered anomaly detection.
        Upload your retinal scan to receive detailed diagnosis and visualization.
        """
    )

    # Sidebar controls for model paths and device selection
    with st.sidebar:
        st.header("Settings")
        device_option = st.radio("Select Device", options=["Auto (GPU if available)", "CPU", "GPU"], index=0)
        if device_option == "CPU":
            device = torch.device("cpu")
        elif device_option == "GPU":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = st.text_input("Model Currently Running", "models/v4_best_b.pth")
        labels_csv = st.text_input("Disease Labels", "data/train/train.csv")

        st.markdown("---")
        st.write("App Version: 1.3")

    # Load model and labels once
    with st.spinner("Loading model and labels..."):
        model, labels = load_model_and_labels(model_path, labels_csv, device)

    # Upload image
    uploaded_file = st.file_uploader("Upload retinal fundus image (jpg, png, tiff)", type=["jpg", "jpeg", "png", "tiff"])
    if not uploaded_file:
        st.info("Please upload a retinal fundus image to begin analysis.")
        return

    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

    # Layout: Two columns for original image & Grad-CAM heatmap side by side
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Original Image")
        img_disp = resize_for_display(img)
        st.image(img_disp, use_container_width=False)

    # Validate if retinal fundus image
    with st.spinner("Validating image..."):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
            img.save(tmp.name)
            is_retinal = is_retinal_image_openai(tmp.name)

    if not is_retinal:
        st.error("Uploaded image does not appear to be a valid retinal fundus scan. Please upload a proper retinal image.")
        return

    # Process prediction
    input_tensor = preprocess_image(img).to(device)
    with st.spinner("Running anomaly detection..."):
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().detach().numpy().squeeze()
        probs = apply_normal_constraint(probs, normal_idx=len(labels) - 1)

    # Sort results
    sorted_results = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

    with col2:
        st.subheader("Attention Heatmap (Grad-CAM)")
        with st.spinner("Generating Grad-CAM heatmap..."):
            gradcam_img_np, pred_idx = generate_gradcam(model, input_tensor, device, original_image=img)
            gradcam_pil = Image.fromarray(gradcam_img_np)
            gradcam_disp = resize_for_display(gradcam_pil)
            st.image(gradcam_disp, use_container_width=False, caption=f"Predicted: {labels[pred_idx]}")

    # Results below images, full width
    st.markdown("---")
    st.subheader("Detection Results")

    df_results = pd.DataFrame(sorted_results, columns=["Condition", "Confidence"])
    st.dataframe(df_results.style.format({"Confidence": "{:.2%}"}))


    #st.subheader("Interpretation")
    explanation_text = explain_prediction(sorted_results)
    explanation_text = explanation_text.replace("*","").replace("-", "\n-")
    
    # --- Interpretation ---
    st.markdown("### Interpretation")
    st.markdown(
        f"<div style='padding: 20px; background: rgb(255 255 255 / 0%); font-family:Arial; font-size:14px; border-radius: 10px; font-family: monospace; color: rgb(255, 255, 255);'>"
        f"{explanation_text}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")



    # ========= Add Medical Report Button =========
    disclaimer = "\n\nDisclaimer: This is an automated assessment and should not be used as a substitute for professional medical advice."
    if st.button("Show Report"):
        st.session_state["report_data"] = {
            "results": sorted_results,
            "explanation": explanation_text + disclaimer
        }
        # Save images too
        st.session_state["original_img"] = img_disp
        st.session_state["gradcam_img"] = gradcam_disp
        st.switch_page("pages/Retinal_Report.py")


    st.caption("Model & data ©  Blueray / Company")

if __name__ == "__main__":
    main()
