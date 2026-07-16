from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import models, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import streamlit as st
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -------------------------------------------------------------------
# 1. PAGE SETTINGS & HEADER
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Horse vs Zebra Classifier",
    page_icon="🦓",
    layout="wide"
)

st.title("🐎 Horse vs Zebra — 3 Model Comparison 🦓")
st.divider()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Horse', 'Zebra']

# -------------------------------------------------------------------
# 2. MODEL ARCHITECTURES & LOADING FUNCTIONS
# -------------------------------------------------------------------


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


@st.cache_resource
def load_models():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    cnn_path = os.path.join(MODELS_DIR, "model_scratch_cnn.pth")
    resnet_path = os.path.join(MODELS_DIR, "model_efficientnet.pth")
    vit_path = os.path.join(MODELS_DIR, "model_deit.pth")

    cnn_model = SimpleCNN().to(device)
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_model.eval()

    # EFFICIENTNET B0
    effi_model = models.efficientnet_b0(weights=None)
    classifier_layer = effi_model.classifier[1]
    if isinstance(classifier_layer, nn.Linear):
        num_features = classifier_layer.in_features
        effi_model.classifier[1] = nn.Linear(num_features, 2)

    effi_model = effi_model.to(device)
    effi_model.load_state_dict(torch.load(resnet_path, map_location=device))
    effi_model.eval()

    # DeiT / ViT
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    deit_model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    deit_model = deit_model.to(device)
    deit_model.load_state_dict(torch.load(vit_path, map_location=device))
    deit_model.eval()

    return cnn_model, effi_model, deit_model, processor


cnn_model = None
effi_model = None
deit_model = None
processor = None
models_loaded = False

try:
    with st.spinner("Loading models, please wait..."):
        cnn_model, effi_model, deit_model, processor = load_models()
        models_loaded = True
    st.sidebar.success(" All models loaded successfully!")
except FileNotFoundError as e:
    st.error(" **Model files not found!** Please make sure that the 3 `.pth` files are inside the `models` folder.")
    st.exception(e)
except Exception as e:
    st.error(f" An error occurred: {e}")

# -------------------------------------------------------------------
# 3. IMAGE PREPROCESSING & PREDICTION
# -------------------------------------------------------------------
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(image, model, is_vit=False):
    with torch.no_grad():
        if is_vit:
            if processor is not None:
                inputs = processor(
                    images=image, return_tensors="pt").to(device)
                outputs = model(**inputs).logits
            else:
                return "Error", 0.0, None
        else:
            img_tensor = cnn_transform(image).unsqueeze(0).to(device)
            outputs = model(img_tensor)

        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)

        idx = int(predicted_class.item())
        return CLASSES[idx], float(confidence.item()) * 100, probabilities.cpu().numpy()


# -------------------------------------------------------------------
# 4. USER INTERFACE
# -------------------------------------------------------------------
if models_loaded and cnn_model is not None and effi_model is not None and deit_model is not None:
    uploaded_file = st.file_uploader(
        "Upload a Horse or Zebra photo...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col_img, col_results = st.columns([1, 2])

        with col_img:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image",
                     use_container_width=True)

        with col_results:
            st.subheader(" Model Predictions")
            st.write(
                "Here are the predictions and confidence scores from the 3 distinct architectures:")

            m_col1, m_col2, m_col3 = st.columns(3)

            # 1. Model: SimpleCNN
            cnn_pred, cnn_conf, _ = predict(image, cnn_model, is_vit=False)
            with m_col1:
                st.info("**SimpleCNN (From-Scratch)**")
                st.metric(label="Prediction", value=cnn_pred)
                st.write(f"**Confidence:** %{cnn_conf:.2f}")
                st.progress(int(cnn_conf))

            # 2. Model: EfficientNet B0
            effi_pred, effi_conf, _ = predict(image, effi_model, is_vit=False)
            with m_col2:
                st.warning("**EfficientNet B0**")
                st.metric(label="Prediction", value=effi_pred)
                st.write(f"**Confidence:** %{effi_conf:.2f}")
                st.progress(int(effi_conf))

            # 3. Model: DeiT / ViT
            deit_pred, deit_conf, _ = predict(image, deit_model, is_vit=True)
            with m_col3:
                st.success("**DeiT / Vision Transformer**")
                st.metric(label="Prediction", value=deit_pred)
                st.write(f"**Confidence:** %{deit_conf:.2f}")
                st.progress(int(deit_conf))

else:
    st.info(
        " Please make sure your model files are loaded. The page will load automatically.")
