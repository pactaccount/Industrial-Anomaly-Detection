import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
st.title("Unsupervised Industrial Anomaly Detection")
st.markdown("A PyTorch-based perception engine using WideResNet-50 feature extraction and Linear Algebra (KNN) to detect manufacturing defects without prior training on defective data.")

# Initialize the model globally to avoid reloading on every interaction
@st.cache_resource
def load_feature_extractor():
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model.eval()
    
    # Store intermediate layer feature maps
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
        
    model.layer2.register_forward_hook(get_features('layer2'))
    model.layer3.register_forward_hook(get_features('layer3'))
    
    return model, features

model, features_dict = load_feature_extractor()

# Standardize images for ResNet inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_tensor):
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Average pooling to reduce spatial dimensionality and noise
    layer2_feat = F.avg_pool2d(features_dict['layer2'], 3, 1, 1)
    layer3_feat = F.avg_pool2d(features_dict['layer3'], 3, 1, 1)
    
    # Resize layer3 features to match layer2 spatial dimensions
    layer3_feat = F.interpolate(layer3_feat, size=layer2_feat.shape[2:], mode='bilinear', align_corners=False)
    
    # Concatenate features along the channel dimension
    combined_features = torch.cat([layer2_feat, layer3_feat], dim=1)
    
    # Reshape to (N, Channels) where N is the number of spatial patches
    b, c, h, w = combined_features.shape
    combined_features = combined_features.view(c, h * w).t().numpy()
    
    return combined_features, (h, w)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Memory Bank Initialization")
    st.markdown("Upload a **Normal (Defect-Free)** image to establish the mathematical baseline.")
    normal_file = st.file_uploader("Upload Normal Image", type=["jpg", "png", "jpeg"], key="normal")

with col2:
    st.subheader("2. Threat Perception (Test)")
    st.markdown("Upload a **Test Image** (potentially defective) to scan for structural anomalies.")
    test_file = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"], key="test")

if normal_file and test_file:
    with st.spinner("Extracting ResNet embeddings and computing matrix distances..."):
        
        normal_img = Image.open(normal_file).convert("RGB")
        test_img = Image.open(test_file).convert("RGB")
        
        normal_tensor = transform(normal_img).unsqueeze(0)
        test_tensor = transform(test_img).unsqueeze(0)
        
        # Extract feature arrays from both images
        normal_features, spatial_dims = extract_features(normal_tensor)
        test_features, _ = extract_features(test_tensor)
        
        # Construct the KNN memory bank using the normal features
        knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        knn.fit(normal_features)
        
        # Calculate spatial divergence (anomaly scores) for the test features
        distances, _ = knn.kneighbors(test_features)
        
        # Reshape distances back into a spatial heatmap
        anomaly_map = distances.reshape(spatial_dims)
        
        # Normalize and upscale the heatmap to the original test image dimensions
        anomaly_map = cv2.resize(anomaly_map, test_img.size)
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
        
        # Render semantic heatmap overlay
        heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        test_img_np = np.array(test_img)
        overlay = cv2.addWeighted(test_img_np, 0.6, heatmap_rgb, 0.4, 0)
        
        st.success("Matrix Operations Complete.")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(test_img, caption="Original Test Image", use_column_width=True)
        with res_col2:
            st.image(overlay, caption="Anomaly Heatmap (Red = Spatial Divergence)", use_column_width=True)
else:
    st.info("Awaiting structural baseline and test image injection.")
