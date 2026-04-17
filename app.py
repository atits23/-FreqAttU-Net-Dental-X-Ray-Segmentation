import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------
# 1. Exact Model Architecture (From your working Notebook)
# ---------------------------------------------------------
class FrequencyEnhancementModule(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(fft)
        magnitude = torch.log1p(magnitude)
        magnitude = F.interpolate(magnitude, size=(x.shape[2], x.shape[3]),
                                  mode='bilinear', align_corners=False)
        freq_feat = self.conv(magnitude)
        return freq_feat


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class FreqAttUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_f=32):
        super().__init__()
        f = base_f 

        self.fem = FrequencyEnhancementModule(out_channels=f)

        self.enc1 = DoubleConv(in_ch, f)
        self.enc2 = DoubleConv(f*2,  f*2)
        self.enc3 = DoubleConv(f*2,  f*4)
        self.enc4 = DoubleConv(f*4,  f*8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f*8, f*16)

        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(f*8,  f*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(f*4,  f*2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(f*2,  f,   2, stride=2)

        self.att4 = AttentionGate(F_g=f*8,  F_l=f*8,  F_int=f*4)
        self.att3 = AttentionGate(F_g=f*4,  F_l=f*4,  F_int=f*2)
        self.att2 = AttentionGate(F_g=f*2,  F_l=f*2,  F_int=f)
        self.att1 = AttentionGate(F_g=f,    F_l=f,    F_int=f//2)

        self.dec4 = DoubleConv(f*16, f*8)
        self.dec3 = DoubleConv(f*8,  f*4)
        self.dec2 = DoubleConv(f*4,  f*2)
        self.dec1 = DoubleConv(f*2,  f)

        self.out_conv = nn.Conv2d(f, out_ch, 1)

    def forward(self, x):
        freq_feat = self.fem(x)

        e1 = self.enc1(x)
        e1 = e1 + freq_feat

        e2_in = torch.cat([self.pool(e1), self.pool(freq_feat)], dim=1)
        e2 = self.enc2(e2_in)

        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        a4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))

        d3 = self.up3(d4)
        a3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------
# 2. Caching and Loading the Model
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FreqAttUNet(in_ch=1, out_ch=1, base_f=32).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Point this to your best_model.pth file in the same directory
MODEL_PATH = "best_model.pth" 
model, device = load_model(MODEL_PATH)


# ---------------------------------------------------------
# 3. Streamlit UI and Logic
# ---------------------------------------------------------
st.set_page_config(page_title="Dental X-Ray Segmentation", layout="wide")

st.title("🦷 FreqAttU-Net: Dental X-Ray Segmentation")
st.write("Upload a panoramic dental X-ray to automatically segment the teeth using explicit frequency-domain features.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L") 
    img_array = np.array(image)
    
    with st.spinner('Segmenting teeth...'):
        img_resized = cv2.resize(img_array, (512, 512))
        
        # Normalize to [-1, 1] as expected by your model
        img_norm = img_resized / 255.0
        img_norm = (img_norm - 0.5) / 0.5 
        
        # Add batch and channel dimensions
        img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        if model is not None:
            with torch.no_grad():
                raw_output = model(img_tensor)
                probabilities = torch.sigmoid(raw_output)
                binary_mask = (probabilities > 0.5).float().squeeze().cpu().numpy()
                
            display_mask = (binary_mask * 255).astype(np.uint8)
            
            # --- Layout for Side-by-Side Display ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original X-Ray")
                st.image(img_resized, use_container_width=True, clamp=True, channels="GRAY")
                
            with col2:
                st.subheader("Predicted Teeth Mask")
                st.image(display_mask, use_container_width=True, clamp=True, channels="GRAY")
                
            # --- Overlay View ---
            st.subheader("Overlay View")
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            mask_rgb = np.zeros_like(img_rgb)
            mask_rgb[:,:,1] = display_mask # Put mask in Green channel
            
            overlay = cv2.addWeighted(img_rgb, 0.7, mask_rgb, 0.3, 0)
            st.image(overlay, use_container_width=True, channels="RGB")