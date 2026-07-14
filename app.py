import streamlit as st
from PIL import Image, ImageDraw
import json
import string
import numpy as np
import cv2
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="DR Mobile Assistant", layout="centered")
st.title("👁️ DR Mobile Assistant with Vascular & Lesion Mapping")

# Define our structured data output so Gemini parses coordinates reliably
class Lesion(BaseModel):
    label: str = Field(description="The type of lesion, e.g., microaneurysm, hemorrhage, hard exudates, cotton wool spot")
    box_2d: list[int] = Field(description="Bounding box coordinates in [ymin, xmin, ymax, xmax] format, normalized to 0-1000")

class RetinalAnalysis(BaseModel):
    dr_stage: str = Field(description="The assigned Diabetic Retinopathy stage (e.g., No DR, Mild NPDR, Moderate NPDR, Severe NPDR, PDR)")
    justification: str = Field(description="Detailed clinical justification for the assigned stage, explaining the detected signs")
    lesions: list[Lesion] = Field(description="List of detected lesions with their spatial coordinates")

# Silently pull the key from Streamlit Secrets backend
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("API Key is missing from Streamlit Secrets backend!")
    api_key = None

# --- NEW: PREPROCESSING IMAGE ENHANCEMENT ENGINE ---
def preprocess_with_clahe(pil_image):
    # Convert PIL Image to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # Convert to LAB color space (L=Lightness, A/B=Color channels)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the Lightness channel to balance dark borders/sides
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge channels back together and convert to RGB
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(enhanced_rgb)

# --- HYBRID MAPPING FUNCTION (OPENCV + PILLOW) ---
def map_retina(pil_image, lesions):
    rgb_image = pil_image.convert("RGB")
    width, height = rgb_image.size
    
    # 1. ADVANCED "UGAT" (VESSEL) EXTRACTION USING OPENCV
    cv_img = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    green = cv_img[:, :, 1] # Extract green channel for maximum vessel contrast
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(green)
    
    background = cv2.medianBlur(contrast_enhanced, 25)
    vessel_subtracted = cv2.subtract(background, contrast_enhanced)
    
    _, thresh = cv2.threshold(vessel_subtracted, 12, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    clean_vessels = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    vessel_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    vessel_rgba[clean_vessels == 255] = [0, 180, 255, 90] # Glowing cyan network
    vessel_layer = Image.fromarray(vessel_rgba, "RGBA")
    
    # 2. LESION HIGHLIGHT OVERLAYS
    overlay_layer = Image.new("RGBA", rgb_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)
    
    alphabet = string.ascii_uppercase
    COLOR_MAP = {
        "hemorrhage": (255, 0, 0),          # Red
        "hard exudates": (255, 255, 0),     # Yellow
        "hard exudate": (255, 255, 0),      # Yellow
        "microaneurysm": (255, 165, 0),     # Orange
        "cotton wool spot": (255, 255, 255)  # White
    }
    
    site_records = []
    
    for i, lesion in enumerate(lesions):
        site_letter = alphabet[i % len(alphabet)]
        label = lesion.get("label", "unknown").lower()
        box_2d = lesion.get("box_2d", [0, 0, 0, 0])
        
        ymin, xmin, ymax, xmax = box_2d
        x1 = int((xmin / 1000) * width)
        y1 = int((ymin / 1000) * height)
        x2 = int((xmax / 1000) * width)
        y2 = int((ymax / 1000) * height)
        
        base_color = COLOR_MAP.get(label, (0, 255, 0)) 
        fill_color = base_color + (80,)               
        outline_color = base_color + (255,)            
        
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=3)
        
        badge_text = f"Site {site_letter}"
        badge_y = max(5, y1 - 20)
        if badge_y <= 5:
            badge_y = y1 + 5
            
        draw.rectangle([x1, badge_y, x1 + 50, badge_y + 15], fill=outline_color)
        text_color = (0, 0, 0, 255) if base_color == (255, 255, 0) else (255, 255, 255, 255)
        draw.text((x1 + 4, badge_y + 1), badge_text, fill=text_color)
        
        color_name = "Yellow" if base_color == (255, 255, 0) else ("Red" if base_color == (255, 0, 0) else ("Orange" if base_color == (255, 165, 0) else "White"))
        site_records.append({
            "site": badge_text,
            "color": color_name,
            "type": label.title(),
            "coordinates": f"X: {x1}-{x2}, Y: {y1}-{y2}"
        })
        
    base_rgba = rgb_image.convert("RGBA")
    with_vessels = Image.alpha_composite(base_rgba, vessel_layer)
    final_output = Image.alpha_composite(with_vessels, overlay_layer)
    
    return final_output.convert("RGB"), site_records


# --- STREAMLIT UI ---
uploaded_file = st.file_uploader("Upload Fundus Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # Set up side-by-side columns to show clinical preprocessing enhancement
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Uploaded Image", use_container_width=True)
        
    with col2:
        # Generate and show the brightened, contrast-equalized image
        enhanced_image = preprocess_with_clahe(original_image)
        st.image(enhanced_image, caption="⚡ CLAHE Preprocessed Image (Enhanced Background & Sides)", use_container_width=True)
    
    if st.button("Analyze & Map Enhanced Retina", type="primary"):
        if not api_key:
            st.error("Configuration Error: API Key not found.")
        else:
            with st.spinner("Executing hybrid analysis on enhanced imagery..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    # Sharpened prompt demanding peripheral border inspections
                    prompt = (
                        "Perform a rigorous clinical analysis of this preprocessed fundus image using the International Clinical Diabetic Retinopathy (ICDR) scale. "
                        "CRITICAL SEARCH INSTRUCTION: You must explicitly and meticulously inspect the outer sides, peripheral boundaries, and extreme dark edges of the retina frame. "
                        "Do not ignore faint yellow spots, pale lesions, or cotton wool structures near the edges. Background shadows have been minimized using CLAHE to make them visible.\n\n"
                        "Classify using these criteria:\n"
                        "- **No DR**: Absolutely no abnormalities.\n"
                        "- **Mild NPDR**: Microaneurysms only.\n"
                        "- **Moderate NPDR**: More than microaneurysms (e.g., hard exudates, cotton wool spots, or blot hemorrhages) but less than Severe NPDR.\n"
                        "- **Severe NPDR**: >20 intraretinal hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, or prominent IRMA.\n"
                        "- **PDR**: Neovascularization or vitreous/preretinal hemorrhage.\n\n"
                        "Locate all detected lesions and output their bounding boxes as [ymin, xmin, ymax, xmax] normalized to 0-1000."
                    )
                    
                    # Send the ENHANCED image to Gemini for superior clarity
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=[enhanced_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=RetinalAnalysis,
                        ),
                    )
                    
                    analysis = json.loads(response.text)
                    
                    # Generate map using the preprocessed version so markers sit on the high-visibility image
                    mapped_image, site_records = map_retina(enhanced_image, analysis.get("lesions", []))
                    
                    st.success("Analysis and Hybrid Mapping Complete!")
                    
                    st.subheader("Interactive Clinical Map")
                    st.image(mapped_image, caption="AI Lesion Overlay + Computer Vision Vasculature Map (Ugat)", use_container_width=True)
                    
                    st.subheader("Diagnostic Report")
                    st.metric(label="Assigned ICDR Stage", value=analysis["dr_stage"])
                    st.write(f"**Clinical Justification:** {analysis['justification']}")
                    
                    st.write("### 🔍 Site Interpretation & Pathology Key")
                    if not site_records:
                        st.info("No active lesion sites were plotted on the canvas.")
                    else:
                        for record in site_records:
                            color_emoji = "🟡" if record['color'] == "Yellow" else ("🔴" if record['color'] == "Red" else "🟠")
                            st.markdown(
                                f"**{record['site']}** ({color_emoji} {record['color']})  \n"
                                f"**Pathology:** {record['type']}  \n"
                                f"**Location Bounds:** `{record['coordinates']}`  \n"
                                f"---"
                            )
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
