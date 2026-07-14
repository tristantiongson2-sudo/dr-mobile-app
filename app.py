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
    label: str = Field(description="The type of lesion, e.g., microaneurysm, hemorrhage, hard exudate, light exudate, cotton wool spot")
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

# --- PREPROCESSING IMAGE ENHANCEMENT ENGINE ---
def preprocess_with_clahe(pil_image):
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
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
    green = cv_img[:, :, 1]
    
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
    site_records = []
    
    for i, lesion in enumerate(lesions):
        site_letter = alphabet[i % len(alphabet)]
        raw_label = lesion.get("label", "unknown").lower()
        box_2d = lesion.get("box_2d", [0, 0, 0, 0])
        
        # --- SMART KEYWORD MATCHING ENGINE ---
        # This handles variations like plurals, casing, or "light" vs "hard"
        if "hemorrhage" in raw_label or "bleed" in raw_label or "blood" in raw_label:
            base_color = (255, 0, 0)          # Red
            display_label = "Hemorrhage"
            color_name = "Red"
        elif "exudate" in raw_label:
            base_color = (255, 255, 0)        # Yellow
            color_name = "Yellow"
            if "light" in raw_label or "soft" in raw_label:
                display_label = "Light Exudate"
            else:
                display_label = "Hard Exudate"
        elif "microaneurysm" in raw_label or "aneurysm" in raw_label:
            base_color = (255, 165, 0)        # Orange
            display_label = "Microaneurysm"
            color_name = "Orange"
        elif "cotton" in raw_label or "wool" in raw_label:
            base_color = (255, 255, 255)      # White
            display_label = "Cotton Wool Spot"
            color_name = "White"
        else:
            base_color = (0, 255, 0)          # Default to Green if unknown
            display_label = raw_label.title()
            color_name = "Green"
            
        ymin, xmin, ymax, xmax = box_2d
        x1 = int((xmin / 1000) * width)
        y1 = int((ymin / 1000) * height)
        x2 = int((xmax / 1000) * width)
        y2 = int((ymax / 1000) * height)
        
        fill_color = base_color + (80,)               
        outline_color = base_color + (255,)            
        
        # Draw the visual heatmap square
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=3)
        
        # --- DYNAMIC EXPLICIT LABELED BADGES ---
        # Create a text badge containing both the Site Letter AND the actual disease name!
        badge_text = f"Site {site_letter}: {display_label}"
        
        # Estimate badge width dynamically based on text length to prevent clipping
        badge_width = len(badge_text) * 7 + 10
        badge_y = max(5, y1 - 20)
        if badge_y <= 5:
            badge_y = y1 + 5
            
        # Draw background badge container
        draw.rectangle([x1, badge_y, x1 + badge_width, badge_y + 16], fill=outline_color)
        
        # Draw high-contrast text label on top of the badge
        text_color = (0, 0, 0, 255) if base_color == (255, 255, 0) else (255, 255, 255, 255)
        draw.text((x1 + 6, badge_y + 1), badge_text, fill=text_color)
        
        site_records.append({
            "site": f"Site {site_letter}",
            "color": color_name,
            "type": display_label,
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Uploaded Image", use_container_width=True)
        
    with col2:
        enhanced_image = preprocess_with_clahe(original_image)
        st.image(enhanced_image, caption="⚡ CLAHE Preprocessed Image (Enhanced Background & Sides)", use_container_width=True)
    
    if st.button("Analyze & Map Enhanced Retina", type="primary"):
        if not api_key:
            st.error("Configuration Error: API Key not found.")
        else:
            with st.spinner("Executing hybrid analysis on enhanced imagery..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    prompt = (
                        "Perform a rigorous clinical analysis of this preprocessed fundus image using the International Clinical Diabetic Retinopathy (ICDR) scale. "
                        "CRITICAL SEARCH INSTRUCTION: You must explicitly and meticulously inspect the outer sides, peripheral boundaries, and extreme dark edges of the retina frame. "
                        "Do not ignore faint yellow spots, pale lesions, or cotton wool structures near the edges. Background shadows have been minimized using CLAHE to make them visible.\n\n"
                        "Classify using these criteria:\n"
                        "- **No DR**: Absolutely no abnormalities.\n"
                        "- **Mild NPDR**: Microaneurysms only.\n"
                        "- **Moderate NPDR**: More than microaneurysms (e.g., hard/light exudates, cotton wool spots, or blot hemorrhages) but less than Severe NPDR.\n"
                        "- **Severe NPDR**: >20 intraretinal hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, or prominent IRMA.\n"
                        "- **PDR**: Neovascularization or vitreous/preretinal hemorrhage.\n\n"
                        "Locate all detected lesions and output their bounding boxes as [ymin, xmin, ymax, xmax] normalized to 0-1000."
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=[enhanced_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=RetinalAnalysis,
                        ),
                    )
                    
                    analysis = json.loads(response.text)
                    
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
                            color_emoji = "🟡" if record['color'] == "Yellow" else ("🔴" if record['color'] == "Red" else ("🟠" if record['color'] == "Orange" else "⚪"))
                            st.markdown(
                                f"**{record['site']}** ({color_emoji} {record['color']})  \n"
                                f"**Pathology:** {record['type']}  \n"
                                f"**Location Bounds:** `{record['coordinates']}`  \n"
                                f"---"
                            )
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
