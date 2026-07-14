import streamlit as st
from PIL import Image, ImageDraw
import json
import string
import numpy as np
import cv2
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="DR Mobile Assistant", layout="wide")
st.title("👁️ DR Mobile Assistant with Super-Vision Preprocessing")

# Define structured output format
class Lesion(BaseModel):
    label: str = Field(description="Type of lesion, e.g., microaneurysm, hemorrhage, hard exudates, light exudate, cotton wool spot")
    box_2d: list[int] = Field(description="Bounding box coordinates in [ymin, xmin, ymax, xmax] format, normalized to 0-1000")

class RetinalAnalysis(BaseModel):
    dr_stage: str = Field(description="The assigned Diabetic Retinopathy stage (No DR, Mild NPDR, Moderate NPDR, Severe NPDR, PDR)")
    justification: str = Field(description="Detailed clinical justification for the assigned stage")
    lesions: list[Lesion] = Field(description="List of detected lesions with spatial coordinates")

# Silently pull the key from Streamlit Secrets backend
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("API Key is missing from Streamlit Secrets backend!")
    api_key = None

# --- INITIALIZE STATE FOR CONVERSATIONAL MEMORY ---
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# --- NEW: ADVANCED SUPER-VISION PREPROCESSING ENGINE ---
def preprocess_super_vision(pil_image):
    # Convert PIL Image to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # STAGE 1: CLAHE (Luminance/Shadow Correction)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    img_clahe = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    
    # STAGE 2: HSV Saturation Tuning (Smart Vibrance Boost)
    hsv = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Scale up saturation to make faint yellows/reds highly distinct from background tissue
    s_boosted = cv2.addWeighted(s, 1.3, np.zeros(s.shape, s.dtype), 0, 10)
    img_vibrant = cv2.cvtColor(cv2.merge((h, s_boosted, v)), cv2.COLOR_HSV2BGR)
    
    # STAGE 3: Unsharp Masking (Sharpness Enhancement for Micro-Lesions)
    gaussian_blur = cv2.GaussianBlur(img_vibrant, (5, 5), 1.5)
    img_sharpened = cv2.addWeighted(img_vibrant, 1.5, gaussian_blur, -0.5, 0)
    
    # Convert back to PIL RGB
    final_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

# --- STANDARD BASE CLAHE FOR SIDE-BY-SIDE VISUAL COMPARISON ---
def preprocess_only_clahe(pil_image):
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB))

# --- HYBRID MAPPING FUNCTION ---
def map_retina(pil_image, lesions):
    rgb_image = pil_image.convert("RGB")
    width, height = rgb_image.size
    
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
    vessel_rgba[clean_vessels == 255] = [0, 180, 255, 90] 
    vessel_layer = Image.fromarray(vessel_rgba, "RGBA")
    
    overlay_layer = Image.new("RGBA", rgb_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)
    
    alphabet = string.ascii_uppercase
    site_records = []
    
    COLOR_MAP = {
        "hemorrhage": (255, 0, 0), "bleed": (255, 0, 0), "blood": (255, 0, 0),
        "exudate": (255, 255, 0), "microaneurysm": (255, 165, 0), "aneurysm": (255, 165, 0),
        "cotton": (255, 255, 255), "wool": (255, 255, 255)
    }
    
    for i, lesion in enumerate(lesions):
        site_letter = alphabet[i % len(alphabet)]
        raw_label = lesion.get("label", "unknown").lower()
        box_2d = lesion.get("box_2d", [0, 0, 0, 0])
        
        base_color = (0, 255, 0)
        display_label = raw_label.title()
        color_name = "Green"
        
        for key, color in COLOR_MAP.items():
            if key in raw_label:
                base_color = color
                color_name = "Yellow" if color == (255, 255, 0) else ("Red" if color == (255, 0, 0) else ("Orange" if color == (255, 165, 0) else "White"))
                if "light" in raw_label: display_label = "Light Exudate"
                elif "hard" in raw_label: display_label = "Hard Exudate"
                break
                
        ymin, xmin, ymax, xmax = box_2d
        x1, y1 = int((xmin / 1000) * width), int((ymin / 1000) * height)
        x2, y2 = int((xmax / 1000) * width), int((ymax / 1000) * height)
        
        draw.rectangle([x1, y1, x2, y2], fill=base_color + (80,), outline=base_color + (255,), width=3)
        badge_text = f"Site {site_letter}: {display_label}"
        badge_width = len(badge_text) * 7 + 12
        badge_y = max(5, y1 - 20) if (y1 - 20) > 5 else y1 + 5
        
        draw.rectangle([x1, badge_y, x1 + badge_width, badge_y + 16], fill=base_color + (255,))
        text_color = (0, 0, 0, 255) if base_color == (255, 255, 0) else (255, 255, 255, 255)
        draw.text((x1 + 6, badge_y + 1), badge_text, fill=text_color)
        
        site_records.append({"site": f"Site {site_letter}", "color": color_name, "type": display_label, "coordinates": f"X: {x1}-{x2}, Y: {y1}-{y2}"})
        
    base_rgba = rgb_image.convert("RGBA")
    final_output = Image.alpha_composite(Image.alpha_composite(base_rgba, vessel_layer), overlay_layer)
    return final_output.convert("RGB"), site_records


# --- STREAMLIT UI ---
uploaded_file = st.file_uploader("Upload Fundus Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.analysis = None
        st.session_state.chat_history = []
        st.session_state.last_uploaded = uploaded_file.name

    original_image = Image.open(uploaded_file)
    clahe_only_image = preprocess_only_clahe(original_image)
    super_enhanced_image = preprocess_super_vision(original_image)
    
    st.write("### 🎛️ Diagnostic Preprocessing Pipeline Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.image(original_image, caption="1. Original Raw Upload", use_container_width=True)
    with col2: 
        st.image(clahe_only_image, caption="2. CLAHE Only (Lighting Balanced)", use_container_width=True)
    with col3: 
        st.image(super_enhanced_image, caption="3. Super-Vision (CLAHE + Vibrance + Sharpness)", use_container_width=True)
    
    if st.button("Run Diagnostics & Open Consultation Room", type="primary"):
        if not api_key:
            st.error("Configuration Error: API Key not found.")
        else:
            with st.spinner("Analyzing cross-referenced image channels natively..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    # Passing both raw and super-enhanced images allows full spatial contrast cross-referencing
                    prompt = (
                        "Analyze these two representations of the same eye (the raw photo and the super-enhanced version combining CLAHE, vibrance color manipulation, and unsharp masking). "
                        "CRITICAL STRUCTURAL INSTRUCTION: Look at the peripheral borders and outer quadrant circles of the retina. "
                        "The super-enhanced imagery has amplified chemical colors and edge definitions specifically to uncover micro-lesions trying to hide in vignetted or blurry regions.\n\n"
                        "Classify using exact ICDR criteria:\n"
                        "- No DR: Zero abnormalities.\n"
                        "- Mild NPDR: Microaneurysms only.\n"
                        "- Moderate NPDR: Explicit presence of hard/light exudates, cotton wool spots, or intraretinal blot hemorrhages.\n"
                        "Locate abnormalities and output boxes as [ymin, xmin, ymax, xmax] normalized to 0-1000."
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=[original_image, super_enhanced_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json", response_schema=RetinalAnalysis,
                        ),
                    )
                    
                    st.session_state.analysis = json.loads(response.text)
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": f"**Diagnostic Assessment:** {st.session_state.analysis['dr_stage']}\n\n**Initial Justification:** {st.session_state.analysis['justification']}"}
                    ]
                    
                except Exception as e:
                    st.error(f"Processing error: {e}")

    # --- DISPLAY DIAGNOSTICS & CONVERSATIONAL FRAME ---
    if st.session_state.analysis is not None:
        st.write("---")
        # Map onto the super enhanced image so boundaries are clear to the user
        mapped_img, records = map_retina(super_enhanced_image, st.session_state.analysis.get("lesions", []))
        
        ui_left, ui_right = st.columns([1.2, 0.8])
        
        with ui_left:
            st.subheader("Interactive Retina Map")
            st.image(mapped_img, caption="Super-Vision Combined Pathology Map", use_container_width=True)
            
            st.metric(label="Calculated ICDR Class", value=st.session_state.analysis["dr_stage"])
            st.info(f"**AI Notes:** {st.session_state.analysis['justification']}")
            
            if records:
                st.write("### 🔍 Plotted Sites Key")
                for r in records:
                    st.markdown(f"**{r['site']}** | {r['type']} ({r['color']}) | Location: `{r['coordinates']}`")
        
        with ui_right:
            st.subheader("💬 Clinical Chat & Feedback Room")
            st.caption("Guide the model toward missed edge variations:")
            
            chat_container = st.container(height=420)
            with chat_container:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
            
            if user_input := st.chat_input("Ex: 'Recalculate. The sharpened edge on the right is an exudate pattern.'"):
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_input)
                
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Analyzing targets..."):
                    client = genai.Client(api_key=api_key)
                    
                    chat_context_prompt = (
                        "You are reviewing an updated clinical fundus report. The user has access to a super-preprocessed visualization matrix. "
                        "Look at the uploaded original image and super-enhanced vision frame closely. "
                        "Respond to the user's feedback. If they call attention to a specific sharpened edge structure or vibrant color point that you missed, "
                        "re-verify the region immediately and offer an adjusted diagnostic perspective based on the clinical data. Keep responses brief."
                    )
                    
                    conversation_payload = [original_image, super_enhanced_image, chat_context_prompt]
                    for history_item in st.session_state.chat_history:
                        conversation_payload.append(f"{history_item['role']}: {history_item['content']}")
                        
                    chat_response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=conversation_payload
                    )
                    
                    ai_reply = chat_response.text
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.write(ai_reply)
                            
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
