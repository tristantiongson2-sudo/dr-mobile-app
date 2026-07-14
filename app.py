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
st.title("👁️ DR Mobile Assistant with Quadrant-Split Visual Engineering")

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

# --- UPGRADED: VIBRANCE & UNSHARP MASK (NO CLAHE) ---
def preprocess_vibrance_sharpen(pil_image):
    # Convert PIL Image to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # 1. HSV Saturation Tuning (Vibrance Boost)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Boost saturation to isolate faint yellow exudates from the red retinal background
    s_boosted = cv2.addWeighted(s, 1.4, np.zeros(s.shape, s.dtype), 0, 15)
    img_vibrant = cv2.cvtColor(cv2.merge((h, s_boosted, v)), cv2.COLOR_HSV2BGR)
    
    # 2. Unsharp Masking (Micro-Edge Sharpening)
    gaussian_blur = cv2.GaussianBlur(img_vibrant, (5, 5), 1.5)
    img_sharpened = cv2.addWeighted(img_vibrant, 1.6, gaussian_blur, -0.6, 0)
    
    # Convert back to PIL RGB
    final_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

# --- NEW: QUADRANT SPLITTING ENGINE ---
def split_into_quadrants(pil_image):
    width, height = pil_image.size
    w_half, h_half = width // 2, height // 2
    
    # Crop into standard coordinate quadrants
    q_top_left = pil_image.crop((0, 0, w_half, h_half))
    q_top_right = pil_image.crop((w_half, 0, width, h_half))
    q_bottom_left = pil_image.crop((0, h_half, w_half, height))
    q_bottom_right = pil_image.crop((w_half, h_half, width, height))
    
    return q_top_left, q_top_right, q_bottom_left, q_bottom_right

# --- HYBRID PATHOLOGY MAPPING ---
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
    processed_image = preprocess_vibrance_sharpen(original_image)
    
    # Split the vibrant/sharpened image into quadrants
    q_tl, q_tr, q_bl, q_br = split_into_quadrants(processed_image)
    
    st.write("### 🎛️ Clinical Preprocessing & Quadrant Division")
    col_orig, col_proc = st.columns(2)
    with col_orig: 
        st.image(original_image, caption="1. Original Raw Retinal File", use_container_width=True)
    with col_proc: 
        st.image(processed_image, caption="2. Vibrance & Edge Sharpened Image", use_container_width=True)
    
    # Display the quadrants in a beautiful 4-column row
    st.write("#### 🧩 Quadrant breakdown (Enhanced for Detailed Local Analysis)")
    q_cols = st.columns(4)
    with q_cols[0]: st.image(q_tl, caption="Quadrant I (Top-Left)", use_container_width=True)
    with q_cols[1]: st.image(q_tr, caption="Quadrant II (Top-Right)", use_container_width=True)
    with q_cols[2]: st.image(q_bl, caption="Quadrant III (Bottom-Left)", use_container_width=True)
    with q_cols[3]: st.image(q_br, caption="Quadrant IV (Bottom-Right)", use_container_width=True)
    
    if st.button("Run Multi-Quadrant Analysis", type="primary"):
        if not api_key:
            st.error("Configuration Error: API Key not found.")
        else:
            with st.spinner("Analyzing all 4 quadrants concurrently..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    # Passing original, overall processed, and all four individual quadrants
                    prompt = (
                        "You are an elite clinical retinal analyst evaluating fundus frames.\n\n"
                        "To prevent missing faint, hidden pathologies near the boundaries, we have processed this image and split it into four distinct quadrants:\n"
                        "- Quadrant 1: Top-Left\n"
                        "- Quadrant 2: Top-Right\n"
                        "- Quadrant 3: Bottom-Left\n"
                        "- Quadrant 4: Bottom-Right\n\n"
                        "CRITICAL SEARCH INSTRUCTION:\n"
                        "Examine each of the four provided quadrants individually. Specifically search the outer edges of each quadrant for very faint yellow spots (light/hard exudates) or cotton wool structures.\n\n"
                        "COORDINATE MAPPING RULE:\n"
                        "You must calculate and output the coordinates (`box_2d`) normalized relative to the coordinate space of the MAIN, OVERALL original image (0 to 1000 scale, format [ymin, xmin, ymax, xmax]). Do not use crop-relative coordinates.\n\n"
                        "Classify using exact ICDR criteria:\n"
                        "- No DR: Zero abnormalities.\n"
                        "- Mild NPDR: Microaneurysms only.\n"
                        "- Moderate NPDR: Explicit presence of hard/light exudates, cotton wool spots, or intraretinal blot hemorrhages."
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=[original_image, processed_image, q_tl, q_tr, q_bl, q_br, prompt],
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

    # --- DISPLAY DIAGNOSTICS & CONVERSATIONAL CHAT ---
    if st.session_state.analysis is not None:
        st.write("---")
        # Draw bounding boxes onto the processed vibrance image for perfect clinical clarity
        mapped_img, records = map_retina(processed_image, st.session_state.analysis.get("lesions", []))
        
        ui_left, ui_right = st.columns([1.2, 0.8])
        
        with ui_left:
            st.subheader("Interactive Retina Map")
            st.image(mapped_img, caption="Combined Quadrant Pathology Map", use_container_width=True)
            
            st.metric(label="Calculated ICDR Class", value=st.session_state.analysis["dr_stage"])
            st.info(f"**AI Notes:** {st.session_state.analysis['justification']}")
            
            if records:
                st.write("### 🔍 Plotted Sites Key")
                for r in records:
                    st.markdown(f"**{r['site']}** | {r['type']} ({r['color']}) | Location: `{r['coordinates']}`")
        
        with ui_right:
            st.subheader("💬 Clinical Chat & Feedback Room")
            st.caption("Point out quadrant-specific details directly to the AI:")
            
            chat_container = st.container(height=420)
            with chat_container:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
            
            if user_input := st.chat_input("Ex: 'Examine Quadrant II again. The sharpened spot on the top right is a soft exudate.'"):
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_input)
                
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Re-evaluating frames with your clinical feedback..."):
                    client = genai.Client(api_key=api_key)
                    
                    chat_context_prompt = (
                        "You are reviewing an updated clinical fundus report. The user has access to a segmented view consisting of 4 quadrants "
                        "(Top-Left, Top-Right, Bottom-Left, Bottom-Right) with custom vibrance and sharpening to find hidden lesions.\n\n"
                        "Respond to the user's feedback. If they call attention to a specific quadrant structure or color point that you missed, "
                        "re-verify that specific quadrant immediately, explain what you see, and offer an adjusted diagnostic perspective. Keep responses brief."
                    )
                    
                    # Feed all visual data and chat memory back to the model
                    conversation_payload = [original_image, processed_image, q_tl, q_tr, q_bl, q_br, chat_context_prompt]
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
