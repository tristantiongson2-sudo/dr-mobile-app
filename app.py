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
st.title("👁️ DR Mobile Assistant with Advanced Quality Restoration")

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

# --- SIDEBAR INTERACTIVE QUALITY CONTROLS ---
st.sidebar.header("🛠️ Image Quality & Restoration")
enable_enhancement = st.sidebar.checkbox("Enable Quality Enhancer", value=True)

if enable_enhancement:
    denoise_strength = st.sidebar.slider(
        "Denoise Strength (Bilateral)", 
        0, 10, 3, 
        help="Smooths out camera sensor grain without blurring lesion edges."
    )
    clahe_clip = st.sidebar.slider(
        "Contrast Stretch (LAB-CLAHE)", 
        0.0, 4.0, 1.5, step=0.5, 
        help="Boosts structural details in dark/shadowed regions without shifting natural colors."
    )
    gamma_val = st.sidebar.slider(
        "Gamma (Exposure Balance)", 
        0.5, 2.0, 1.0, step=0.1, 
        help="Adjusts brightness. >1.0 brightens shadows; <1.0 tones down bright flash hotspots."
    )
else:
    denoise_strength = 0
    clahe_clip = 0.0
    gamma_val = 1.0

# --- QUALITY RESTORATION PIPELINE ---
def apply_quality_restoration(pil_image, denoise_sigma, clahe_clip_limit, gamma):
    # Convert PIL Image to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # 1. Bilateral Filter (Smooths noise, keeps crisp edges)
    if denoise_sigma > 0:
        img_bgr = cv2.bilateralFilter(
            img_bgr, 
            d=9, 
            sigmaColor=denoise_sigma * 10, 
            sigmaSpace=denoise_sigma * 10
        )
    
    # 2. Gamma Correction (Illumination balance)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img_bgr = cv2.LUT(img_bgr, lookup_table)
        
    # 3. LAB-Space CLAHE (Color-safe local contrast stretch)
    if clahe_clip_limit > 0:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE only on the Lightness (L) channel to keep original colors pure
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        img_bgr = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# --- VIBRANCE & TARGETED 5X YELLOW-BOOST PREPROCESSING ---
def preprocess_vibrance_sharpen(pil_image):
    # Convert PIL Image to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # 1. Convert to HSV to isolate color channels
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Define the HSV range for yellow-ish hues (captures light yellow, faint orange, and greenish-yellow)
    lower_yellow_hue = 12
    upper_yellow_hue = 35
    
    # Create a mask specifically for yellow pixels.
    # Minimum threshold for Saturation (s > 20) and Value (v > 20) avoids boosting background noise.
    yellow_mask = (h >= lower_yellow_hue) & (h <= upper_yellow_hue) & (s > 20) & (v > 20)
    
    # Convert Saturation to float to prevent mathematical overflow during 5x multiplication
    s_float = s.astype(np.float32)
    s_float[yellow_mask] = s_float[yellow_mask] * 5.0
    s_boosted = np.clip(s_float, 0, 255).astype(np.uint8)
    
    # Give the yellow pixels a 1.2x brightness (Value) boost so they glow slightly against dark red tissue
    v_float = v.astype(np.float32)
    v_float[yellow_mask] = v_float[yellow_mask] * 1.2
    v_boosted = np.clip(v_float, 0, 255).astype(np.uint8)
    
    # Merge the boosted yellow channels back together
    img_vibrant = cv2.cvtColor(cv2.merge((h, s_boosted, v_boosted)), cv2.COLOR_HSV2BGR)
    
    # 2. Unsharp Masking (Micro-Edge Sharpening)
    gaussian_blur = cv2.GaussianBlur(img_vibrant, (5, 5), 1.5)
    img_sharpened = cv2.addWeighted(img_vibrant, 1.6, gaussian_blur, -0.6, 0)
    
    # Convert back to PIL RGB
    final_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

# --- QUADRANT SPLITTING ENGINE ---
def split_into_quadrants(pil_image):
    width, height = pil_image.size
    w_half, h_half = width // 2, height // 2
    
    # Crop into standard coordinate quadrants
    q_top_left = pil_image.crop((0, 0, w_half, h_half))
    q_top_right = pil_image.crop((w_half, 0, width, h_half))
    q_bottom_left = pil_image.crop((0, h_half, w_half, height))
    q_bottom_right = pil_image.crop((w_half, h_half, width, height))
    
    return q_top_left, q_top_right, q_bottom_left, q_bottom_right

# --- PATHOLOGY ONLY MAPPING (UGAT MAPPING REMOVED) ---
def map_retina(pil_image, lesions):
    rgb_image = pil_image.convert("RGB")
    width, height = rgb_image.size
    
    # Dynamic transparent canvas layer for lesion polygons and labels
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
    # Composite the bounding boxes directly onto the image without any vessel layers
    final_output = Image.alpha_composite(base_rgba, overlay_layer)
    return final_output.convert("RGB"), site_records


# --- STREAMLIT UI ---
uploaded_file = st.file_uploader("Upload Fundus Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.analysis = None
        st.session_state.chat_history = []
        st.session_state.last_uploaded = uploaded_file.name

    original_image = Image.open(uploaded_file)
    
    # Step 1: Run restoration (Denoise, Exposure, Dynamic contrast stretching)
    restored_image = apply_quality_restoration(original_image, denoise_strength, clahe_clip, gamma_val)
    
    # Step 2: Run diagnostic features (Vibrance, Yellow-channel boost, Edge sharpening)
    processed_image = preprocess_vibrance_sharpen(restored_image)
    
    # Step 3: Split the fully processed image into 4 quadrant frames
    q_tl, q_tr, q_bl, q_br = split_into_quadrants(processed_image)
    
    st.write("### 🎛️ Clinical Preprocessing & Quadrant Division")
    col_orig, col_proc = st.columns(2)
    with col_orig: 
        st.image(original_image, caption="1. Original Raw Retinal File", use_container_width=True)
    with col_proc: 
        st.image(processed_image, caption="2. Restored & Yellow-Boosted (5x) Image", use_container_width=True)
    
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
            st.image(mapped_img, caption="Combined Quadrant Pathology Map (Without Vessels)", use_container_width=True)
            
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
                        "(Top-Left, Top-Right, Bottom-Left, Bottom-Right) with custom yellow-boosted vibrance and sharpening to find hidden lesions.\n\n"
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
