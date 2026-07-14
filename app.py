import streamlit as st
from PIL import Image, ImageDraw
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="DR Mobile Assistant", layout="centered")
st.title("👁️ DR Mobile Assistant with Lesion Mapping")

# Define our structured data output so Gemini parses coordinates reliably
class Lesion(BaseModel):
    label: str = Field(description="The type of lesion, e.g., microaneurysm, hemorrhage, hard exudate")
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

uploaded_file = st.file_uploader("Upload Fundus Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Keep a clean original copy, and a markup copy to draw on
    original_image = Image.open(uploaded_file)
    markup_image = original_image.copy()
    
    st.image(original_image, caption="Original Fundus Photo", use_container_width=True)
    
    if st.button("Analyze & Map Retina", type="primary"):
        if not api_key:
            st.error("Configuration Error: API Key not found.")
        else:
            with st.spinner("Scanning retina and mapping lesions..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    prompt = (
                        "Perform a rigorous clinical analysis of this fundus image. "
                        "Locate all visible microaneurysms, hemorrhages, and exudates. "
                        "Provide their bounding boxes using [ymin, xmin, ymax, xmax] coordinates normalized to 0-1000."
                    )
                    
                    # Call Gemini forcing structured JSON output matching our RetinalAnalysis class
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=[original_image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=RetinalAnalysis,
                        ),
                    )
                    
                    # Parse the structured JSON output
                    analysis = json.loads(response.text)
                    
                    st.success("Analysis and Mapping Complete!")
                    
                    # --- DRAWING THE HIGHLIGHTS ---
                    width, height = markup_image.size
                    draw = ImageDraw.Draw(markup_image, "RGBA") # RGBA allows transparency
                    
                    detected_any = False
                    for lesion in analysis.get("lesions", []):
                        detected_any = True
                        ymin, xmin, ymax, xmax = lesion["box_2d"]
                        
                        # Convert normalized coordinates back to actual image pixel sizes
                        abs_ymin = int((ymin / 1000) * height)
                        abs_xmin = int((xmin / 1000) * width)
                        abs_ymax = int((ymax / 1000) * height)
                        abs_xmax = int((xmax / 1000) * width)
                        
                        # Set custom colors based on lesion type
                        label_lower = lesion["label"].lower()
                        if "hemorrhage" in label_lower:
                            fill_color = (255, 0, 0, 80)      # Semi-transparent red
                            border_color = (255, 0, 0, 255)
                        elif "exudate" in label_lower:
                            fill_color = (255, 255, 0, 80)    # Semi-transparent yellow
                            border_color = (255, 255, 0, 255)
                        else: # Microaneurysms or others
                            fill_color = (255, 165, 0, 80)    # Semi-transparent orange
                            border_color = (255, 165, 0, 255)
                        
                        # Draw the highlighted rectangle
                        draw.rectangle(
                            [abs_xmin, abs_ymin, abs_xmax, abs_ymax], 
                            fill=fill_color, 
                            outline=border_color, 
                            width=3
                        )
                    
                    # Display the highlighted image
                    st.subheader("Mapped Lesions")
                    st.image(markup_image, caption="AI-Assisted Lesion Highlight Map", use_container_width=True)
                    
                    # Display clinical data
                    st.subheader("Diagnostic Report")
                    st.metric(label="Assigned Stage", value=analysis["dr_stage"])
                    st.write(f"**Clinical Justification:** {analysis['justification']}")
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
