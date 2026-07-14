import streamlit as st
from PIL import Image
from google import genai

st.set_page_config(page_title="DR Mobile Assistant", layout="centered")
st.title("👁️ DR Mobile Assistant")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google GenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload Fundus Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    if st.button("Analyze Retina", type="primary"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar!")
        else:
            with st.spinner("Analyzing..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = "You are an expert ophthalmologist. Identify lesions, state the DR stage, and give a clear clinical justification."
                    response = client.models.generate_content(model='gemini-3.5-flash', contents=[image, prompt])
                    st.success("Analysis Complete!")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")
