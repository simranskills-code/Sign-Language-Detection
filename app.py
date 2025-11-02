import streamlit as st
from PIL import Image
from detect_utils import detect_signs

st.set_page_config(page_title="Sign Detection App", layout="centered")
st.title("🖐️ Sign Language Detection with YOLOv5")

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

# Upload Image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting..."):
            classes = detect_signs(image)

        st.subheader("✅ Detected Classes:")
        if classes:
            for c in classes:
                st.markdown(f"- **{c}**")
        else:
            st.warning("⚠️ No signs detected.")

# Webcam Input
elif option == "Use Webcam":
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

        with st.spinner("Detecting..."):
            classes = detect_signs(image)

        st.subheader("✅ Detected Classes:")
        if classes:
            for c in classes:
                st.markdown(f"- **{c}**")
        else:
            st.warning("⚠️ No signs detected.")
