import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import pyttsx3

# 🧠 Caption generation function
def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):
    # Load models
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Preprocess image
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)

    # Generate caption
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    # Display
    st.image(image_path, use_column_width=True, caption="📷 Uploaded Image")
    st.markdown(f"<div class='caption-box'>📝 <strong>Generated Caption:</strong><br>{caption}</div>", unsafe_allow_html=True)

    # Voice-over
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()


# 🎨 Streamlit App Interface
def main():
    st.set_page_config(page_title="🖼️ Image Caption Generator", layout="centered")

    # Inject Custom CSS
    st.markdown("""
        <style>
            .main {
                background-color: #f0f2f6;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            .title {
                text-align: center;
                color: #1f77b4;
                font-size: 2.5em;
                font-weight: 700;
            }
            .caption-box {
                text-align: center;
                color: #333333;
                font-size: 1.2em;
                font-style: italic;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 class='title'>🧠 Image Caption Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload an image to let AI describe it!</p>", unsafe_allow_html=True)

    # Upload
    uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with st.spinner("🧠 Thinking..."):
            # Save image
            with open("uploaded_image.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Model paths
            model_path = "model.keras"
            tokenizer_path = "tokenizer.pkl"
            feature_extractor_path = "feature_extractor.keras"

            # Generate and display
            generate_and_display_caption("uploaded_image.jpg", model_path, tokenizer_path, feature_extractor_path)
        st.success("✅ Caption generated successfully!")

    st.markdown("<hr style='margin-top:2rem;'>", unsafe_allow_html=True)
    st.markdown("<center><small>Built with ❤️ using TensorFlow & Streamlit</small></center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
