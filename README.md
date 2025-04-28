🧠 Image Caption Generator 

📄 Project Description
This is a deep learning-based Image Caption Generator that automatically generates natural language descriptions for uploaded images.
It combines Computer Vision and Natural Language Processing using a CNN Encoder and LSTM Decoder trained on image-caption datasets.

📁 Project Structure
ImageCaptionGenerator/
├── main.py                # Streamlit web app to run the model
├── model.keras             # Trained caption generation model (LSTM Decoder)
├── feature_extractor.keras # CNN model to extract image features
├── tokenizer.pkl           # Tokenizer used for text preprocessing

⚙️ How It Works
feature_extractor.keras: A CNN (e.g., InceptionV3 or ResNet50) extracts a 2048-d feature vector from the uploaded image.

tokenizer.pkl: Maps words to integers and vice versa, allowing the model to predict captions.

model.keras: The LSTM-based decoder model generates a caption word-by-word based on image features.

main.py: A Streamlit app where users upload an image and see the AI-generated caption displayed (and hear it spoken with text-to-speech).

🚀 How to Use
1. Install Required Libraries
Make sure you have Python 3 installed.

Run:
pip install tensorflow streamlit numpy pillow matplotlib pyttsx3

2. Run the Streamlit App
Inside the project folder:
streamlit run main.py

3. Upload an Image
After running, a web app will open in your browser.

Upload a .jpg, .jpeg, or .png image.

The model will display a generated caption for your image.

The caption will also be spoken aloud using pyttsx3!

🔑 Important Notes
Make sure the following files are in the same directory: model.keras, feature_extractor.keras, tokenizer.pkl, and main.py.

The model expects images of size 224x224 (resized internally).

The maximum caption length is set to 34 tokens (you can change max_length inside main.py if needed).

🌟 Features
🖼️ Upload and analyze new images in real-time.

🗣️ Text-to-Speech: The generated caption is read aloud automatically.

🎨 Modern, clean Streamlit UI.

🛠️ Files Explained

File	Purpose
main.py	Main script to run the Streamlit web app
model.keras	Pretrained LSTM caption generator
feature_extractor.keras	CNN model to extract image features
tokenizer.pkl	Vocabulary and word-to-index mapping for caption generation

🎯 Conclusion
This project is ready for demo, presentation, or even real-world minor deployments.
Easy to run, no heavy configuration, beginner-friendly + intermediate-level deep learning project!
