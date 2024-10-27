import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set page config
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# Function to load the tokenizer
@st.cache_resource
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Function to load the models
@st.cache_resource
def load_models(model_path):
    # Load the trained caption model
    caption_model = load_model(model_path)
    
    # Load and modify VGG16 model
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    
    return caption_model, vgg_model

# Function to convert integer to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    
    # Remove start and end tokens
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Function to extract features from image
def extract_features(image, model):
    # Resize image
    image = image.resize((224, 224))
    # Convert image pixels to numpy array
    image = img_to_array(image)
    # Reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preprocess image for VGG
    image = preprocess_input(image)
    # Extract features
    features = model.predict(image, verbose=0)
    return features

def main():
    st.title("🖼️ Image Caption Generator")
    st.write("Upload an image and get its caption!")
    
    # Load models and tokenizer
    try:
        caption_model, vgg_model = load_models('working/best_model.keras')
        tokenizer = load_tokenizer('working/tokenizer.pkl')
        max_length = 35  # Same as in training
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Add a button to generate caption
            if st.button('Generate Caption'):
                with st.spinner('Generating caption...'):
                    # Extract features
                    features = extract_features(image, vgg_model)
                    
                    # Generate caption
                    caption = predict_caption(caption_model, features, tokenizer, max_length)
                    
                    # Display caption
                    st.success("Generated Caption:")
                    st.write(f"📝 {caption.capitalize()}")
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.write("Please make sure the model and tokenizer files are in the correct location.")

if __name__ == "__main__":
    main()