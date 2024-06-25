import streamlit as st
from PIL import Image
from segmenter import Segmenter
from image_combine import combine

def process_image_and_prompt(image, prompt, threshold):
    # Your function logic here
    # Process the image and prompt
    # Return the result
    # Convert the uploaded image to a PIL image
    image = Image.open(image)
    segmenter = Segmenter()
    preds = segmenter.segment(image, prompt)
    out = combine(image, preds, threshold)
    return out

def main():
    st.title("CLIP - Zero Shot Segmentation")
    
    # Upload image
    image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    # Display the uploaded image
    if image is not None:
        st.image(image, caption='Uploaded Image')
    
    # Input prompt
    prompt = st.text_input("Enter objects to segment (comma separated)")

    # Threshold slider
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    
    # Process button
    if st.button("Process"):
        if image is not None and prompt != "":
            prompt = prompt.split(",")
            result = process_image_and_prompt(image, prompt, threshold)
            st.image(result, caption='Result Segmentation')
        else:
            st.write("Please upload an image and enter a prompt.")

if __name__ == "__main__":
    main()