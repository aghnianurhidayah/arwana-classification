# run this code by command prompt, "streamlit run (this path file)"

import numpy as np
import pickle
import streamlit as st
import tensorflow.keras.utils as image
from PIL import Image

loaded_model = pickle.load(open('trained_model_arwana_classification.sav', 'rb')) 

def classification(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = loaded_model.predict(images, batch_size=10)

    if classes[0,0]:
        result = 'Arwana Golden'
    elif classes[0,1]:
        result = 'Arwana Hitam'
    elif classes[0,2]:
        result = 'Arwana Merah'
    elif classes[0,3]:
        result = 'Arwana Silver'

    return result

def main():
    st.title("Arwana Classification")

    uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)
    
    clf = ''
    if st.button('Classification Result'):
        if uploaded_file is None:
            st.text("Please upload an a image")
        else:
            clf = classification(uploaded_file)
            st.success(clf)

if __name__ == '__main__':
    main()
