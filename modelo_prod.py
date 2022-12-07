#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from io import StringIO 
from tensorflow.keras.models import load_model
import cv2
import streamlit as st
import os
from  PIL import Image, ImageEnhance

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
face_model = cv2.CascadeClassifier(cv2.data.haarcascades
                                   + 'haarcascade_frontalface_default.xml'
                                   )
mask_label = {0: 'MASK', 1: 'NO MASK'}
dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}  # rectangle color

model = load_model('https://drive.google.com/file/d/1VugKC4aG_BFL64TKZW2GUcaS43PfjF7h/view?usp=sharing')
def plot_image(image, subplot):
    plt.subplot(*subplot)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    st.image(image)


def predict_image(image_dir):
    
    img = cv2.cvtColor(image_dir, cv2.IMREAD_GRAYSCALE
)
    faces = face_model.detectMultiScale(img, scaleFactor=1.1,
            minNeighbors=4)

    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = out_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop).argmax()
        cv2.rectangle(out_img, (x, y), (x + w, y + h),
                      dist_label[mask_result], 1)

    plot_image(out_img, (1, 2, 2))



uploaded_file = st.file_uploader("Suba la imagen")
#Add 'before' and 'after' columns
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    print(image)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Original</p>',unsafe_allow_html=True)
        st.image(image,width=300)  

    with col2:
        opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
        image_resize = cv2.resize(opencvImage, (400, 300))
        st.markdown('<p style="text-align: center;">Predicción</p>',unsafe_allow_html=True)
        predict_image(image_resize)


        #predict_image('maksssksksss86.png')

st.sidebar.markdown('<p class="font">Nuestra primera aplicación</p>', unsafe_allow_html=True)
with st.sidebar.expander("Acerca de nuestro proyecto"):
     st.write("""
       Queriamos realizar un proyecto que nos permitiera mostrar si se tiene mascarilla o no. Autores: Norman Vicente y Michelle Bloomfield
     """)
