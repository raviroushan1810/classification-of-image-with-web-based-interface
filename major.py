# run this code by:
# python -m streamlit run major.py
# Importing modules
import streamlit as st
import numpy as np
import cv2

# Creating model
# kerel_size = specifying the height and width of the 2D convolution window.
# Convolution window : A convolution layer defines a window by which we examine a subset of the image.
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Text or heading's
st.markdown("<h2 style='text-align: center; color: green;'><b>CHECK</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>IMAGE</b></h5>", unsafe_allow_html=True)
# Just for separation
st.write("")
from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
x  = 'Dataset'
path = Path(x)
data = ImageDataLoaders.from_folder(path, train='train_folder', valid='valid_folder',
                                   valid_pct=0.2, item_tfms=Resize(224),
                                   batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
                                   num_workers=4)
learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir = Path(''),path = Path("."))
learn.load("stage-1")

uploaded_file = st.file_uploader("Choose a file")

if st.button('Start ANALYSIS(Click here)'):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        p=learn.predict(opencv_image)
        # Now do something with the image! For example, let's display it:
        st.markdown("<h5 style='text-align: left; color: grey;'>Image Type:<b style='color: green;'>Processed</b></h5>", unsafe_allow_html=True)
        st.image(opencv_image, channels="BGR", caption="Processed Image",width=400)
        flag=False
        set_accuracy=0.70
        for predict in p[2]:
            if predict>set_accuracy:
                flag=True
        message=p[0]
        if flag==False:
            message="This model is not trained for this image. accuracy is too low.\nThis model detect the image as "+p[0]
        st.markdown("<h5 style='text-align: center; color: grey;'>Image Class:<b style='color: red;'>{}</b></h5>".format(message), unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: grey;'>Extra Info:<b style='color: green;'>{}</b></h5>".format(p), unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")



