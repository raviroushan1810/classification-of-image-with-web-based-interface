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
# Set background image
# st.set_page_config(page_title='Streamlit App', page_icon=':chart:', layout='wide',
#                    initial_sidebar_state='expanded', background_image='path/to/your/image.jpg')
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
        # Define the data for the pie chart
        print(p[2])
        # tensora = [p[2].tensor([1]), p.tensor([2]), p.tensor([3]), p.tensor([4]), p.tensor([5]), p.tensor([6])]
        tensora = [p[2][0], p[2][1], p[2][2], p[2][3], p[2][4], p[2][5]]
        # Convert the values to integers using scientific notation
        int_list = [value*100 for value in tensora]
        print("list",int_list)
        sizes1 = [round(float(value)) for value in int_list]
        sizes = sizes1
        print("sizes",sizes)

        labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        # Now do something with the image! For example, let's display it:
        # Create the pie chart
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFFF99','#8F00FF']

        # Create the pie chart
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

        # Add labels and values outside the pie chart
        bbox_props = dict(boxstyle='square,pad=0.3', fc='white', ec='black', lw=0.5)
        plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        for autotext, text, wedge in zip(autotexts, texts, wedges):
            percentage = autotext.get_text()
            x = autotext.get_position()[0]
            y = autotext.get_position()[1]
            
            ax.annotate(percentage, (x, y), xytext=(2*x,2*y), textcoords='data', ha='center', va='center_baseline', bbox=bbox_props)
            autotext.set_visible(False)
            text.set_visible(False)
            # Set aspect ratio to be equal so that pie is drawn as a circle
        ax.axis('equal')
        st.markdown("<h5 style='text-align: left; color: grey;'>Image Type:<b style='color: green;'>Processed</b></h5>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.image(opencv_image, channels="BGR", caption="Processed Image",width=400)
        flag=False
        set_accuracy=0.70
        for predict in p[2]:
            if predict>set_accuracy:
                flag=True
        message=p[0]
        if flag==False:
            message="This model is not trained for multi class image. accuracy is too low.\nThis model detect the image as multiclass"+p[0]
        st.markdown("<h5 style='text-align: center; color: grey;'>Image Class:<b style='color: red;'>{}</b></h5>".format(message), unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: grey;'>Extra Info:<b style='color: green;'>{}</b></h5>".format(p[0]), unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")



