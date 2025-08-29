import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Life Expectancy Prediction App")
st.write("This app predicts life expectancy based on various health and demographic factors.")  
st.image("https://media.istockphoto.com/id/1395725088/vector/life-expectancy-blue-gradient-concept-icon.jpg?s=1024x1024&w=is&k=20&c=1AMtGu0kFV4DSeuNsb3-k2ZOzefBHYs-S8sQ88tlZdY=",
         caption="Life Expectancy",
         width=100,
         use_container_width=False)

