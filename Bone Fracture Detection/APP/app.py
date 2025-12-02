import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import joblib

# load model
model = joblib.load(r"C:\code playground\code playground\Data science\Machine_Learning_new\Bone Fracture Detection\bone_fracture_xgb_model.pt")
# get Label 
def get_label(img):
    # convert into gray scale 
    img_gray = img.convert("L")
    # resize in 100, 100
    img_res = img_gray.resize((100,100))
    # convert into numpy array
    img_arr = np.array(img_res).flatten()
    # convert into df and TransPost
    img_df = pd.DataFrame(img_arr).T
    # predict with the model 
    pre = model.predict(img_df)
    # return the value
    if pre == 0: 
        return 'Fractured'
    elif pre == 1: 
        return 'Not Fractured'
    return pre



st.title("Bone Fracture Detection")
st.header("A computer vision project")
file = st.file_uploader("Upload your file", type = 'png')

try: 
    if file is not None: 
        # read image 
        img = Image.open(file)
        # show Image 
        st.image(img, "The uploaded image")
        prediction = get_label(img)
        

        st.write(f"The Bone is : {prediction}")

    else: 
        st.write("Empty File cannot be read")



except Exception as e: 
    st.write(f"{e} occured")

finally: 
    st.write("Thanks for using our service")