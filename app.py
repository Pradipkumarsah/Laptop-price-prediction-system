import pickle
import numpy as np
import streamlit as st

# import the model
# to load pipeline,sklearn is required.
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Prediction System !!")

# brand
company = st.selectbox("Which Brand of Laptop would you like to Purchase ?", df['Company'].unique())

# type of laptop
typ = st.selectbox('Please Select the Type of the above mentioned Brand .', df['TypeName'].unique())

# Ram
ram = st.selectbox('Please Select the Size of RAM (in GB) which you want to be in the Laptop .', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Please Provide the Weight of the Laptop .')

# Touchscreen
touchscreen = st.selectbox(' Would you want Touchscreen in the Laptop ?', ['No', 'Yes'])

# IPS
ips = st.selectbox('Would you want IPS Full HD Display in the Laptop ?', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Please,Provide the Screen size which you prefer in the Laptop .')

# resolution
resolution = st.selectbox('Please, Select the Screen Resolution Size .',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                           '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('Please,Select the Processor which you want in the Laptop.', df['Cpu brand'].unique())

# memory(HDD and SSD)
hdd = st.selectbox('Select the size of HDD(in GB) .', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('Select the size of SSD(in GB) .', [0, 8, 128, 256, 512, 1024])

# graphic (GPU)
gpu = st.selectbox('Select the Graphics(Graphics Processing Unit,GPU ) .', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('Select the Operating System (OS) .', df['os'].unique())

if st.button('Predict Price '):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, typ, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.subheader("CONGRATULATIONS...!! , YOUR PREDICTED PRICE OF THIS CONFIGURATION IS :- Rs. " + str(int(np.exp(pipe.predict(query)[0]))))
