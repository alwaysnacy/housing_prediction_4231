import streamlit as st
import numpy as np
import pickle
import keras
from keras.models import load_model
import datetime
from pickle import load
import pandas as pd
import requests
from geopy.distance import geodesic

towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN']

flat_type = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM"]

storey_range = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
       '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', 
       '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']

flat_models = ['Improved', 'New Generation', 'DBSS', 'Standard', 'Apartment',
       'Simplified', 'Model A', 'Premium Apartment', 'Adjoined flat',
       'Model A-Maisonette', 'Maisonette', 'Type S1', 'Type S2',
       'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Maisonette',
       'Multi Generation', 'Premium Apartment Loft', '2-room']


def get_coordinates(postal_code):
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+postal_code+'&returnGeom=Y&getAddrDetails=N&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return [resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']]
    else:
        return []

def load_model_from_file():
    model = load_model('DL_model.h5')
    
    return model

data = load_model_from_file()

with open('encoder.pkl', 'rb') as file:
    encoder = load(file)

def show_prediction_page():

    if 'min_mrt' not in st.session_state:
        st.session_state.min_mrt = 0
    if 'min_mall' not in st.session_state:
        st.session_state.min_mall = 0

    def get_min_distances(postal_code):
        if (len(str(int(postal_code))) != 6):
            return 0
        coordinates = get_coordinates(str(int(postal_code)))
        if (len(coordinates) == 0):
            return 0
        lat, long = coordinates
        print("lat, long", lat, long)
        malls = pd.read_csv("mall_data.csv")
        mall_cor = list(zip(malls['latitude'], malls['longitude']))
        mrt = pd.read_csv("station_data.csv")
        print(mrt.columns)
        mrt_cor = list(zip(mrt[' latitude'], mrt[' longtitude']))
        address = list(zip([lat], [long]))

        list_of_dist_mrt = []
        for destination in range(0, len(mrt_cor)):
            list_of_dist_mrt.append(geodesic(address, mrt_cor[destination]).meters)
        
        list_of_dist_mall = []
        for destination in range(0, len(mall_cor)):
            list_of_dist_mall.append(geodesic(address, mall_cor[destination]).meters)

        mrt_shortest = (min(list_of_dist_mrt))
        mall_shortest = (min(list_of_dist_mall))
        st.session_state.min_mrt = mrt_shortest
        st.session_state.min_mall = mall_shortest
        return 1

    st.title("Housing Price Prediction")
    st.write("""### We need some information of your house to give the best prediction.""")
    town = st.selectbox("Town", towns)
    flat_model = st.selectbox("Flat Model", flat_models)

    model = load_model_from_file()

    with open('encoder.pkl', 'rb') as file:
        encoder = load(file)   
    with open('scaler.pkl', 'rb') as file:
        scaler = load(file) 
    with open('lr_model.pkl', 'rb') as file:
        lr_model = load(file)

    storey = st.slider("Storey", 1, 51)
    storey_index = (storey - 1) // 3
    num_room = st.radio("Number of rooms", [1, 2, 3, 4])
    postal_code = 100050
    postal_code = st.number_input(label='Your postal code', value=0)
    is_success = 1
    if (postal_code > 0):
        is_success = get_min_distances(postal_code)
    
    if (is_success == 0):
        st.write("Invalid postal code")
    else:
        st.write("Min distance to the nearest MRT (m)", st.session_state.min_mrt)
        st.write("Min distance to the nearest mall (m)", st.session_state.min_mall)

    lease_commence_date = st.date_input(
     "Commence date",
    min_value=datetime.date(1950, 1, 1))
    lease_year = lease_commence_date.strftime("%Y")
    #lease_commence_date = st.number_input("Commence date", min_value=1950, max_value=2022, value=1950)

    now = datetime.datetime.now()
    now_year = now.year
    now_month = now.month
    # latest time in the data is 2021-07
    difference = (now_year - 2021) * 12 + (now_month - 7)

    dl_ok = st.button('Predict price using Deep Learning model')
    lr_ok = st.button('Predict price using Linear Regression model')

    data_dict = {"month": ['2021-07'], "town": [town], "flat_type": [flat_type[num_room - 1]], "storey_range": [storey_range[storey_index]], "floor_area_sqm": [44.0], "flat_model": [flat_model],	"lease_commence_date": [lease_year],"min_dist_mrt_in_m": [st.session_state.min_mrt],"min_dist_mall_in_m": [st.session_state.min_mall]}

    if dl_ok:
        data_df = pd.DataFrame(data_dict)
        X = encoder.transform(data_df).toarray()
        # the encoder for the month
        X[0][0] = X[0][0] + difference
        scaled_X = scaler.transform(X)
        price = model.predict(scaled_X)

        st.subheader(f"The estimated price is ${round(price[0][0], 2)}")
    
    if lr_ok:
        data_dict = {"month": ['2021-07'], "town": [town], "flat_type": [flat_type[num_room - 1]], "storey_range": [storey_range[storey_index]], "floor_area_sqm": [44.0], "flat_model": [flat_model],	"lease_commence_date": [lease_year],"min_dist_mrt_in_m": [994.360],"min_dist_mall_in_m": [993.885]}
        data_df = pd.DataFrame(data_dict)
        X = encoder.transform(data_df).toarray()
        price = lr_model.predict(X)

        st.subheader(f"The estimated price is ${round(price[0], 2)}")