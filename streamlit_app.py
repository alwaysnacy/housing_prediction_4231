import streamlit as st
from prediction_page import show_prediction_page
from report_page import show_report_page

page = st.sidebar.selectbox("Report or Predict", ("Report", "Predict"))

if (page == "Predict"): 
    show_prediction_page()
else:
    show_report_page()
