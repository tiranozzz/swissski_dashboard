import streamlit as st
import matplotlib.pyplot as plt
from Queries import get_hist_data, get_test_files, read_file
from ConnectUtil import DatabaseConnection, Config
#import numpy as np

st.set_page_config(layout="centered", page_title="Swiss Ski Ampelsystem")
# Load Config
with st.spinner("Loading configuration..."):
    c = Config()
    config, current_path = c.config, c.path
# Establish Database Connection
with st.spinner("Establishing database connection..."):
    dbConn = DatabaseConnection(config=config).get_connection()
# Loading historical data
with st.spinner("Loading historical data..."):
    histData = get_hist_data(dbConn, config=config)

st.title("Swissski Dashboard")
# Show data files
with st.spinner("Loading data files..."):
    data_path = current_path + "\\data"
    st.header("File list")
    files = get_test_files(dir_path=data_path)
    for f in files:
        st.subheader(f)
        file_df = read_file(data_path + "\\" + f)
        st.dataframe(file_df)

st.header("Historical data")
st.subheader("Population")
attribute_sel = st.radio('Select test-attribute',(config["test_attributes"].keys()))
st.write('You selected:', attribute_sel)
# Filter data
attribute_data = histData[histData["ATTRIBUTE"] == attribute_sel]
# Plot data
col1, col2 = st.columns(2)
with col1:
   fig = plt.figure()
   st.header("Distribution male")
   plt.hist(attribute_data[attribute_data["GENDER"] == "M"]["NUMERICALVALUE"], bins=20)
   plt.xlabel(attribute_sel)
   st.pyplot(fig)

with col2:
   fig = plt.figure()
   st.header("Distribution female")
   plt.hist(attribute_data[attribute_data["GENDER"] == "F"]["NUMERICALVALUE"], bins=20, color="orange")
   plt.xlabel(attribute_sel)
   st.pyplot(fig)

# st.subheader("Given athlete")
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.scatter(histData[histData["NAME"] == "Luchsinger Jan"]["TESTDATE"], histData[histData["NAME"] == "Luchsinger Jan"]["NUMERICALVALUE"])
# ax.set_title("Past test values")
# ax.set_xlabel(attribute_sel)
# st.pyplot(fig)


