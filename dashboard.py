import streamlit as st
from Queries import get_hist_data
from ConnectUtil import DatabaseConnection, Config
#import numpy as np
#import matplotlib.pyplot as plt

# Load Config
with st.spinner("Loading configuration..."):
    c = Config()
# Establish Database Connection
with st.spinner("Establishing database connection..."):
    dbConn = DatabaseConnection(config=c).get_connection()
# Loading historical data
with st.spinner("Loading historical data..."):
    histData = get_hist_data(dbConn, config=c)

st.text(len(histData))


def get_test_files():
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    return res

def read_file(file):
    df = pd.read_csv(file, sep=";")
    return df

# files = get_test_files()
# st.title("Swissski Dashboard")
# st.header("File list")
# for f in files:
#     st.subheader(f)
#     file_content = read_file(dir_path + f)
#     st.table(file_content)

# st.header("Historical data")
# st.subheader("Population")
# attribute_sel = st.selectbox(
#     'Select test-attribute',
#     ('Gewicht', 'Groesse', 'BMI'))
# st.write('You selected:', attribute_sel)
# histData = get_hist_data(dbConn=dbConn)
# print(histData.head())
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].set_title("Distribution (male)")
# ax[0].hist(histData[histData["GENDER"] == "M"]["NUMERICALVALUE"], bins=20)
# ax[0].set_xlabel(attribute_sel)
# ax[1].set_title("Distribution (female)")
# ax[1].hist(histData[histData["GENDER"] == "F"]["NUMERICALVALUE"], bins=20, color="orange")
# ax[1].set_xlabel(attribute_sel)
# st.pyplot(fig)

# st.subheader("Given athlete")
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.scatter(histData[histData["NAME"] == "Luchsinger Jan"]["TESTDATE"], histData[histData["NAME"] == "Luchsinger Jan"]["NUMERICALVALUE"])
# ax.set_title("Past test values")
# ax.set_xlabel(attribute_sel)
# st.pyplot(fig)


