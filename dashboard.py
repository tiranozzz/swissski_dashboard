import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from Queries import get_hist_data, get_test_files, read_file
from ConnectUtil import DatabaseConnection, Config



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
        sel_row = AgGrid(
            data=file_df, 
            gridOptions=config["ag_grid"]["grid_options"],
            #height=300, 
            width='100%',
            data_return_mode="AS_INPUT", 
            update_mode="GRID_CHANGED",
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
            enable_enterprise_modules=False
        )

# Filter data
st.header("Historical data")
if len(sel_row["selected_rows"]) > 0:
    sel_row_data = sel_row["selected_rows"][0]
    attribute_data = histData[histData["ATTRIBUTE"] == sel_row_data.get("Attribute")]
    # Plot data
    # fig = plt.figure()
    # plt.title("Histogram population")
    # plt.hist(attribute_data["NUMERICALVALUE"], bins=20, color="skyblue")
    # plt.plot(sel_row_data.get("Value"), 0, "r|", markersize=15, label=(sel_row_data.get("Athlete") + " (" + str(sel_row_data.get("Value")) + ")"))
    # plt.xlabel(sel_row_data.get("Attribute"))
    # plt.ylabel("#Tests")
    # plt.legend()
    # st.pyplot(fig)

    fig = px.histogram(attribute_data, x="NUMERICALVALUE", title="Histogram population")
    fig.add_vline(x=sel_row_data.get("Value"), line_width=10, line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# st.subheader("Given athlete")
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.scatter(histData[histData["NAME"] == "Luchsinger Jan"]["TESTDATE"], histData[histData["NAME"] == "Luchsinger Jan"]["NUMERICALVALUE"])
# ax.set_title("Past test values")
# ax.set_xlabel(attribute_sel)
# st.pyplot(fig)


