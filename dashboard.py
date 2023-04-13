import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from queries import get_hist_data, get_test_files, read_file, get_athlete_data
from models import get_linear_model, get_pred
from connect_util import DatabaseConnection, Config
from datetime import datetime, date
from PIL import Image
import requests
from io import BytesIO


st.set_page_config(layout="wide", page_title="Swiss Ski Ampelsystem")
# Load Config
with st.spinner("Loading configuration..."):
    c = Config()
    config, current_path = c.config, c.path
# Establish Database Connection
with st.spinner("Establishing database connection..."):
    dbConn = DatabaseConnection(config=config).get_connection()
# Loading data from Data Warehouse
with st.spinner("Loading data from Data Warehouse..."):
    hist_data = get_hist_data(dbConn, config=config)
    athlete_data = get_athlete_data(dbConn, config=config)

st.title("Swissski Dashboard")
# Show data files
with st.spinner("Loading data files..."):
    data_path = current_path + "\\data"
    st.header("File list")
    files = get_test_files(dir_path=data_path)
    for f in files:      
        file_df = read_file(data_path + "\\" + f)
        st.markdown(f":open_file_folder: {f} ({len(file_df)} rows). Check results: {len(file_df)} :white_check_mark: | 0 :warning: | 0 :exclamation:")
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

# Historical data
if len(sel_row["selected_rows"]) > 0:
    st.header("Historical data")
    col1, col2= st.columns(2)
    sel_row_data = sel_row["selected_rows"][0]
    sel_row_testdate = date.fromisoformat(sel_row_data.get("TESTDATE")) # datetime.strptime(sel_row_data.get("TESTDATE"), "%Y-%m-%d").date()
    attribute_data = hist_data[hist_data["ATTRIBUTE"] == sel_row_data.get("Attribute")]
    attribute_athlete_data = attribute_data[attribute_data["NAME"] == sel_row_data.get("Athlete")]
    attribute_athlete_data = attribute_athlete_data.sort_values(by=["TESTDATE"])

    # Metadata of athlete
    athlete_metadata = athlete_data.loc[sel_row_data.get("Athlete")]
    athlete_birthday = date.fromisoformat(athlete_metadata["GEBDAT"]) # datetime.strptime(athlete_metadata["GEBDAT"], "%Y-%m-%d").date()
    
    # Histogram
    with col1:
        col1_1, col1_2= st.columns(2)
        with col1_1:
            gender_options = [athlete_metadata["GENDER"]] + [e for e in attribute_data["GENDER"].unique().tolist() if e != athlete_metadata["GENDER"]] + ["All"]      
            selectbox_gender = st.selectbox(label='Gender', options=gender_options)
        with col1_2:
            sport_options = [athlete_metadata["SPORTART"]] + [e for e in attribute_data["SPORTART"].unique().tolist() if e != athlete_metadata["SPORTART"]] + ["All"]
            selectbox_sport = st.selectbox(label='Sport', options=sport_options)
        if selectbox_gender == "All":
            filtered_attribute_data = attribute_data
        else:
            filtered_attribute_data = attribute_data[attribute_data["GENDER"] == selectbox_gender]
        if selectbox_sport == "All":
            filtered_attribute_data = filtered_attribute_data
        else:
            filtered_attribute_data = filtered_attribute_data[filtered_attribute_data["SPORTART"] == selectbox_sport]                
        fig = px.histogram(filtered_attribute_data, x="NUMERICALVALUE", title="Histogram population")
        fig.add_vline(x=sel_row_data.get("Value"), line_width=10, line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        st.table(athlete_metadata)

    # Past values
    with col2:
        athlete_hist_df = attribute_athlete_data[["TESTDATE", "NUMERICALVALUE"]]
        athlete_hist_df["DATA_TYPE"] = "Historical data"
        athlete_hist_df["BIRTHDAY"] = athlete_birthday
        athlete_hist_df["DAYS_DIFF"] = (athlete_hist_df["TESTDATE"] - athlete_hist_df["BIRTHDAY"])
        athlete_hist_df["DAYS_DIFF"] = (athlete_hist_df["TESTDATE"] - athlete_hist_df["BIRTHDAY"]).dt.days

        # Train model
        x_train = athlete_hist_df["DAYS_DIFF"]
        y_train = athlete_hist_df["NUMERICALVALUE"]
        n_samples = 4
        model = get_linear_model(x_train=x_train, y_train=y_train, n_samples=n_samples)

        # Predict values for first and last datapoint
        x_pred = athlete_hist_df[["TESTDATE", "DAYS_DIFF"]][-n_samples:]
        x_pred = x_pred.append(pd.DataFrame([[sel_row_testdate, (sel_row_testdate - athlete_birthday).days]], columns=["TESTDATE", "DAYS_DIFF"]))
        y_pred, y_pred_80, y_pred_95 = get_pred(model, x_pred["DAYS_DIFF"])
        #print(y_pred_80)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=athlete_hist_df["TESTDATE"], y=athlete_hist_df["NUMERICALVALUE"], mode="markers", name="Historical values"))
        fig.add_trace(go.Scatter(x=[sel_row_testdate], y=[sel_row_data.get("Value")], mode="markers", name="Current test"))
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"], y=y_pred, mode="lines", name="Linear model (last" + str(n_samples) + " samples)"))
        fig.update_traces(marker_size=10)
        fig.update_layout(
            title='Past values of selected athlete',
            xaxis_title='Test date',
            yaxis_title=sel_row_data.get("Attribute"))
        st.plotly_chart(fig, use_container_width=True)

        # # Show image
        # if athlete_metadata["FOTOURL"] != "":
        #     response = requests.get(athlete_metadata["FOTOURL"])
        #     img = Image.open(BytesIO(response.content))
        #     st.image(img, caption=sel_row_data.get("Athlete"))
