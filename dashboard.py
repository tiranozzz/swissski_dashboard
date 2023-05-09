import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from queries import get_hist_data, get_test_files, read_file, get_athlete_data
from models import get_linear_model, get_pred, get_light_color_class
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
        y_pred, y_pred_green_light, y_pred_yellow_light = get_pred(
            model=model, 
            x_pred=x_pred["DAYS_DIFF"],
            pi_green_light=0.8,
            pi_yellow_light=0.95)
        # st.table(y_pred_green_light)
        # st.table(y_pred_yellow_light)
        # st.table(y_pred)
       
        fig = go.Figure()
        # Green light
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=y_pred_green_light["obs_ci_lower"],
                                 fill=None, 
                                 mode="lines",
                                 line=dict(width=0.5, color="green"),
                                 line_color="green",
                                 name=None,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=y_pred_green_light["obs_ci_upper"],
                                 fill="tonexty",
                                 mode="lines",
                                 line=dict(width=0.5, color="green"),
                                 line_color="green",
                                 showlegend=False))
        # Yellow light
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=y_pred_yellow_light["obs_ci_lower"],
                                 fill=None, 
                                 mode="lines",
                                 line=dict(width=0.5, color="yellow"),
                                 line_color="yellow",
                                 name=None,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=y_pred_yellow_light["obs_ci_upper"],
                                 fill="tonexty",
                                 mode="lines",
                                 line=dict(width=0.5, color="yellow"),
                                 line_color="yellow",
                                 showlegend=False))
        # Red light
        # We must find the automatically generated boundaries first
        y_mins = []
        y_maxs = []
        for trace_data in fig.data:
            y_mins.append(min(trace_data.y))
            y_maxs.append(max(trace_data.y))
        y_min = min(y_mins)
        y_max = max(y_maxs)
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=[y_min],
                                 fill=None, 
                                 mode="lines",
                                 line=dict(width=0.5, color="red"),
                                 line_color="red",
                                 name=None,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=y_pred_yellow_light["obs_ci_lower"],
                                 fill="tonexty", 
                                 mode="lines",
                                 line=dict(width=0.5, color="red"),
                                 line_color="red",
                                 name=None,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=y_pred_yellow_light["obs_ci_upper"],
                                 fill=None, 
                                 mode="lines",
                                 line=dict(width=0.5, color="red"),
                                 line_color="red",
                                 name=None,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=[y_max],
                                 fill="tonexty", 
                                 mode="lines",
                                 line=dict(width=0.5, color="red"),
                                 line_color="red",
                                 name=None,
                                 showlegend=False))
        # Historical values
        fig.add_trace(go.Scatter(x=athlete_hist_df["TESTDATE"],
                                 y=athlete_hist_df["NUMERICALVALUE"],
                                 mode="markers",
                                 marker=dict(color="blue"),
                                 name="Historical values"))
        # Current test value
        fig.add_trace(go.Scatter(x=[sel_row_testdate],
                                 y=[sel_row_data.get("Value")],
                                 mode="markers",
                                 marker=dict(color="white", line=dict(color="black", width=2)),
                                 name="Current test"))
        # Model
        fig.add_trace(go.Scatter(x=x_pred["TESTDATE"],
                                 y=y_pred, fill=None,
                                 mode="lines",
                                 line_color="red",
                                 name=f"Linear model (last {str(n_samples)} samples)"))
        fig.update_traces(marker_size=10)
        fig.update_layout(
            title=f"Past values of \"{sel_row_data.get('Athlete')}\"",
            xaxis_title='Test date',
            yaxis_title=sel_row_data.get("Attribute"))
        st.plotly_chart(fig, use_container_width=True)

        # Print boundaries
        green_lower_bound = y_pred_green_light["obs_ci_lower"][y_pred_green_light.index[-1]]
        green_upper_bound = y_pred_green_light["obs_ci_upper"][y_pred_green_light.index[-1]]
        yellow_lower_bound = y_pred_yellow_light["obs_ci_lower"][y_pred_green_light.index[-1]]
        yellow_upper_bound = y_pred_yellow_light["obs_ci_upper"][y_pred_green_light.index[-1]]
        
        # fig = go.Figure(layout_xaxis_range=[yellow_lower_bound * 0.8, yellow_upper_bound * 1.2], layout_yaxis_range=[0,1])
        # fig.add_vline(x=sel_row_data.get("Value"), line_width=2, line_dash="dash", line_color="black")
        # fig.add_trace(go.Scatter(x=[yellow_lower_bound],
        #                          y=[1],
        #                          fill="tonextx", 
        #                          mode="lines",
        #                          line=dict(width=0.5, color="yellow"),
        #                          line_color="yellow",
        #                          name=None,
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=[yellow_upper_bound],
        #                          y=[1],
        #                          fill=None,
        #                          mode="lines",
        #                          line=dict(width=0.5, color="yellow"),
        #                          line_color="yellow",
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=[yellow_lower_bound],
        #                          y=[1],
        #                          fill="tonextx", 
        #                          mode="lines",
        #                          line=dict(width=0.5, color="yellow"),
        #                          line_color="yellow",
        #                          name=None,
        #                          showlegend=False))
        # fig.update_traces(marker_size=10)
        # st.plotly_chart(fig, use_container_width=True)

        st.text(f"Red: [{-np.inf}, {yellow_lower_bound}], [{yellow_upper_bound}, {np.inf}]")
        st.text(f"Yellow: [{yellow_lower_bound}, {green_lower_bound}], [{green_upper_bound}, {yellow_upper_bound}]")
        st.text(f"Green: [{green_lower_bound}, {green_upper_bound}]")

        light_color_class = get_light_color_class(y_test=sel_row_data.get("Value"), 
                                      green_lower_bound=green_lower_bound, 
                                      green_upper_bound=green_upper_bound, 
                                      yellow_lower_bound=yellow_lower_bound, 
                                      yellow_upper_bound=yellow_upper_bound)
        st.text("Color:" + light_color_class)

        # # Show image
        # if athlete_metadata["FOTOURL"] != "":
        #     response = requests.get(athlete_metadata["FOTOURL"])
        #     img = Image.open(BytesIO(response.content))
        #     st.image(img, caption=sel_row_data.get("Athlete"))
