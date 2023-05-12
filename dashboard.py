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

# Create models
model_dict = {}
# Gather data from files first
for index, row in file_df.iterrows():
    test_dict = {}
    for col, val in zip(file_df.columns, row.values):
        test_dict[col] = val
        if col == "Attribute":
            test_dict["ModelType"] = config["test_attributes"][val]["model"]
            test_dict["n_samples"] = config["test_attributes"][val]["n_samples"]
            test_dict["pi_green_light"] = config["test_attributes"][val]["pi_green_light"]      # Prediction interval green light
            test_dict["pi_yellow_light"] = config["test_attributes"][val]["pi_yellow_light"]    # Prediction interval yellow light
        elif col == "TestDate":
            test_dict["TestDateDatetime"] = datetime.strptime(val, "%Y-%m-%d").date()
    # Enrich with data from athlete
    athlete_metadata_series = athlete_data.loc[row["Athlete"]]
    athlete_birthday = date.fromisoformat(athlete_metadata_series["GEBDAT"])
    test_dict["AthleteBirthday"] = athlete_birthday
    test_dict["AthleteGender"] = athlete_metadata_series["GENDER"]
    test_dict["AthleteSport"] = athlete_metadata_series["SPORTART"]
    model_dict[index] = test_dict
for key, item in model_dict.items():
    # Gather historical data from athlete
    attribute_data_df = hist_data[hist_data["ATTRIBUTE"] == item["Attribute"]]
    attribute_athlete_data_df = attribute_data_df[attribute_data_df["NAME"] == item["Athlete"]]
    attribute_athlete_data_df = attribute_athlete_data_df.sort_values(by=["TESTDATE"])
    athlete_hist_df = attribute_athlete_data_df[["TESTDATE", "NUMERICALVALUE"]]
    athlete_hist_df["BIRTHDAY"] = athlete_birthday
    athlete_hist_df["DAYS_DIFF"] = (athlete_hist_df["TESTDATE"] - athlete_hist_df["BIRTHDAY"])
    athlete_hist_df["DAYS_DIFF"] = (athlete_hist_df["TESTDATE"] - athlete_hist_df["BIRTHDAY"]).dt.days
    # Train model
    if item["ModelType"] == "LinearModel":
        model = get_linear_model(x_train=athlete_hist_df["DAYS_DIFF"], 
                                 y_train=athlete_hist_df["NUMERICALVALUE"], 
                                 n_samples=item["n_samples"])
        # Predict values for the last n datapoints and current test
        x_pred = athlete_hist_df[["TESTDATE", "DAYS_DIFF"]][-item["n_samples"]:]
        x_pred = x_pred.append(pd.DataFrame([[item["TestDateDatetime"], (item["TestDateDatetime"] - athlete_birthday).days]], columns=["TESTDATE", "DAYS_DIFF"]))
        y_pred, y_pred_green_light, y_pred_yellow_light = get_pred(
            model=model, 
            x_pred=x_pred["DAYS_DIFF"],
            pi_green_light=item["pi_green_light"],
            pi_yellow_light=item["pi_yellow_light"])
        model_dict[key]["model"] = model
        model_dict[key]["x_pred_TestDates"] = x_pred["TESTDATE"].values.tolist()
        model_dict[key]["x_pred_DaysDiff"] = x_pred["DAYS_DIFF"].values.tolist()
        model_dict[key]["y_pred_GreenLowerBound"] = y_pred_green_light["obs_ci_lower"].values.tolist()
        model_dict[key]["y_pred_GreenUpperBound"] = y_pred_green_light["obs_ci_upper"].values.tolist()
        model_dict[key]["y_pred_YellowLowerBound"] = y_pred_yellow_light["obs_ci_lower"].values.tolist()
        model_dict[key]["y_pred_YellowUpperBound"] = y_pred_yellow_light["obs_ci_upper"].values.tolist()
        model_dict[key]["y_pred"] = y_pred.values.tolist()
        model_dict[key]["athlete_hist_TestDate"] = athlete_hist_df["TESTDATE"].values.tolist()
        model_dict[key]["athlete_hist_Value"] = athlete_hist_df["NUMERICALVALUE"].values.tolist()
    else:
        st.error(f"Model type {item['ModelType']} not implemented")
# # Display
# for key, item in model_dict.items():
#     st.text(f"{key} -> {item}")

# Historical data
if len(sel_row["selected_rows"]) > 0:
    st.header("Historical data")
    col1, col2= st.columns(2)
    sel_row_data = sel_row["selected_rows"][0]
    attribute_data_df = hist_data[hist_data["ATTRIBUTE"] == sel_row_data.get("Attribute")]
    attribute_athlete_data_df = attribute_data_df[attribute_data_df["NAME"] == sel_row_data.get("Athlete")]
    attribute_athlete_data_df = attribute_athlete_data_df.sort_values(by=["TESTDATE"])
    # Get model dictionary (trained before)
    sel_model_dict = model_dict[sel_row_data["_selectedRowNodeInfo"]["nodeRowIndex"]]
    st.text(sel_model_dict)

    
    # Histogram
    with col1:
        col1_1, col1_2= st.columns(2)
        with col1_1:
            gender_options = [sel_model_dict["AthleteGender"]] + [e for e in attribute_data_df["GENDER"].unique().tolist() if e != sel_model_dict["AthleteGender"]] + ["All"]      
            selectbox_gender = st.selectbox(label='Gender', options=gender_options)
        with col1_2:
            sport_options = [sel_model_dict["AthleteSport"]] + [e for e in attribute_data_df["SPORTART"].unique().tolist() if e != sel_model_dict["AthleteSport"]] + ["All"]
            selectbox_sport = st.selectbox(label='Sport', options=sport_options)
        if selectbox_gender == "All":
            filtered_attribute_data_df = attribute_data_df
        else:
            filtered_attribute_data_df = attribute_data_df[attribute_data_df["GENDER"] == selectbox_gender]
        if selectbox_sport == "All":
            filtered_attribute_data_df = filtered_attribute_data_df
        else:
            filtered_attribute_data_df = filtered_attribute_data_df[filtered_attribute_data_df["SPORTART"] == selectbox_sport]                
        fig = px.histogram(filtered_attribute_data_df, x="NUMERICALVALUE", title="Histogram population")
        fig.add_vline(x=sel_row_data.get("Value"), line_width=3, line_color="red", line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    # Past values
    with col2:
        st.text(sel_model_dict)  
        fig = go.Figure()
        # Historical values
        fig.add_trace(go.Scatter(x=sel_model_dict["athlete_hist_TestDate"],
                                 y=sel_model_dict["athlete_hist_Value"],
                                 mode="markers",
                                 marker=dict(color="blue"),
                                 name="Historical values"))
        # Current test value
        fig.add_trace(go.Scatter(x=[sel_model_dict["TestDate"]],
                                 y=[sel_model_dict["Value"]],
                                 mode="markers",
                                 marker=dict(color="white", line=dict(color="black", width=2)),
                                 name="Current test"))
        # Model
        fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                 y=sel_model_dict["y_pred"], fill=None,
                                 mode="lines",
                                 line_color="red",
                                 name=f"Linear model (last {str(sel_model_dict['n_samples'])} samples)"))
        # Green light
        fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                 y=sel_model_dict["y_pred_GreenLowerBound"],
                                 fill=None, 
                                 mode="lines",
                                 line=dict(width=0.5, color="green"),
                                 line_color="green",
                                 name=None,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                 y=sel_model_dict["y_pred_GreenUpperBound"],
                                 fill="tonexty",
                                 mode="lines",
                                 line=dict(width=0.5, color="green"),
                                 line_color="green",
                                 showlegend=False))
        # Yellow light
        fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                 y=sel_model_dict["y_pred_YellowLowerBound"],
                                 fill=None, 
                                 mode="lines",
                                 line=dict(width=0.5, color="yellow"),
                                 line_color="yellow",
                                 name=None,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                 y=sel_model_dict["y_pred_YellowUpperBound"],
                                 fill="tonexty",
                                 mode="lines",
                                 line=dict(width=0.5, color="yellow"),
                                 line_color="yellow",
                                 showlegend=False))
        # # Red light
        # # We must find the automatically generated boundaries first
        # y_mins = []
        # y_maxs = []
        # for trace_data in fig.data:
        #     y_mins.append(min(trace_data.y))
        #     y_maxs.append(max(trace_data.y))
        # y_min = np.tile(A=min(y_mins), reps=len(sel_model_dict["x_pred_TestDates"]))
        # y_max = np.tile(A=min(y_maxs), reps=len(sel_model_dict["x_pred_TestDates"]))
        # st.text(y_min)
        # st.text(y_max)
        # fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
        #                          y=y_min,
        #                          fill=None, 
        #                          mode="lines",
        #                          line=dict(width=0.5, color="red"),
        #                          line_color="red",
        #                          name=None,
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
        #                          y=sel_model_dict["y_pred_YellowLowerBound"],
        #                          fill="tonexty", 
        #                          mode="lines",
        #                          line=dict(width=0.5, color="red"),
        #                          line_color="red",
        #                          name=None,
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
        #                          y=sel_model_dict["y_pred_YellowUpperBound"],
        #                          fill=None, 
        #                          mode="lines",
        #                          line=dict(width=0.5, color="red"),
        #                          line_color="red",
        #                          name=None,
        #                          showlegend=False))
        # fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
        #                          y=y_max,
        #                          fill="tonexty", 
        #                          mode="lines",
        #                          line=dict(width=0.5, color="red"),
        #                          line_color="red",
        #                          name=None,
        #                          showlegend=False))
        # Plot figure
        fig.update_traces(marker_size=10)
        fig.update_layout(
            title=f"Past values of \"{sel_row_data.get('Athlete')}\"",
            xaxis_title='Test date',
            yaxis_title=sel_row_data.get("Attribute"))
        st.plotly_chart(fig, use_container_width=True)

    #     # Print boundaries
    #     green_lower_bound = y_pred_green_light["obs_ci_lower"][y_pred_green_light.index[-1]]
    #     green_upper_bound = y_pred_green_light["obs_ci_upper"][y_pred_green_light.index[-1]]
    #     yellow_lower_bound = y_pred_yellow_light["obs_ci_lower"][y_pred_green_light.index[-1]]
    #     yellow_upper_bound = y_pred_yellow_light["obs_ci_upper"][y_pred_green_light.index[-1]]
        
    #     # fig = go.Figure(layout_xaxis_range=[yellow_lower_bound * 0.8, yellow_upper_bound * 1.2], layout_yaxis_range=[0,1])
    #     # fig.add_vline(x=sel_row_data.get("Value"), line_width=2, line_dash="dash", line_color="black")
    #     # fig.add_trace(go.Scatter(x=[yellow_lower_bound],
    #     #                          y=[1],
    #     #                          fill="tonextx", 
    #     #                          mode="lines",
    #     #                          line=dict(width=0.5, color="yellow"),
    #     #                          line_color="yellow",
    #     #                          name=None,
    #     #                          showlegend=False))
    #     # fig.add_trace(go.Scatter(x=[yellow_upper_bound],
    #     #                          y=[1],
    #     #                          fill=None,
    #     #                          mode="lines",
    #     #                          line=dict(width=0.5, color="yellow"),
    #     #                          line_color="yellow",
    #     #                          showlegend=False))
    #     # fig.add_trace(go.Scatter(x=[yellow_lower_bound],
    #     #                          y=[1],
    #     #                          fill="tonextx", 
    #     #                          mode="lines",
    #     #                          line=dict(width=0.5, color="yellow"),
    #     #                          line_color="yellow",
    #     #                          name=None,
    #     #                          showlegend=False))
    #     # fig.update_traces(marker_size=10)
    #     # st.plotly_chart(fig, use_container_width=True)

    #     st.text(f"Red: [{-np.inf}, {yellow_lower_bound}], [{yellow_upper_bound}, {np.inf}]")
    #     st.text(f"Yellow: [{yellow_lower_bound}, {green_lower_bound}], [{green_upper_bound}, {yellow_upper_bound}]")
    #     st.text(f"Green: [{green_lower_bound}, {green_upper_bound}]")

    #     light_color_class = get_light_color_class(y_test=sel_row_data.get("Value"), 
    #                                   green_lower_bound=green_lower_bound, 
    #                                   green_upper_bound=green_upper_bound, 
    #                                   yellow_lower_bound=yellow_lower_bound, 
    #                                   yellow_upper_bound=yellow_upper_bound)
    #     st.text("Color:" + light_color_class)

        # # Show image
        # if athlete_metadata["FOTOURL"] != "":
        #     response = requests.get(athlete_metadata["FOTOURL"])
        #     img = Image.open(BytesIO(response.content))
        #     st.image(img, caption=sel_row_data.get("Athlete"))
