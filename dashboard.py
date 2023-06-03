import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dropbox

from st_aggrid import AgGrid
from st_aggrid.shared import ColumnsAutoSizeMode
from queries import get_hist_data, get_athlete_data
from file_util import read_excel, get_test_files_from_dropbox
from models import get_linear_model, get_pred, get_light_color_class
from util import DatabaseConnection, Config
from datetime import date

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

# Dropbox connection
dbx_input_folder = config["dropbox"]["files_to_display_path"]
dbx = dropbox.Dropbox(app_key=st.secrets["DROPBOX_APP_KEY"],
                      app_secret=st.secrets["DROPBOX_APP_SECRET"],
                      oauth2_refresh_token=st.secrets["DROPBOX_REFRESH_TOKEN"])

# Train the models and write models to Dictionary
def train_models(file_df, filename):
    model_dict = {}
    # Gather data from files first
    for index, row in file_df.iterrows():
        if row["AttributeSupported"]:
            test_dict = {}
            for col, val in zip(file_df.columns, row.values):
                test_dict[col] = val
                if col == "Attribute":
                    test_dict["ModelType"] = config["test_attributes"][val]["model"]
                    test_dict["n_samples"] = config["test_attributes"][val]["n_samples"]
                    test_dict["pi_green_light"] = config["test_attributes"][val]["pi_green_light"]      # Prediction interval green light
                    test_dict["pi_yellow_light"] = config["test_attributes"][val]["pi_yellow_light"]    # Prediction interval yellow light
                elif col == "TestDate":
                    test_dict["TestDateDatetime"] = val.to_pydatetime().date()
            test_dict["filename"] = filename
            # Enrich with data from athlete
            test_dict["AthleteExists"] = row["Athlete"] in athlete_data.index.values.tolist()
            if test_dict["AthleteExists"]:
                athlete_metadata_series = athlete_data.loc[row["Athlete"]]
                athlete_birthday = date.fromisoformat(athlete_metadata_series["GEBDAT"])
                test_dict["AthleteBirthday"] = athlete_birthday
                test_dict["AthleteGender"] = athlete_metadata_series["GENDER"]
                test_dict["AthleteSport"] = athlete_metadata_series["SPORTART"]
            model_dict[test_dict["TestID"]] = test_dict
            
    for key, item in model_dict.items():
        if item["AthleteExists"]:
            # Gather historical data from athlete
            attribute_data_df = hist_data[hist_data["ATTRIBUTE"] == item["Attribute"]]
            attribute_athlete_data_df = attribute_data_df[attribute_data_df["NAME"] == item["Athlete"]]
            attribute_athlete_data_df = attribute_athlete_data_df.sort_values(by=["TESTDATE"])
            athlete_hist_df = attribute_athlete_data_df[["TESTDATE", "NUMERICALVALUE"]]
            # TODO: check FutureWarning
            athlete_hist_df.loc[:, "BIRTHDAY"] = athlete_birthday
            athlete_hist_df["DAYS_DIFF"] = (athlete_hist_df["TESTDATE"] - athlete_hist_df["BIRTHDAY"]).dt.days
            # Train model
            if item["ModelType"] == "LinearModel":
                model, exception_msg = get_linear_model(x_train=athlete_hist_df["DAYS_DIFF"], 
                                        y_train=athlete_hist_df["NUMERICALVALUE"], 
                                        n_samples=item["n_samples"])
                model_dict[key]["exception_msg"] = exception_msg
                if exception_msg != None:
                    model_dict[key]["light_color_class"] = "ERROR"
                else:
                    # Predict values for the last n datapoints and current test
                    x_pred = athlete_hist_df[["TESTDATE", "DAYS_DIFF"]][-item["n_samples"]:]
                    x_pred = pd.concat([x_pred, pd.DataFrame([[item["TestDateDatetime"], (item["TestDateDatetime"] - athlete_birthday).days]], columns=["TESTDATE", "DAYS_DIFF"])])
                    y_pred, y_pred_green_light, y_pred_yellow_light = get_pred(
                        model=model, 
                        x_pred=x_pred["DAYS_DIFF"],
                        pi_green_light=item["pi_green_light"],
                        pi_yellow_light=item["pi_yellow_light"])
                    # Get light color
                    green_lower_bound = y_pred_green_light["obs_ci_lower"][y_pred_green_light.index[-1]]
                    green_upper_bound = y_pred_green_light["obs_ci_upper"][y_pred_green_light.index[-1]]
                    yellow_lower_bound = y_pred_yellow_light["obs_ci_lower"][y_pred_green_light.index[-1]]
                    yellow_upper_bound = y_pred_yellow_light["obs_ci_upper"][y_pred_green_light.index[-1]]
                    light_color_class = get_light_color_class(y_test=item["Value"], 
                                                green_lower_bound=green_lower_bound, 
                                                green_upper_bound=green_upper_bound, 
                                                yellow_lower_bound=yellow_lower_bound, 
                                                yellow_upper_bound=yellow_upper_bound)
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
                    model_dict[key]["light_color_class"] = light_color_class
            else:
                st.error(f"Model type {item['ModelType']} not implemented")
        else:
            model_dict[key]["exception_msg"] = f"Athlete '{item['Athlete']}' not found in the database"
            model_dict[key]["light_color_class"] = "ERROR"
            
    # Display
    # for key, item in model_dict.items():
    #     st.text(f"{key} -> {item}")
    return model_dict

# Get color sort value
def get_color_sort_val(color, exception_msg):
    if exception_msg != None:
        return 99
    if color == "red":
        return 98
    if color == "yellow":
        return 50
    else:
        return 0

st.title("Swissski Dashboard")
col1, col2 = st.columns(2)
with col1:
    # Show data files
    files_dict = {"model_dict": {}}
    sel_row_list = []
    # data_path = current_path + "\\data"
    st.header("File list")
    # files = get_test_files(dir_path=data_path)
    files = get_test_files_from_dropbox(dbx, dbx_input_folder)
    if len(files) == 0:
        st.error("No files found")
    for f in files:
        # files_dict["path"] = data_path
        files_dict["filename"] = f
        file_df, file_val_errors = read_excel(file=files_dict["filename"].path_display, supported_attributes=config["test_attributes"], config=config, dbx=dbx)
        files_dict["df"] = file_df
        if len(file_val_errors) == 0:
        # Train models
            file_model_dict = train_models(file_df=file_df, filename=f)
            files_dict["model_dict"] = files_dict["model_dict"] | file_model_dict       # Merge dictionaries
            for idx, row in file_df.iterrows():
                if row["Attribute"] in config["test_attributes"]:
                    color = files_dict["model_dict"].get(row["TestID"])["light_color_class"]
                    exception_msg = files_dict["model_dict"].get(row["TestID"])["exception_msg"]
                    file_df.at[idx, "LightColor"] = color
                    file_df.at[idx, "LightColorSortVal"] = get_color_sort_val(color=color, exception_msg=exception_msg)
            file_df = file_df.sort_values(by="LightColorSortVal", ascending=False)
            # Get counts
            green_cnt, yellow_cnt, red_cnt, error_cnt = 0, 0, 0, 0
            for val in files_dict["model_dict"].values():
                if val["filename"] == f:
                    if val["exception_msg"] != None:
                        error_cnt += 1
                    elif val["light_color_class"] == "green":
                        green_cnt += 1
                    elif val["light_color_class"] == "yellow":
                        yellow_cnt += 1
                    elif val["light_color_class"] == "red":
                        red_cnt += 1
            # Display files and table
            text_expander = f":page_facing_up: {f.name} ({len(file_df)} rows). Model results: {green_cnt} :large_green_square: | {yellow_cnt} :large_yellow_square: | {red_cnt} :large_red_square:"
            if error_cnt > 0:
                text_expander += f":exclamation: {error_cnt} errors occured"           
            with st.expander(text_expander):
                sel_row = AgGrid(
                    data=file_df, 
                    gridOptions=config["ag_grid"]["grid_options"],
                    width="100%",
                    data_return_mode="AS_INPUT", 
                    update_mode="GRID_CHANGED",
                    allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                    enable_enterprise_modules=False,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                )
                sel_row_list.append(sel_row)
        else: # File Validation errors
            text_expander = f":page_facing_up: {f.name} :exclamation: ERROR reading file"
            with st.expander(text_expander):
                for val_error in file_val_errors:
                    st.error(val_error)

# Get firt selected row (if there are multiple file)
def get_sel_row(sel_row_list):
    for sel_row in sel_row_list:
        if len(sel_row["selected_rows"]) > 0:
            return sel_row

with col2:
    # Catch exception if no file is loaded
    try:
        sel_row = get_sel_row(sel_row_list)
    except NameError:
        sel_row = None

    # Historical data
    if sel_row != None and len(sel_row["selected_rows"]) > 0:
        if sel_row["selected_rows"][0]["Attribute"] not in config["test_attributes"]:
            st.info(f"No model defined for attribute '{sel_row['selected_rows'][0]['Attribute']}'", icon="ℹ️")
        else:
            st.header("Historical data")
            sel_row_data = sel_row["selected_rows"][0]
            attribute_data_df = hist_data[hist_data["ATTRIBUTE"] == sel_row_data.get("Attribute")]
            attribute_athlete_data_df = attribute_data_df[attribute_data_df["NAME"] == sel_row_data.get("Athlete")]
            attribute_athlete_data_df = attribute_athlete_data_df.sort_values(by=["TESTDATE"])
            st.text(f"Selected athlete: '{sel_row_data.get('Athlete')}', selected attribute: '{sel_row_data.get('Attribute')}'")
            # Get model dictionary (trained before)
            sel_model_dict = files_dict["model_dict"][sel_row_data["TestID"]]
            if sel_model_dict["exception_msg"] != None:
                st.error(sel_model_dict["exception_msg"])
            else:
                if pd.isna(sel_model_dict["Value"]):
                    st.info(f"No value was provided for attribute '{sel_row_data.get('Attribute')}'")
                # Past values
                fig = go.Figure()
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
                # Red light
                # We must find the automatically generated boundaries first, therefore this must be plotted last
                y_mins = []
                y_maxs = []
                for trace_data in fig.data:
                    y_mins.append(min(trace_data.y))
                    y_maxs.append(max(trace_data.y))
                y_min = np.tile(A=min(y_mins), reps=len(sel_model_dict["x_pred_TestDates"]))
                y_max = np.tile(A=max(y_maxs), reps=len(sel_model_dict["x_pred_TestDates"]))
                fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                    y=y_min,
                    fill=None, 
                    mode="lines",
                    line=dict(width=0.5, color="red"),
                    line_color="red",
                    name=None,
                    showlegend=False))
                fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                        y=sel_model_dict["y_pred_YellowLowerBound"],
                                        fill="tonexty",
                                        mode="lines",
                                        line=dict(width=0.5, color="red"),
                                        line_color="red",
                                        name=None,
                                        showlegend=False))
                fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                        y=sel_model_dict["y_pred_YellowUpperBound"],
                                        fill=None, 
                                        mode="lines",
                                        line=dict(width=0.5, color="red"),
                                        line_color="red",
                                        name=None,
                                        showlegend=False))
                fig.add_trace(go.Scatter(x=sel_model_dict["x_pred_TestDates"],
                                        y=y_max,
                                        fill="tonexty",
                                        mode="lines",
                                        line=dict(width=0.5, color="red"),
                                        line_color="red",
                                        name=None,
                                        showlegend=False))
                # Plot figure
                fig.update_traces(marker_size=10)
                fig.update_layout(
                    title=f"Past values of \"{sel_row_data.get('Athlete')}\" for attribute \"{sel_row_data.get('Attribute')}\"",
                    xaxis_title='Test date',
                    yaxis_title=sel_row_data.get("Attribute"))
                st.plotly_chart(fig, use_container_width=True)

                # Histogram
                col2_1, col2_2= st.columns(2)
                with col2_1:
                    gender_options = [sel_model_dict["AthleteGender"]] + [e for e in attribute_data_df["GENDER"].unique().tolist() if e != sel_model_dict["AthleteGender"]] + ["All"]      
                    selectbox_gender = st.selectbox(label='Gender', options=gender_options)
                with col2_2:
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
