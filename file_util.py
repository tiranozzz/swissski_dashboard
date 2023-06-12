import os
import io
import pandas as pd
import dropbox
import streamlit as st

def translate_umlaute(input_string):
    special_char_map = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss', ord('Ä'):'Ae', ord('Ü'):'Ue', ord('Ö'):'Oe'}
    return input_string.translate(special_char_map)

def get_test_files_from_dropbox(dbx, input_folder):
    res = []
    try:
        for entry in dbx.files_list_folder(input_folder).entries:
            res.append(entry)
    except Exception as err:
        st.error(err)
    return res

def get_test_files(dir_path):
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)) and str(path).endswith(".xlsx") and not str(path).startswith("~"):
            res.append(path)
    return res

def read_excel(file, supported_attributes, config, dbx):
    mandatory_cols = config["test_files"]["mandatory_columns"]
    drop_cols = config["test_files"]["ignore_columns"]
    show_only_supported_attributes = config["show_only_supported_attributes"]
    df_tests = pd.DataFrame()
    # Read local file
    # with open(file, "rb") as f:
    #     df = pd.read_excel(io=f)
    # Read from dropbox
    _, res = dbx.files_download(file)
    with io.BytesIO(res.content) as stream:
        df = pd.read_excel(io=stream.read())
    # Validate file: check if all mandatory columns are there
    file_val_errors = []
    for mand_col in mandatory_cols:
        if mand_col not in df.columns:
            file_val_errors.append(f"Mandatory column '{mand_col}' does not exist in file")
        else:
            if df[mand_col].isna().sum() > 0:
                file_val_errors.append(f"Mandatory column '{mand_col}' has {df[mand_col].isna().sum()} empty values")
    if len(file_val_errors) == 0:
    # Extract each column to a single row in the dataframe
        for row_idx, row_val in df.iterrows():
            for col in df.columns:
                if not col in drop_cols:
                    test_dict = {}
                    test_dict["AttributeSupported"] = col in supported_attributes
                    if show_only_supported_attributes == False or (show_only_supported_attributes == True and test_dict["AttributeSupported"] == True):
                        test_dict["Athlete"] = translate_umlaute(row_val["Name Vorname"])
                        test_dict["TestDate"] = row_val["Testdatum"]
                        test_dict["Birthday"] = row_val["Geb"]
                        test_dict["Attribute"] = col
                        # Convert "," to "." for supported attributes in value (depends on which language the Excel is saved as)
                        if col in config["test_attributes"] and isinstance(row_val[col], str):
                            test_dict["Value"] = float(row_val[col].replace(",", "."))
                        else:
                            test_dict["Value"] = row_val[col]
                        test_dict["TestID"] = file + "|" + row_val["Name Vorname"] + "|" + str(row_val["Testdatum"]) + "|" + col
                        df_tests = pd.concat([df_tests, pd.DataFrame([test_dict])], ignore_index=True)
    return df_tests, file_val_errors