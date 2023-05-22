import os
import pandas as pd
import streamlit as st

@st.cache_data
def get_hist_data(_dbConn, config):
    attribute_filter = "'" + "','".join(config["test_attributes"].keys()) + "'"
    _dbConn.cursor().execute("USE DATABASE SSKI")
    cur = _dbConn.cursor().execute("""
        SELECT tr.*, a.ATHLETEID, a.GENDER, a.GIVENNAME, a.KADER, a.NAME, a.NATION, a.SPORTART, a.SWISSSKINR, a.TGSPEZIALITAET, t.TESTDATE as TESTDATE
        FROM DBT_SSKI_DATAAPI.FACTTESTRESULTS tr
        INNER JOIN DBT_SSKI_DATAAPI.DIMATHLETE a on tr.ATHLETEHASHKEY = a.ATHLETEHASHKEY
        INNER JOIN DBT_SSKI_DATAAPI.DIMTEST t on tr.TESTHASHKEY = t.TESTHASHKEY
        WHERE ATTRIBUTE IN ({0})
    """.format(attribute_filter))
    df = cur.fetch_pandas_all()
    # In the database the Umlaute are saved as Umlaute whereas in the test files they are usually rewritten without (e.g. ö -> oe). Align this
    for idx, row in df.iterrows():
        df.at[idx, "NAME"] = translate_umlaute(row["NAME"])
    return df

@st.cache_data
def get_athlete_data(_dbConn, config):
    _dbConn.cursor().execute("USE DATABASE SSKI")
    # Correct birthdays like 0094-03-02 to 1994-03-02
    cur = _dbConn.cursor().execute("""
        SELECT ATHLETEHASHKEY, ATHLETEID, FAMILYNAME, FISCODE, FOTOURL, GENDER, GIVENNAME, ISSWISSSKIATHLETE, KADER, NAME, SPORTART, STATUS, SWISSSKINR, TGSPEZIALITAET, 
        CASE WHEN SUBSTR(GEBDAT, 1, 2) = '00' THEN '19' || SUBSTR(GEBDAT, 3, LENGTH(GEBDAT)) ELSE TO_CHAR(GEBDAT, 'YYYY-MM-DD') END AS GEBDAT
        from DBT_SSKI_DATAAPI.DIMATHLETE
        WHERE ISSWISSSKIATHLETE = 1""")
    df = cur.fetch_pandas_all()
    # In the database the Umlaute are saved as Umlaute whereas in the test files they are usually rewritten without (e.g. ö -> oe). Align this
    for idx, row in df.iterrows():
        df.at[idx, "NAME"] = translate_umlaute(row["NAME"])
    df = df.set_index("NAME")
    return df

def translate_umlaute(input_string):
    special_char_map = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss', ord('Ä'):'Ae', ord('Ü'):'Ue', ord('Ö'):'Oe'}
    return input_string.translate(special_char_map)

def get_test_files(dir_path):
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)) and str(path).endswith(".xlsx") and not str(path).startswith("~"):
            res.append(path)
    return res

def read_csv(file):
    df = pd.read_csv(file, sep=";")
    df["LightColor"] = None
    return df

def read_excel(file, supported_attributes):
    # TODO: move to config
    drop_cols = ["Code (Reihe)", "TestNr Jahr", "Name Vorname", "Name", "Vorname", "Geb", "Alter", "sex", "TG", "Kader", "Testdatum", "Ort", "Alti", "Leiter", "Geraet", "Protokoll", "Start_load", "Stufendauer", "Inkrement", "Training", "Gesundheit", "Temp", "Luftdruck", "Notizen"]
    with open(file, "rb") as f:
        df = pd.read_excel(io=f)
    # Extract each column to a single row in the dataframe
    df_tests = pd.DataFrame()
    for row_idx, row_val in df.iterrows():
        for col in df.columns:
            if not col in drop_cols:
                test_dict = {}
                test_dict["Athlete"] = row_val["Name Vorname"]
                test_dict["TestDate"] = row_val["Testdatum"]
                test_dict["Birthday"] = row_val["Geb"]
                test_dict["Attribute"] = col
                test_dict["AttributeSupported"] = col in supported_attributes
                test_dict["Value"] = row_val[col]
                test_dict["TestID"] = file + "|" + row_val["Name Vorname"] + "|" + str(row_val["Testdatum"]) + "|" + col
                # TODO: move to pd.concat       
                df_tests = df_tests.append(test_dict, ignore_index=True)
    return df_tests