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
    df = df.set_index("NAME")
    return df

@st.cache_data
def get_test_files(dir_path):
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    return res

def read_file(file):
    df = pd.read_csv(file, sep=";")
    df["Check"] = None
    return df