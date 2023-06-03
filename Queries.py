import streamlit as st
from file_util import translate_umlaute

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