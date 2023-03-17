def get_hist_data(dbConn, config):
    attribute_filter = "'" + "','".join(config.config["test_attributes"].keys()) + "'"
    dbConn.cursor().execute("USE DATABASE SSKI")
    cur = dbConn.cursor().execute("""
        SELECT tr.*, a.ATHLETEID, a.GENDER, a.GIVENNAME, a.KADER, a.NAME, a.NATION, a.SPORTART, a.SWISSSKINR, a.TGSPEZIALITAET, t.TESTDATE
        FROM DBT_SSKI_DATAAPI.FACTTESTRESULTS tr
        INNER JOIN DBT_SSKI_DATAAPI.DIMATHLETE a on tr.ATHLETEHASHKEY = a.ATHLETEHASHKEY
        INNER JOIN DBT_SSKI_DATAAPI.DIMTEST t on tr.TESTHASHKEY = t.TESTHASHKEY
        WHERE ATTRIBUTE IN ({0})
    """.format(attribute_filter))
    df = cur.fetch_pandas_all()
    return df