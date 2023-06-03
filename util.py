import snowflake.connector
import yaml
import os
import streamlit as st

class DatabaseConnection:

    def __init__(self, config):
        self.database_config = config["database"]

    @st.cache_resource
    def get_connection(_self):
        conn = snowflake.connector.connect(
            user=st.secrets["DB_USER"],
            account=st.secrets["DB_ACCOUNT"],
            password=st.secrets["DB_PASSWORD"],
            host=st.secrets["DB_HOST"],
            database=_self.database_config["database"],
            session_parameters=_self.database_config["session_parameters"]
        )
        return conn
      
class Config:

    def __init__(self):
        self.path = os.path.realpath(os.path.dirname(__file__))
        self.config = self.read_config()

    @st.cache_resource
    def read_config(_self):
        with open(_self.path + "/config.yaml", "r", encoding="utf-8") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
