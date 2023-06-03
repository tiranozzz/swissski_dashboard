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
            user=_self.database_config["user"],
            account=_self.database_config["account"],
            password=st.secrets["DB_PASSWORD"],
            host=_self.database_config["host"],
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
