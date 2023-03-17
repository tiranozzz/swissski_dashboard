import snowflake.connector
import yaml
import os
import pandas as pd

class DatabaseConnection:

    def __init__(self, config):
        self.database_config = config["database"]

    def get_connection(self):
        conn = snowflake.connector.connect(
            user=self.database_config["user"],
            account=self.database_config["account"],
            password=self.database_config["password"],
            host=self.database_config["host"],
            database=self.database_config["database"],
            session_parameters=self.database_config["session_parameters"]
        )
        return conn
      
class Config:

    def __init__(self):
        self.path = os.path.realpath(os.path.dirname(__file__))
        self.config = self.read_config()

    def read_config(self):
        with open(self.path + "\\config.yaml", "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
