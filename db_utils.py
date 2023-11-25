import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import yaml
from sqlalchemy import inspect
from sqlalchemy import text


class RDSDatabaseConnector:

    def __init__(self, credentials_location):

        credentials = self.load_credentials(credentials_location)
        
        self.db_type = credentials['RDS_DATABASE_TYPE']
        self.db_api = credentials['RDS_DBAPI']
        self.host = credentials['RDS_HOST']
        self.port = credentials['RDS_PORT']
        self.database = credentials['RDS_DATABASE']
        self.user = credentials['RDS_USER']
        self.password = credentials['RDS_PASSWORD']

        self._create_engine()
    
   
    def load_credentials(self, file_path = "credentials.yaml"):
       with open(file_path,'r') as file:
        return yaml.safe_load(file.read())      


    def _create_engine(self):
       self.engine = create_engine(f"{self.db_type}+{self.db_api}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")

    def execute(self, query):
        # How to use: result = db.execute("SELECT * FROM loan_payments")
        with self.engine.connect() as connection:
           result = connection.execute(text(query))
           for row in result:
              print(row)
    
    def read_sql_table(self, table):
       return pd.read_sql_table(table, self.engine)
            
  

class DataframeOperations:

    def __init__(self, df):
       self.df = df

    def save_to_csv(self, save_path):
       self.df.to_csv(save_path, index=False)

    def get_total_load_amount(self):
       print(sum(self.df["loan_amount"]))
   

if __name__ == "__main__":

    # Create connector and read table onto a dataframe
    db = RDSDatabaseConnector("AWS/credentials.yaml")
    df = db.read_sql_table("loan_payments")

    # Perform operations on that dataframe
    df_ops = DataframeOperations(df)
    df_ops.save_to_csv("C:/Users/George Takkou/Desktop/VS Code/loan_payments_from_child.csv")
