import psycopg2
import PyYAML
from sqlalchemy import create_engine

host = "project-aicore.c5ofryhnh2op.eu-west-1.rds.amazonaws.com"
port = "5432"
database = "postgres"
user = "postgres"
password = "omonoia1996"


class RDSDatabaseConnector:
    
    def connect(self):
        self.connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        print("connected")


    def close_connection(self):
        self.connection.close()
        print("disconnected")

    def load_credentials(file_path = "credentialt.yaml"):
       with open(file_path,'r') as file:
        return load_credentials
       
    def __init__(self, credentials):
        self.host = credentials['host']
        self.port = credentials['user']
        self.database = credentials['database']
        self.user = credentials['user']
        self.password = credentials['password']

        self.connection = None

    def _create_engine(self):
        # Create and return an SQLAlchemy engine
        engine = create_engine(
            f"mysql+mysqlconnector://{self.host}:{self.password}@{self.host}/{self.database}",
            pool_size=5, pool_timeout=30, pool_recycle=3600
        )
        return engine


    # Other method to get data

if __name__ == "__main__":

    db = RDSDatabaseConnector(
        host, port, database, user, password
    )
    db.connect()




    db.close_connection()

def load_credentials(file_path = "credentialt.yaml"):
    open(file_path,'r') as file:
return credentials