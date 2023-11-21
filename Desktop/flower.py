from sqlalchemy import create_engine

# Database connection details
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = "iris.c5ofryhnh2op.eu-west-1.rds.amazonaws.com"
USER = 'postgres'
PASSWORD = "omonoia1996"
PORT = 5432
DATABASE = 'postgres'
engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")

engine.connect()