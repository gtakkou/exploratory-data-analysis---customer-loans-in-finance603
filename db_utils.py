import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import yaml
from sqlalchemy import inspect
from sqlalchemy import text
import matplotlib.pyplot as plt
import numpy as np
import scipy
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns

DOWNLOAD_FROM_DB = False
CSV_LOCATION = "C:/Users/George Takkou/Desktop/VS Code/loan_payments_from_child.csv"
CSV_DATE_COLUMNS = [
   "issue_date",
   "earliest_credit_line",
   "last_payment_date",
   "next_payment_date",
   "last_credit_pull_date"
]

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
      return sum(self.df["loan_amount"])
   


class DataTransform:
    def __init__(self, df):
        self.df = df

    def convert_columns_to_correct_format(self):
        self.df.id = self.df.id.astype("category")
        self.df.member_id = self.df.member_id.astype("category")
        self.df.home_ownership = self.df.home_ownership.astype("string")
        self.df.grade = self.df.grade.astype("string")
        self.df.verification_status = self.df.verification_status.astype("string")
        self.df.payment_plan = self.df.payment_plan.astype("string")
        self.df.purpose = self.df.purpose.astype("string")
        self.df.application_type = self.df.application_type.astype("string")
        
class DataFrameInfo:
   def __init__(self, df):
        self.df = df
   
   # Describe all columns in the DataFrame to check their data types
   def check_data_types(self):
      print(self.df.info())
   
   # Extract statistical values: median, standard deviation and mean from the columns and the DataFrame
         # print(self.df.describe())
   def describe_all_columns(self):
      print(self.df.describe())

   
   # Count distinct values in categorical columns
   def num_unique_ids(self):
      print("\nUnique product IDs:")
      print(self.df["member_id"].nunique())

  
   # Print out the shape of the DataFrame
   def shape(self):
      print("Shape of the DataFrame:", self.df.shape)
   
   # Generate a count/percentage count of NULL values in each column
     
class Null:
   def __init__(self, df):
        self.df = df
        
   def null_values_sum(self):
      null_count = df.isnull().sum()
      total_count = len(df)
      null_percentage = (null_count / total_count) * 100
      
      null_info = pd.DataFrame({
         'Null Count': null_count,
         'Null Percentage': null_percentage
      })

      print("Null values in each column:")
      print(null_info)
      
class Plotter:
   def __init__(self, df):
        self.df = df
    
   def plot_null_values(df, title="NULL Values Before and After Removal"):
      null_before = df.isnull().sum()
        
      df = df.dropna(subset=['next_payment_date', 'mths_since_last_major_derog'])
        
      null_after = df.isnull().sum()

      fig, ax = plt.subplots(figsize=(10, 6))

      null_before.plot(kind='bar', ax=ax, color='blue', position=0, width=0.4, label='Before Removal')
      null_after.plot(kind='bar', ax=ax, color='green', position=1, width=0.4, label='After Removal')

      ax.set_xlabel('Columns')
      ax.set_ylabel('Number of NULL Values')
      ax.set_title(title)
      ax.legend()

      plt.show()
      
   def plot_skewness(self):
      #df = self.df(as_frame=True).frame
      df['annual_inc'].hist(bins=50)
      print(f"Skew of annual income column is {df['annual_inc'].skew()}")
      #print("this indicates a strong positive skew")
      
   
   def plot_boxplot(data, columns, title="Boxplot"):
      data[columns].boxplot()
      plt.title(title)
      plt.show()
      df_transform.visualize_outliers(['annual_inc', 'delinq_2yrs'], title="Outliers Visualization")


   
  
   # Any other methods you may find useful
   def information_from_data(self, df):
      self.df.info
      
class DataFrameTransform:
   def __init__(self, df):
      self.df = df

   def drop_null_values(self):
      self.df = self.df.dropna(subset=['next_payment_date', 'mths_since_last_major_derog'])  # drops rows with missing values
      return self.df

   def impute(self):
      self.df['funded_ammount'].fillna(self.df['funded_ammount'].median(), inplace=True)  # Median
      return self.df
   
   def identify_skewed_columns(self, skew_threshold_range=(-2, 2)):
      numeric_columns = self.df.select_dtypes(include=np.number).columns
      skewed_columns = []

      for col in numeric_columns:
         skewness = self.df[col].skew()
         if abs(skewness) > skew_threshold_range[1]:
               skewed_columns.append(col)

      return skewed_columns
   
   def correct_skewness(self):
      log_annual_inc = df["annual_inc"].map(lambda i: np.log(i) if i > 0 else 0)
      t=sns.histplot(log_annual_inc,label="Skewness: %.2f"%(log_annual_inc.skew()) )
      t.legend()
      
   
   
    
     
#class DataFrameTransform:
   ##def __init__(self, df):
        #self.df = df
           
   #def drop_null_values(self):
      #self.df.dropna(subset=['next_payment_date','mths_since_last_major_derog','mths_since_last_record',]) # drops rows with missing values
      #return self.df 
      
   #def impute(self):
      #self.df['funded_ammount'].fillna(df['funded_ammount'].median())  # Median
      #return self.df
      
  

if __name__ == "__main__":

   print("Script Started")
   if DOWNLOAD_FROM_DB:
      # Create connector and read table onto a dataframe
      db = RDSDatabaseConnector("AWS/credentials.yaml")
      df = db.read_sql_table("loan_payments")

      # Perform operations on that dataframe
      df_ops = DataframeOperations(df)
      df_ops.save_to_csv(CSV_LOCATION)
      print(f"Total Loan Amount: {df_ops.get_total_load_amount()}")
   else:
      df = pd.read_csv(CSV_LOCATION)
      
   # Operations after this point
   print(df['issue_date'].iloc[0])
   df[CSV_DATE_COLUMNS] = df[CSV_DATE_COLUMNS].apply(pd.to_datetime) # Convert all date columns to datetime type
   # print(df["last_credit_pull_date"])
   
   # Convert columns
   data_transformation = DataTransform(df)
   data_transformation.convert_columns_to_correct_format()
   
   df_info = DataFrameInfo(df)
   df_info.describe_all_columns()
   
   df_unique = DataFrameInfo(df)
   df_unique.num_unique_ids()
   
   df_shape = DataFrameInfo(df)
   df_shape.shape()
   
   df_null = Null(df)
   df_null.null_values_sum()
       
   df_transform = DataFrameTransform(df)
   Plotter.plot_null_values(df_transform.df, title="NULL Values Before and After Removal")
   skewed_columns = df_transform.identify_skewed_columns(skew_threshold_range=(-2, 2))
   print("Skewed Columns:", skewed_columns)
   df_transform.correct_skewness()
   
   
   
   df_plot_skewness = Plotter(df)
   df_plot_skewness.plot_skewness()
   
   df_outliers = Plotter(df)
   df_outliers.plot_boxplot
   
   
   
   
   
   
   
     
   # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
   #    print(df.iloc[:5])
   
   print("Script Ended")
   

current_state_df = df[['loan_status', 'recoveries', 'funded_amount_inv', 'total_rec_prncp']]
current_state_df['recovery_percentage'] = (current_state_df['recoveries'] / current_state_df['funded_amount_inv']) * 100
# Calculate the percentage of loans recovered against the total amount funded
current_state_df['total_recovery_percentage'] = (current_state_df['recoveries'] / (current_state_df['funded_amount_inv'] + current_state_df['total_rec_prncp'])) * 100

# Visualize recovery percentages
plt.figure(figsize=(10, 6))
plt.bar(['Investor Funding', 'Total Funding'], [current_state_df['recovery_percentage'].mean(), current_state_df['total_recovery_percentage'].mean()])
plt.title('Average Percentage of Loans Recovered')
plt.ylabel('Percentage')
plt.show()

future_recovery_df = current_state_df[current_state_df['next_payment_date'] <= pd.Timestamp.now() + pd.DateOffset(months=6)]

# Calculate the percentage of loans recovered up to 6 months in the future
future_recovery_percentage = (future_recovery_df['recoveries'] / (future_recovery_df['funded_amount_inv'] + future_recovery_df['total_rec_prncp'])) * 100

# Visualize future recovery percentages
plt.figure(figsize=(10, 6))
plt.bar(['Up to 6 Months'], [future_recovery_percentage.mean()])
plt.title('Average Percentage of Loans Recovered Up to 6 Months in the Future')
plt.ylabel('Percentage')
plt.show()

charged_off_df = df[df['loan_status'] == 'Charged Off']

# percentage of charged off loans historically
charged_off_percentage = (charged_off_df.shape[0] / df.shape[0]) * 100

# total amount paid towards charged off loans
total_paid_before_charge_off = charged_off_df['total_payment'].sum()

# Display results
print(f"Percentage of Charged Off Loans: {charged_off_percentage:.2f}%")
print(f"Total Amount Paid towards Charged Off Loans: ${total_paid_before_charge_off:.2f}")

remaining_term = charged_off_df['term'] - charged_off_df['installment']

# projected loss for each loan
charged_off_df['projected_loss'] = charged_off_df['installment'] * remaining_term

Calculate the total projected loss
total_projected_loss = charged_off_df['projected_loss'].sum()

# Display the total projected loss
print(f"Total Projected Loss for Charged Off Loans: ${total_projected_loss:.2f}")

# Visualize the loss projected over the remaining term
plt.figure(figsize=(10, 6))
plt.bar(charged_off_df['id'], charged_off_df['projected_loss'])
plt.xlabel('Loan ID')
plt.ylabel('Projected Loss ($)')
plt.title('Projected Loss of Charged Off Loans Over Remaining Term')
plt.show()

late_payments_df = df[df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)'])]
percentage_late_payments = (late_payments_df.shape[0] / df.shape[0]) * 100
total_late_customers = late_payments_df.shape[0]
loss_due_to_late_payments = late_payments_df['total_payment'].sum()
remaining_term_late_payments = late_payments_df['term'] - late_payments_df['installment']
late_payments_df['projected_loss'] = late_payments_df['installment'] * remaining_term_late_payments
total_projected_loss_late_payments = late_payments_df['projected_loss'].sum()

# Calculate the percentage of total expected revenue represented by late payments and already defaulted loans
total_expected_revenue = df['total_payment'].sum()
percentage_of_total_expected_revenue = ((total_projected_loss_late_payments + total_projected_loss) / total_expected_revenue) * 100

# Display results
print(f"Percentage of Users with Late Payments: {percentage_late_payments:.2f}%")
print(f"Total Late Customers: {total_late_customers}")
print(f"Loss Due to Late Payments: ${loss_due_to_late_payments:.2f}")
print(f"Projected Loss of Late Payments: ${total_projected_loss_late_payments:.2f}")
print(f"Percentage of Total Expected Revenue Represented by Late Payments and Defaulted Loans: {percentage_of_total_expected_revenue:.2f}%")

non_payment_subset = df[df['loan_status'].isin(['Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'])]

# Define columns of interest
columns_of_interest = ['grade', 'purpose', 'home_ownership', 'employment_length', 'dti', 'delinq_2yrs', 'inq_last_6mths']

# Visualize the impact of each column on the likelihood of not paying
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_of_interest, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, hue='loan_status', data=non_payment_subset, palette='viridis')
    plt.title(f'Impact of {column} on Loan Status')
    plt.xlabel(column)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()