import pandas as pd

# Replace 'A01.csv' with the actual filename of one subject's data
csv_file = 'BCICIV_2a_1.csv'

# Read only the first row to get column names
df = pd.read_csv(csv_file, nrows=0)
print("Column names in the CSV file:")
print(list(df.columns))