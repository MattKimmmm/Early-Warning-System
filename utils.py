import pandas as pd
import os

# read csv file
def read_csv(file_name):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "data", file_name)

    try:
        df = pd.read_csv(file_path)
        print(df.head(5))
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    
# print unique column values given file name and column name
def unique_column_values(file_name, column_name):
    df = read_csv(file_name)
    if df is not None:
        print(df[column_name].unique())

# get icd codes given keyword
def get_icd_codes(keyword, file='D_ICD_DIAGNOSES.csv'):
    df = read_csv(file)
    
    # get the value of icd9_code column where either of short_title and long_title column contains the keyword
    icd_codes = df['icd9_code'][(df['short_title'].str.contains(keyword, case=False)) | 
                                (df['long_title'].str.contains(keyword, case=False))]
    print(f"ICD codes for {keyword}:")
    # row_id, icd_code
    print(icd_codes.head(5))