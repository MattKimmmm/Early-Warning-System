import pandas as pd
import matplotlib.pyplot as plt

import os


# read csv file
def read_csv(file_name):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "data", file_name)

    try:
        df = pd.read_csv(file_path)
        # print(df.head(5))
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
    
    # row_id, icd_code (77895 for newborn cardiac arrest, we might want to exclude this)
    # print(f"ICD codes for {keyword}:")
    # print(icd_codes.head(5))

    return icd_codes

# get analytics for a given column (for numeric columns)
def column_analytics(file_name, column_name, num_bins):
    df = read_csv(file_name)
    if df is not None:
        print(f"Mean: {df[column_name].mean()}")
        print(f"Median: {df[column_name].median()}")
        print(f"Standard Deviation: {df[column_name].std()}")
        print(f"Minimum: {df[column_name].min()}")
        print(f"Maximum: {df[column_name].max()}")
        
        # 10-bin histogram
        plt.hist(df[column_name], bins=num_bins)
        plt.show()

# save single pd dataframe to csv
def save_to_csv(df, file_name):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "data", "processed", file_name)
    df.to_csv(file_path, index=False)
    print(f"Dataframe saved to {file_path}")

# save multiple pd dataframes to csv
def save_to_csv_multiple(dfs, file_names):
    for i in range(len(dfs)):
        save_to_csv(dfs[i], file_names[i])