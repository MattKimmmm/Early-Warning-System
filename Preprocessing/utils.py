import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os


# read csv file
def read_csv(file_name, columns_to_use=None):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "data", file_name)

    # if column names are in lowercase
    if columns_to_use is not None:
        columns_to_use = [column.lower() for column in columns_to_use]

    try:
        # prevent memory explosion
        df_list = []
        chunks = None
        if columns_to_use is not None:
            # chunks = pd.read_csv(file_path, usecols=columns_to_use, chunksize=1000, low_memory=False)

            df = pd.read_csv(file_path, usecols=columns_to_use, low_memory=False)
        else:
            # chunks = pd.read_csv(file_path, chunksize=1000, low_memory=False)
            df = pd.read_csv(file_path, low_memory=False)
        
        # for chunk in chunks:
        #     chunk.columns = chunk.columns.str.upper()
        #     df_list.append(chunk)
        
        # df = pd.concat(df_list, axis=0)
            
        df.columns = df.columns.str.upper()
        # print(df.head(5))
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    
# print unique columns given df
def unique_column(df, name):
    print(f"Unique Columns in {name}")
    print(df.columns.tolist())
    
# print unique column values given file name and column name
def unique_column_values(file_name, column_name):
    df = read_csv(file_name)
    if df is not None:
        print(df[column_name].unique())

# print unique column values given df and column name
def unique_column_df(df, column_name):
    print(df[column_name].unique())
    print(f"Number of unique values: {len(df[column_name].unique())}")
    return df[column_name].unique()

# get icd codes given keyword
def get_icd_codes(keyword, file='D_ICD_DIAGNOSES.csv'):
    df = read_csv(file)
    unique_column(df, "D_ICD_DIAGNOSES")
    
    # get the value of icd9_code column where either of short_title and long_title column contains the keyword
    icd_codes = df['ICD9_CODE'][(df['SHORT_TITLE'].str.contains(keyword, case=False)) | 
                                (df['LONG_TITLE'].str.contains(keyword, case=False))]
    
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

# get analytics for a given df and column name (for numeric columns)
def column_analytics_df(df, column_name, num_bins):
    print(f"Mean: {df[column_name].mean()}")
    print(f"Median: {df[column_name].median()}")
    print(f"Standard Deviation: {df[column_name].std()}")
    print(f"Minimum: {df[column_name].min()}")
    print(f"Maximum: {df[column_name].max()}")
    
    # 10-bin histogram
    plt.hist(df[column_name], bins=num_bins)
    plt.show()

# save the list as npy file
def save_to_npy(list, file_name):
    np_array = np.array(list)
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data", "input", file_name)
    np.save(file_path, np_array)
    print(f"Data saved to {file_path}")

# save single pd dataframe to csv
def save_to_csv(df, file_name):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data", "processed", file_name)
    df.to_csv(file_path, index=False)
    print(f"Dataframe saved to {file_path}")

# save multiple pd dataframes to csv
def save_to_csv_multiple(dfs, file_names):
    for i in range(len(dfs)):
        save_to_csv(dfs[i], file_names[i])

# load single np
def load_np(file_name):
    # print(os.getcwd())
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data", "input", file_name)
    # print(file_path)

    loaded = np.load(file_path)
    print(f"Loaded {file_name}")
    print(f"shape: {loaded.shape}")
    print(loaded)
    return 

# load single pd dataframe from csv
def load_from_csv(file_name):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data", "processed", file_name)
    df = pd.read_csv(file_path)
    return df

# load multiple pd daraframes from csv
def load_from_csv_multiple(file_names):
    dfs = []
    for file_name in file_names:
        dfs.append(load_from_csv(file_name))
    return dfs

# given a joined patient/event df, split it into dfs for each patient
def split_by_patients(df):
    unique_patients = df['SUBJECT_ID'].unique()
    patient_dfs = []
    for patient in unique_patients:
        patient_dfs.append(df[df['SUBJECT_ID'] == patient])
    return patient_dfs

# given patient-specific event df, aggregate items by hour
def aggregate_events_by_time(df, intime):
    # print(intime)
    # print(pd.to_datetime(intime))
    intime = pd.to_datetime(intime)
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])

    df['HOUR_DIFF'] = ((df['CHARTTIME'] - intime).dt.total_seconds() // 3600).astype(int)
    hourly_dfs = []

    for hour in range(72):
        hourly_df = df[df['HOUR_DIFF'] == hour]
        if hourly_df.empty:
            # append a empty df
            hourly_dfs.append(pd.DataFrame())
            # print(f"Hour {hour}: 0 events")
        else:
            hourly_dfs.append(hourly_df)
            # print(f"Hour {hour}: {len(hourly_df)} events")
    
    return hourly_dfs

# given patient-specific event df, aggregate unique items from events df return aggregated dfs for each item
def aggregate_events_for_item(df, unique_items):
    item_dfs = []
    count = 0
    for item in unique_items:
        df_item = df[df['ITEMID'] == item]
        if df_item.empty:
            # append a empty df
            item_dfs.append(pd.DataFrame())
        else:
            item_dfs.append(df_item)
            count += 1
        # print(f"Number of events for item {item}: {df_item.shape[0]}")
        # print(df_item.head(5))
    # print(f"Number of unique items: {len(unique_items)}")
    # print(f"Number of items excluded: {len(unique_items) - len(item_dfs)}")
    # print(f"Number of items included: {count}\n")
    return item_dfs

# given patient-time-specific df, calculate the average of "VALUE" column
def column_average(df, column_name):
    return df[column_name].mean()