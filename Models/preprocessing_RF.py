# What will be my features? important lab results 

# Feature list
#
#
# 1. Create a model for the specified hour with (cardiact arrest, ARF, heart attack)
#    So example for hour 3, search the INTIME+3 ~ INTIME+4   
# 2. Get all unique ITEMIDS = [40055, 40059, 40048, 40049, 40054, 40060] from OUTPUTEVENTS
#    for that hour.
# 3. For missing data fill it in with mean data.
# 4. y...0 or 1 for cardiac arrest but this doesn't work for our current
#   purpose predicting the risk of cardiac arrest... like all of the y variables are 1 right now.

# Instead should I mix with (cardiact arrest, ARF, heart Storke) = 0,1,2


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#df_CHARTEVENTS = pd.read_csv('./BigData/CHARTEVENTS.csv')
#df_CHARTEVENTS = df_CHARTEVENTS.sample(frac=0.7)
df_OUTPUTEVENTS = pd.read_csv('./BigData/OUTPUTEVENTS.csv')
df_ICUSTAYS = pd.read_csv('./BigData/ICUSTAYS.csv')
df_DIAGNOSES_ICD = pd.read_csv('./BigData/DIAGNOSES_ICD.csv')
df_LABEVENTS = pd.read_csv('./BigData/LABEVENTS.csv')

#1. When you get back home, extract the data for ARF and heart attack, merge all csv files.

# ICD9_CODE
# Cardiac Arrest = 4280 => 0 , sample subjects = [175533, 133550, 145834], Item IDs to consider = []
# ARF = 51881  => 1   , ss = [124271, 190159, 176764], items  = []
# Heart Stroke = 4019 => 2, ss = [160617, 109518, 144980], items = []
# 4280, 4019


icd9_codes = ['4280', '51881', '4019']
icd9_to_target = {'4280': 0, '51881': 1, '4019': 2}

item_ids = [ 50971, 51221, 50983, 50912, 50902, 51006, 50882, 50868, 50882, 50868]  # Ensure ITEMID is an integer

# Put data in pd df
df_LABEVENTS['STORETIME'] = pd.to_datetime(df_LABEVENTS['CHARTTIME'])
df_OUTPUTEVENTS['STORETIME'] = pd.to_datetime(df_OUTPUTEVENTS['STORETIME'])
df_ICUSTAYS['INTIME'] = pd.to_datetime(df_ICUSTAYS['INTIME'])

# Get unique subject_ids with the icd9_codes
filtered_diagnoses = df_DIAGNOSES_ICD[df_DIAGNOSES_ICD['ICD9_CODE'].isin(icd9_codes)]
unique_subject_ids = filtered_diagnoses['SUBJECT_ID'].unique()
df_patients = pd.DataFrame(unique_subject_ids, columns=['SUBJECT_ID'])  # unique subject ids


# Combine the chart and output events, then merge with ICU stays to align on SUBJECT_ID
df_events = pd.concat([df_LABEVENTS])
df_events = df_events[df_events['ITEMID'].isin(item_ids)]
df_events = df_events.merge(df_ICUSTAYS[['SUBJECT_ID', 'INTIME']], on='SUBJECT_ID', how='inner')

df_events['STORETIME'] = pd.to_datetime(df_events['STORETIME'])
df_events['INTIME'] = pd.to_datetime(df_events['INTIME'])
df_events = df_events[(df_events['STORETIME'] >= df_events['INTIME']) & (df_events['STORETIME'] <= df_events['INTIME'] + pd.Timedelta(hours=48))]

df_pivoted = df_events.pivot_table(index='SUBJECT_ID', columns='ITEMID', values='VALUE', aggfunc='first')

icd9_mapping = df_DIAGNOSES_ICD[df_DIAGNOSES_ICD['ICD9_CODE'].isin(icd9_codes)].drop_duplicates(subset='SUBJECT_ID')
icd9_mapping['ICD9_CODE'] = icd9_mapping['ICD9_CODE'].astype(str)
icd9_mapping.set_index('SUBJECT_ID', inplace=True)
icd9_code_map = icd9_mapping['ICD9_CODE'].to_dict()
df_pivoted['TARGET'] = df_pivoted.index.map(icd9_code_map)
df_pivoted['TARGET'] = df_pivoted['TARGET'].map(icd9_to_target)

# Reset index to turn SUBJECT_ID from an index into a column
df_pivoted = df_pivoted.dropna()
df_pivoted.reset_index(inplace=True)


features = df_pivoted.drop(['TARGET', 'SUBJECT_ID'], axis=1)

columns = [ 50971, 51221, 50983, 50912, 50902, 51006, 50882, 50868, 50882, 50868]

for column in columns:
    features[column] = pd.to_numeric(features[column], errors ='coerce')

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
df_features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_features_scaled['TARGET'] = df_pivoted['TARGET']
df_features_scaled['SUBJECT_ID'] = df_pivoted.reset_index()['SUBJECT_ID']


# Optionally, you can save this pivoted DataFrame to a new CSV
df_features_scaled.to_csv('./data.csv', index=False)

















