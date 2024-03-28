# What will be my features? important lab results 

# Feature list
#
#
# 1. Create a model for the specified hour with (cardiact arrest, ARF, heart attack)
#    So example for hour 3, search the INTIME+3 ~ INTIME+4   
# 2. Get all unique ITEMIDS = [40055, 40069, 40085, ...,40051] from OUTPUTEVENTS
#    for that hour.
# 3. For missing data fill it in with mean data.
# 4. y...0 or 1 for cardiac arrest but this doesn't work for our current
#   purpose predicting the risk of cardiac arrest... like all of the y variables are 1 right now.

# Instead should I mix with (cardiact arrest, ARF, heart attack) = 0,1,2


import pandas as pd
import numpy as np
from datetime import datetime


#1. When you get back home, extract the data for ARF and heart attack, merge all csv files.

df_OUTPUTEVENTS = pd.read_csv('../Preprocessing/preprocessed_data/OUTPUTEVENTS.csv')
df_ICUSTAYS = pd.read_csv('../Preprocessing/preprocessed_data/ICUSTAYS.csv')

#1.1 Finding the model for hour h

# STORETIME: in OUTPUTEVENTS when the lab result was recorded.
# INTIME: in ICUSTAYS, make sure to get matching SUBJECT_ID

h = 3  #Need to get it from EWS.py

df_OUTPUTEVENTS['STORETIME'] = pd.to_datetime(df_OUTPUTEVENTS['STORETIME'])
df_ICUSTAYS['INTIME'] = pd.to_datetime(df_ICUSTAYS['INTIME'])

df_OUTPUTEVENTS.set_index('SUBJECT_ID', inplace=True)
df_ICUSTAYS.set_index('SUBJECT_ID', inplace=True)

# Define the hour h for which you want to predict
h = 3

# Use a vectorized operation to calculate the time window for each ICU stay
df_ICUSTAYS['start_time'] = df_ICUSTAYS['INTIME'] + pd.Timedelta(hours=h)
df_ICUSTAYS['end_time'] = df_ICUSTAYS['start_time'] + pd.Timedelta(hours=1)

# Merge on 'SUBJECT_ID' and filter rows based on the time window
# This is a more efficient way than iterating row by row
merged_df = df_OUTPUTEVENTS.merge(df_ICUSTAYS[['start_time', 'end_time']], left_index=True, right_index=True)
filtered_output = merged_df[(merged_df['STORETIME'] >= merged_df['start_time']) & (merged_df['STORETIME'] < merged_df['end_time'])]

# Pivot and fill missing values
features = filtered_output.pivot_table(index='SUBJECT_ID', columns='ITEMID', values='VALUE', aggfunc='mean')
features = features.dropna()

print(features)










