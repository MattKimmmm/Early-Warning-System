import pandas as pd
import numpy as np

# ARF icd9: 51881
# related item_id
# 50803: Calculated Bicarbonate, Whole Blood
# 50817: Oxygen Saturation
# 50820: pH
itemIds = [50803,50817,50820]

# Cardiac Arrest icd9: 4275
# related item_id
# 211: Heart Rate
# 220045: Blood Pressure
# 220277: ECG
# 50822: Potassium
# 51003: Troponin
itemIds1 = [211, 220045, 220277, 50822, 51003]
items = itemIds + itemIds1
icd9_code = ["51881","4275"]

path = "/Users/yurockheo/Desktop/Early-Warning-System/data/"
df1 = pd.read_csv(path+"DIAGNOSES_ICD.csv")
df2 = pd.read_csv(path+"LABEVENTS.csv")
subjectIds = df1.loc[df1['icd9_code'].isin(icd9_code)].loc[:,'subject_id'].to_numpy()

labevent_arf = df2[(df2['itemid'].isin(items)) & (df2['subject_id'].isin(subjectIds))]

# Train Test Split
icd9_subset = df1[df1['icd9_code'].isin(icd9_code)][['subject_id', 'icd9_code']].drop_duplicates()
data = pd.merge(df2, icd9_subset, on='subject_id', how='inner')
data = data[data['itemid'].isin(items)]


X = data.drop(columns=['icd9_code', 'hadm_id', 'charttime','row_id'])
y = data['icd9_code']


X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_
