import pandas as pd
import os
import numpy as np
import time
import torch

from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils import read_csv, get_icd_codes, save_to_csv_multiple, load_from_csv_multiple, unique_column_df, column_analytics_df, load_from_csv
from utils import split_by_patients, aggregate_events_for_item, aggregate_events_by_time, column_average, save_to_npy
from utils import unique_column, common_patients, load_np

# aggregate patient/event tables for a given keyword (deterioration)
def patients_events(keyword):
    # get icd codes for the given keyword
    icd_codes = get_icd_codes(keyword)

    if icd_codes is not None:
        # get subject_id and hadm_id from DIAGNOSES_ICD where icd9_code is in icd_codes
        df_diagnoses_icd = read_csv('DIAGNOSES_ICD.csv')
        df_diagnoses_icd = df_diagnoses_icd[df_diagnoses_icd['ICD9_CODE'].isin(icd_codes)]
        print("df_diagnoses_icd.head")
        print(df_diagnoses_icd.head(5))

        # get patients from PATIENTS where subject_id is in df_diagnoses_icd
        df_patients = read_csv('PATIENTS.csv')
        print(df_patients.head(5))
        print(f"Number of patients : {df_patients.shape[0]}")
        df_patients = df_patients[df_patients['SUBJECT_ID'].isin(df_diagnoses_icd['SUBJECT_ID'])]
        print(df_patients.head(5))
        print(f"Number of patients with {keyword}: {df_patients.shape[0]}")

        # get icu stays from ICUSTAYS where subject_id is in df_patients
        df_icustays = read_csv('ICUSTAYS.csv')
        df_icustays_subset = df_icustays[['ICUSTAY_ID', 'INTIME', 'OUTTIME']]
        df_icustays = df_icustays[df_icustays['SUBJECT_ID'].isin(df_patients['SUBJECT_ID'])]
        # only stays < 3 days
        # df_icustays = df_icustays[df_icustays['LOS'] <= 3]
        print(df_icustays.head(5))
        print(f"Number of ICU stays with {keyword}: {df_icustays.shape[0]}")

        # df_icustays_hadm = df_icustays[df_icustays['hadm_id'].isin(df_diagnoses_icd['hadm_id'])]
        # print(df_icustays_hadm.head(5))
        # print(f"(using hadm)Number of ICU stays with {keyword}: {df_icustays_hadm.shape[0]}")
        
        # get lab results for selected patients' icu stays
        columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "STORETIME", "CGID", "VALUENUM", "VALUEUOM", "WARNING", "ERROR"]
        # df_chart_events= read_csv('CHARTEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_chart_events= read_csv('CHARTEVENTS.csv')
        # df_chart_events = pd.merge(df_chart_events, df_icustays_subset, on='ICUSTAY_ID', how='inner')
        # print(df_chart_events.head(5))
        # print(f"Number of chart events with {keyword}: {df_chart_events.shape[0]}\n")

        columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM", "CGID"]
        df_output_events = read_csv('OUTPUTEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_output_events = read_csv('OUTPUTEVENTS.csv')
        df_output_events = pd.merge(df_output_events, df_icustays_subset, on='ICUSTAY_ID', how='inner')
        print(df_output_events.head(5))
        print(f"Number of output events with {keyword}: {df_output_events.shape[0]}\n")

        columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUENUM", "VALUEUOM"]
        df_lab_events = read_csv('LABEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_lab_events = read_csv('LABEVENTS.csv')
        df_lab_icu_joined = pd.merge(df_lab_events, df_icustays, on=['SUBJECT_ID', 'HADM_ID'])

        df_lab_icu_specific = df_lab_icu_joined[
        (df_lab_icu_joined['CHARTTIME'] >= df_lab_icu_joined['INTIME']) &
        (df_lab_icu_joined['CHARTTIME'] <= df_lab_icu_joined['OUTTIME'])
        ]
        print(df_lab_icu_specific.head(5))
        print(f"Number of lab events in ICU with {keyword}: {df_lab_icu_specific.shape[0]}\n")

        # read dictionaries
        columns_to_use = ["ROW_ID", "ITEMID", "DBSOURCE", "LINKSTO"]
        df_d_items = read_csv('D_ITEMS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_d_items = read_csv('D_ITEMS.csv')
        print(df_d_items.head(5))
        print(f"Number of items: {df_d_items.shape[0]}\n")

        columns_to_use = ["ROW_ID", "ITEMID", "LABEL", "FLUID", "CATEGORY", "LOINC_CODE"]
        df_d_labitems = read_csv('D_LABITEMS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_d_labitems = read_csv('D_LABITEMS.csv')
        print(df_d_labitems.head(5))
        print(f"Number of lab items: {df_d_labitems.shape[0]}\n")

        # join events with dictionaries for item description
        # df_chart_events = pd.merge(df_chart_events, df_d_items, on='ITEMID')
        df_output_events = pd.merge(df_output_events, df_d_items, on='ITEMID')
        df_lab_icu_specific = pd.merge(df_lab_icu_specific, df_d_labitems, on='ITEMID')

        #


        # save df to csv
        # save_to_csv_multiple([df_patients, df_chart_events, df_output_events, df_lab_icu_specific],
        #                      [f'patients_{keyword}', f'chart_events_{keyword}',
        #                       f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])
        save_to_csv_multiple([df_patients, df_output_events, df_lab_icu_specific],
                             [f'patients_{keyword}',
                              f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])

        # return (df_patients, df_chart_events, df_output_events, df_lab_icu_specific)
        return (df_patients, df_output_events, df_lab_icu_specific)

# aggregate patient/event tables for a given keyword (deterioration),
# but sample negative patients (not diagnosed with keyword)
def patients_events_nagative(keyword, num_patients, random_seed):
    # get icd codes for the given keyword
    icd_codes = get_icd_codes(keyword)

    if icd_codes is not None:
        # get subject_id and hadm_id from DIAGNOSES_ICD where icd9_code is in icd_codes
        df_diagnoses_icd = read_csv('DIAGNOSES_ICD.csv')
        df_diagnoses_icd = df_diagnoses_icd[df_diagnoses_icd['ICD9_CODE'].isin(icd_codes)]
        print("df_diagnoses_icd.head")
        print(df_diagnoses_icd.head(5))

        # get patients from PATIENTS where subject_id is in df_diagnoses_icd
        df_patients = read_csv('PATIENTS.csv')
        print(df_patients.head(5))
        print(f"Number of patients : {df_patients.shape[0]}")
        df_patients_not_diagnosed = df_patients[~df_patients['SUBJECT_ID'].isin(df_diagnoses_icd['SUBJECT_ID'])]
        df_patients_not_diagnosed = df_patients_not_diagnosed.sample(n=num_patients, random_state=random_seed)

        print(df_patients_not_diagnosed.head(5))
        print(f"Number of patients without {keyword}: {df_patients_not_diagnosed.shape[0]}")

        # get icu stays from ICUSTAYS where subject_id is in df_patients
        df_icustays = read_csv('ICUSTAYS.csv')
        df_icustays_subset = df_icustays[['ICUSTAY_ID', 'INTIME', 'OUTTIME']]
        df_icustays = df_icustays[df_icustays['SUBJECT_ID'].isin(df_patients_not_diagnosed['SUBJECT_ID'])]
        # only stays < 3 days
        # df_icustays = df_icustays[df_icustays['LOS'] <= 3]
        print(df_icustays.head(5))
        print(f"Number of ICU stays with {keyword}: {df_icustays.shape[0]}")

        # df_icustays_hadm = df_icustays[df_icustays['hadm_id'].isin(df_diagnoses_icd['hadm_id'])]
        # print(df_icustays_hadm.head(5))
        # print(f"(using hadm)Number of ICU stays with {keyword}: {df_icustays_hadm.shape[0]}")
        
        # get lab results for selected patients' icu stays
        # columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "STORETIME", "CGID", "VALUENUM", "VALUEUOM", "WARNING", "ERROR"]
        # df_chart_events= read_csv('CHARTEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_chart_events= read_csv('CHARTEVENTS.csv')
        # df_chart_events = pd.merge(df_chart_events, df_icustays_subset, on='ICUSTAY_ID', how='inner')
        # print(df_chart_events.head(5))
        # print(f"Number of chart events with {keyword}: {df_chart_events.shape[0]}\n")

        columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM", "CGID"]
        df_output_events = read_csv('OUTPUTEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_output_events = read_csv('OUTPUTEVENTS.csv')
        df_output_events = pd.merge(df_output_events, df_icustays_subset, on='ICUSTAY_ID', how='inner')
        print(df_output_events.head(5))
        print(f"Number of output events with {keyword}: {df_output_events.shape[0]}\n")

        columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUENUM", "VALUEUOM"]
        df_lab_events = read_csv('LABEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_lab_events = read_csv('LABEVENTS.csv')
        df_lab_icu_joined = pd.merge(df_lab_events, df_icustays, on=['SUBJECT_ID', 'HADM_ID'])

        df_lab_icu_specific = df_lab_icu_joined[
        (df_lab_icu_joined['CHARTTIME'] >= df_lab_icu_joined['INTIME']) &
        (df_lab_icu_joined['CHARTTIME'] <= df_lab_icu_joined['OUTTIME'])
        ]
        print(df_lab_icu_specific.head(5))
        print(f"Number of lab events in ICU with {keyword}: {df_lab_icu_specific.shape[0]}\n")

        # read dictionaries
        columns_to_use = ["ROW_ID", "ITEMID", "DBSOURCE", "LINKSTO"]
        df_d_items = read_csv('D_ITEMS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_d_items = read_csv('D_ITEMS.csv')
        print(df_d_items.head(5))
        print(f"Number of items: {df_d_items.shape[0]}\n")

        columns_to_use = ["ROW_ID", "ITEMID", "LABEL", "FLUID", "CATEGORY", "LOINC_CODE"]
        df_d_labitems = read_csv('D_LABITEMS.csv', columns_to_use=columns_to_use)
        # for demo
        # df_d_labitems = read_csv('D_LABITEMS.csv')
        print(df_d_labitems.head(5))
        print(f"Number of lab items: {df_d_labitems.shape[0]}\n")

        # join events with dictionaries for item description
        # df_chart_events = pd.merge(df_chart_events, df_d_items, on='ITEMID')
        df_output_events = pd.merge(df_output_events, df_d_items, on='ITEMID')
        df_lab_icu_specific = pd.merge(df_lab_icu_specific, df_d_labitems, on='ITEMID')

        # save df to csv
        # save_to_csv_multiple([df_patients, df_chart_events, df_output_events, df_lab_icu_specific],
        #                      [f'patients_{keyword}', f'chart_events_{keyword}',
        #                       f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])
        save_to_csv_multiple([df_patients_not_diagnosed, df_output_events, df_lab_icu_specific],
                             [f'patients_{keyword}_neg',
                              f'output_events_{keyword}_neg', f'lab_icu_specific_{keyword}_neg'])

        # return (df_patients, df_chart_events, df_output_events, df_lab_icu_specific)
        return (df_patients_not_diagnosed, df_output_events, df_lab_icu_specific)

# aggregate events into hourly bins
def aggregate_events_output_lab(keyword):
    # load dataframes
    df_patients = load_from_csv(f'patients_{keyword}_neg')
    # [df_chart_event, df_output_event, df_lab_icu_specific] = load_from_csv_multiple([f'chart_events_{keyword}', f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])
    [df_output_event, df_lab_icu_specific] = load_from_csv_multiple([f'output_events_{keyword}_neg', f'lab_icu_specific_{keyword}_neg'])
    unique_column(df_output_event, "df_output_event")
    unique_column(df_lab_icu_specific, "df_lab_icu_specific")

    # print(f"Unique items in CHARTEVENTS with {keyword}:")
    # df_chart_unique = unique_column_df(df_chart_event, 'ITEMID')

    df_output_unique = unique_column_df(df_output_event, 'ITEMID')
    print(f"Unique items in OUTPUTEVENTS with {keyword}: {len(df_output_unique)} items")
    print(df_output_unique)

    df_lab_unique = unique_column_df(df_lab_icu_specific, 'ITEMID')
    print(f"Unique items in LABEVENTS with {keyword}: {len(df_lab_unique)} items")
    print(df_lab_unique)

    # join patients with events
    # df_chart_patient = pd.merge(df_patients, df_chart_event, on='SUBJECT_ID', how='inner')
    # print(f"Number of unique patients with {keyword}: {df_chart_patient['SUBJECT_ID'].nunique()}")
    # print(f"Number of chart events with {keyword}: {df_chart_patient.shape[0]}")
    # print(df_chart_patient.head(5))
    df_output_patient = pd.merge(df_patients, df_output_event, on='SUBJECT_ID', how='inner')
    print(f"Number of unique patients with {keyword}: {df_output_patient['SUBJECT_ID'].nunique()}")
    print(f"Number of output events with {keyword}: {df_output_patient.shape[0]}")
    unique_column(df_output_patient, "df_output_patient")
    # print(df_output_patient.head(5))
    df_lab_patient = pd.merge(df_patients, df_lab_icu_specific, on='SUBJECT_ID', how='inner')
    print(f"Number of unique patients with {keyword}: {df_lab_patient['SUBJECT_ID'].nunique()}")
    print(f"Number of lab events with {keyword}: {df_lab_patient.shape[0]}")
    unique_column(df_lab_patient, "df_lab_patient")
    # print(df_lab_patient.head(5))

    # merge the two resulting dfs, include patients that are present in both
    patient_output = split_by_patients(df_output_patient)
    patient_lab = split_by_patients(df_lab_patient)
    patient_output, patient_lab = common_patients(patient_output, patient_lab)

    # aggregate events by subjects
    # input shape: [# patients, # hours (72), # unique items]
    print(f"Starting Aggregation for Output Events for {len(patient_output)} patients")
    since = time.time()
    input_output = []
    args_list_output = []

    # aggregate args for multiprocessing
    for patient in patient_output:
        args = (patient, df_output_unique)
        args_list_output.append(args)

    with Pool(processes=8) as pool:
        results = pool.map(patient_specifics, args_list_output)
    
    for r in results:
        input_output.append(r)

    # print input shape
    num_patients = len(input_output)
    num_hours = len(input_output[0])
    num_items = len(input_output[0][0])

    print(f"Outputevents Aggregation done in {time.time() - since} seconds")
    # print(input[0][0])
    print(f"Input dimensions: {num_patients} patients x {num_hours} hours x {num_items} items")
    # print(input)
    save_to_npy(input_output, f'input_{keyword}_output_avg_neg')


    print(f"Starting Aggregation for Lab Events for {len(patient_lab)} patients")
    since = time.time()
    input_lab = []
    args_list_lab = []

    # aggregate args for multiprocessing
    for patient in patient_lab:
        args = (patient, df_lab_unique)
        args_list_lab.append(args)

    with Pool(processes=8) as pool:
        results = pool.map(patient_specifics, args_list_lab)
    
    for r in results:
        input_lab.append(r)

    # print input shape
    num_patients = len(input_lab)
    num_hours = len(input_lab[0])
    num_items = len(input_lab[0][0])

    print(f"Labevents Aggregation done in {time.time() - since} seconds")
    # print(input[0][0])
    print(f"Input dimensions: {num_patients} patients x {num_hours} hours x {num_items} items")
    # print(input)
    save_to_npy(input_lab, f'input_{keyword}_lab_avg_neg')

# aggregate events into hourly bins for negative samples
def aggregate_events_output_lab_negative(keyword):
    # load dataframes
    df_patients = load_from_csv(f'patients_{keyword}_neg')
    # [df_chart_event, df_output_event, df_lab_icu_specific] = load_from_csv_multiple([f'chart_events_{keyword}', f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])
    [df_output_event, df_lab_icu_specific] = load_from_csv_multiple([f'output_events_{keyword}_neg', f'lab_icu_specific_{keyword}_neg'])
    unique_column(df_output_event, "df_output_event")
    unique_column(df_lab_icu_specific, "df_lab_icu_specific")

    [df_output_event_org, df_lab_icu_specific_org] = load_from_csv_multiple([f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])
    
    # print(f"Unique items in CHARTEVENTS with {keyword}:")
    # df_chart_unique = unique_column_df(df_chart_event, 'ITEMID')

    df_output_unique = unique_column_df(df_output_event_org, 'ITEMID')
    print(f"Unique items in OUTPUTEVENTS with {keyword}: {len(df_output_unique)} items")
    print(df_output_unique)

    df_lab_unique = unique_column_df(df_lab_icu_specific_org, 'ITEMID')
    print(f"Unique items in LABEVENTS with {keyword}: {len(df_lab_unique)} items")
    print(df_lab_unique)

    # join patients with events
    # df_chart_patient = pd.merge(df_patients, df_chart_event, on='SUBJECT_ID', how='inner')
    # print(f"Number of unique patients with {keyword}: {df_chart_patient['SUBJECT_ID'].nunique()}")
    # print(f"Number of chart events with {keyword}: {df_chart_patient.shape[0]}")
    # print(df_chart_patient.head(5))
    df_output_patient = pd.merge(df_patients, df_output_event, on='SUBJECT_ID', how='inner')
    print(f"Number of unique patients with {keyword}: {df_output_patient['SUBJECT_ID'].nunique()}")
    print(f"Number of output events with {keyword}: {df_output_patient.shape[0]}")
    unique_column(df_output_patient, "df_output_patient")
    # print(df_output_patient.head(5))
    df_lab_patient = pd.merge(df_patients, df_lab_icu_specific, on='SUBJECT_ID', how='inner')
    print(f"Number of unique patients with {keyword}: {df_lab_patient['SUBJECT_ID'].nunique()}")
    print(f"Number of lab events with {keyword}: {df_lab_patient.shape[0]}")
    unique_column(df_lab_patient, "df_lab_patient")
    # print(df_lab_patient.head(5))

    # merge the two resulting dfs, include patients that are present in both
    patient_output = split_by_patients(df_output_patient)
    patient_lab = split_by_patients(df_lab_patient)
    patient_output, patient_lab = common_patients(patient_output, patient_lab)

    # aggregate events by subjects
    # input shape: [# patients, # hours (72), # unique items]
    print(f"Starting Aggregation for Output Events for {len(patient_output)} patients")
    since = time.time()
    input_output = []
    args_list_output = []

    # aggregate args for multiprocessing
    for patient in patient_output:
        args = (patient, df_output_unique)
        args_list_output.append(args)

    with Pool(processes=8) as pool:
        results = pool.map(patient_specifics, args_list_output)
    
    for r in results:
        input_output.append(r)

    # print input shape
    num_patients = len(input_output)
    num_hours = len(input_output[0])
    num_items = len(input_output[0][0])

    print(f"Outputevents Aggregation done in {time.time() - since} seconds")
    # print(input[0][0])
    print(f"Input dimensions: {num_patients} patients x {num_hours} hours x {num_items} items")
    # print(input)
    save_to_npy(input_output, f'input_{keyword}_output_avg_neg')


    print(f"Starting Aggregation for Lab Events for {len(patient_lab)} patients")
    since = time.time()
    input_lab = []
    args_list_lab = []

    # aggregate args for multiprocessing
    for patient in patient_lab:
        args = (patient, df_lab_unique)
        args_list_lab.append(args)

    with Pool(processes=8) as pool:
        results = pool.map(patient_specifics, args_list_lab)
    
    for r in results:
        input_lab.append(r)

    # print input shape
    num_patients = len(input_lab)
    num_hours = len(input_lab[0])
    num_items = len(input_lab[0][0])

    print(f"Labevents Aggregation done in {time.time() - since} seconds")
    # print(input[0][0])
    print(f"Input dimensions: {num_patients} patients x {num_hours} hours x {num_items} items")
    # print(input)
    save_to_npy(input_lab, f'input_{keyword}_lab_avg_neg')

def patient_specifics(args):
    patient, df_unique = args

    since = time.time()
    patient_specific = []
    # print(f"Number of output events for patient {patient['SUBJECT_ID'].iloc[0]}: {patient.shape[0]}")
    # print(patient.head(5))

    intime = patient['INTIME'].iloc[0]

    # items_dfs = aggregate_events_for_item(patient, df_output_unique)

    times_dfs = aggregate_events_by_time(patient, intime)
    # print(f"times_dfs: {times_dfs}")
    # print(f"df_unique: {df_unique}")

    # for each hour, aggregate value by item and make a list
    for i, times_df in enumerate(times_dfs):
        unique_item_aggregated = []
        count = 0

        if times_df.empty:
            unique_item_aggregated = [0] * len(df_unique)
        else:
            items_dfs = aggregate_events_for_item(times_df, df_unique)
            # print(items_dfs)
            # print(f"items_dfs length: {len(items_dfs)}")
            for item_df in items_dfs:
                if item_df.empty:
                    unique_item_aggregated.append(0)
                else:
                    # print(f"Columns in item_df: {item_df.columns}")
                    item_df_numeric = pd.to_numeric(item_df["VALUE"], errors='coerce')
                    # print(f"item_df_numeric:\n{item_df_numeric}")

                    mean_val = item_df_numeric.mean()
                    if pd.isna(mean_val):
                        mean_val = 0
                        count -= 1

                    unique_item_aggregated.append(mean_val)
                    count += 1

            # print(unique_item_aggregated)
            # print(f"length: {len(unique_item_aggregated)}")
            # print(f"{count} out of {len(items_dfs)} items included for this hourly window")
        
        patient_specific.append(unique_item_aggregated)
    
    # print input shape
    num_hours = len(patient_specific)
    num_items = len(patient_specific[0])

    # for hour in range(num_hours):
    #     print(f"For hour {hour}, {len(patient_specific[0])} items present")
    #     print(patient_specific[hour][0])

    # print(f"Single patient processing took {time.time() - since} seconds")

    return patient_specific

# given keyword, load processed dataset and create dataloaders
def create_dataset(keyword, batch_size, random_seed) :
    # [# patients X 72 hours X # items]
    lab_name = f"input_" + keyword + "_lab_avg.npy"
    output_name = f"input_" + keyword + "_output_avg.npy"

    lab_name_neg = f"input_" + keyword + "_lab_avg_neg.npy"
    output_name_neg = f"input_" + keyword + "_output_avg_neg.npy"

    # load np files
    lab_np = load_np(lab_name)
    output_np = load_np(output_name)

    lab_neg_np = load_np(lab_name_neg)
    output_neg_np = load_np(output_name_neg)

    # aggregate items from labevents and outputevents
    combined = np.concatenate((output_np, lab_np), axis=2)
    combined_neg = np.concatenate((output_neg_np, lab_neg_np), axis=2)
    combined_len = len(combined)
    combined_neg_len = len(combined_neg)
    print(f"combined length: {combined_len}")
    print(f"combined_neg length: {combined_neg_len}")
    
    # truncate the positive dataset
    indices = np.random.choice(combined.shape[0], combined_neg_len, replace=False)
    combined = combined[indices]

    # labels: positive=1 | negative=0
    labels = np.ones((combined.shape[0], 1))
    labels_neg = np.zeros((combined_neg.shape[0], 1))

    final = np.concatenate((combined, combined_neg), axis=0)
    # print(f"final concatenated np array data shape: {final.shape}")
    final_labels = np.concatenate((labels, labels_neg), axis=0)
    # print(f"label shape: {final_labels.shape}")

    # Split the dataset into training and testing
    data_train, data_test, labels_train, labels_test = train_test_split(final, final_labels, test_size=0.2, random_state=random_seed, stratify=final_labels)
    print(f"data_train shape: {data_train.shape}")
    print(f"labels_train shape: {labels_train.shape}")
    print(f"data_test shape: {data_test.shape}")
    print(f"labels_test shape: {labels_test.shape}")

    # Convert to tensors
    data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
    labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
    data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
    labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "../data", "tensors")

    # save tensors and dataloader params
    torch.save(data_train_tensor, os.path.join(file_path, f'data_train_tensor_{keyword}.pt'))
    torch.save(labels_train_tensor, os.path.join(file_path, f'labels_train_tensor_{keyword}.pt'))
    torch.save(data_test_tensor, os.path.join(file_path, f'data_test_tensor_{keyword}.pt'))
    torch.save(labels_test_tensor, os.path.join(file_path, f'labels_test_tensor_{keyword}.pt'))

    # Save DataLoader parameters
    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 8  # Adjust according to your DataLoader setup
    }
    torch.save(dataloader_params, os.path.join(file_path, f'dataloader_params_{keyword}.pt'))

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(data_train_tensor, labels_train_tensor)
    test_dataset = TensorDataset(data_test_tensor, labels_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader