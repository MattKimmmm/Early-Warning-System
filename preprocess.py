import pandas as pd
import os

from utils import read_csv, get_icd_codes, save_to_csv_multiple, load_from_csv_multiple, unique_column_df, column_analytics_df, load_from_csv

# aggregate patient/event tables for a given keyword (deterioration)
def patients_events(keyword):
    # get icd codes for the given keyword
    icd_codes = get_icd_codes(keyword)

    if icd_codes is not None:
        # get subject_id and hadm_id from DIAGNOSES_ICD where icd9_code is in icd_codes
        df_diagnoses_icd = read_csv('DIAGNOSES_ICD.csv')
        df_diagnoses_icd = df_diagnoses_icd[df_diagnoses_icd['ICD9_CODE'].isin(icd_codes)]
        # print(df_diagnoses_icd.head(5))

        # get patients from PATIENTS where subject_id is in df_diagnoses_icd
        df_patients = read_csv('PATIENTS.csv')
        df_patients = df_patients[df_patients['SUBJECT_ID'].isin(df_diagnoses_icd['SUBJECT_ID'])]
        print(df_patients.head(5))
        print(f"Number of patients with {keyword}: {df_patients.shape[0]}")

        # get icu stays from ICUSTAYS where subject_id is in df_patients
        df_icustays = read_csv('ICUSTAYS.csv')
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
        df_chart_events= read_csv('CHARTEVENTS.csv')
        df_chart_events = df_chart_events[df_chart_events['ICUSTAY_ID'].isin(df_icustays['ICUSTAY_ID'])]
        print(df_chart_events.head(5))
        print(f"Number of chart events with {keyword}: {df_chart_events.shape[0]}\n")

        columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID", "VALUENUM", "VALUEUOM", "FLAG"]
        # df_output_events = read_csv('OUTPUTEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        df_output_events = read_csv('OUTPUTEVENTS.csv')
        df_output_events = df_output_events[df_output_events['ICUSTAY_ID'].isin(df_icustays['ICUSTAY_ID'])]
        print(df_output_events.head(5))
        print(f"Number of output events with {keyword}: {df_output_events.shape[0]}\n")

        columns_to_use = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM", "CGID"]
        # df_lab_events = read_csv('LABEVENTS.csv', columns_to_use=columns_to_use)
        # for demo
        df_lab_events = read_csv('LABEVENTS.csv')
        df_lab_icu_joined = pd.merge(df_lab_events, df_icustays, on=['SUBJECT_ID', 'HADM_ID'])

        df_lab_icu_specific = df_lab_icu_joined[
        (df_lab_icu_joined['CHARTTIME'] >= df_lab_icu_joined['INTIME']) &
        (df_lab_icu_joined['CHARTTIME'] <= df_lab_icu_joined['OUTTIME'])
        ]
        print(df_lab_icu_specific.head(5))
        print(f"Number of lab events in ICU with {keyword}: {df_lab_icu_specific.shape[0]}\n")

        # read dictionaries
        columns_to_use = ["ROW_ID", "ITEMID", "DBSOURCE", "LINKSTO"]
        # df_d_items = read_csv('D_ITEMS.csv', columns_to_use=columns_to_use)
        # for demo
        df_d_items = read_csv('D_ITEMS.csv')
        print(df_d_items.head(5))
        print(f"Number of items: {df_d_items.shape[0]}\n")

        columns_to_use = ["ROW_ID", "ITEMID", "LABEL", "FLUID", "CATEGORY", "LOINC_CODE"]
        # df_d_labitems = read_csv('D_LABITEMS.csv', columns_to_use=columns_to_use)
        # for demo
        df_d_labitems = read_csv('D_LABITEMS.csv')
        print(df_d_labitems.head(5))
        print(f"Number of lab items: {df_d_labitems.shape[0]}\n")

        # join events with dictionaries for item description
        df_chart_events = pd.merge(df_chart_events, df_d_items, on='ITEMID')
        df_output_events = pd.merge(df_output_events, df_d_items, on='ITEMID')
        df_lab_icu_specific = pd.merge(df_lab_icu_specific, df_d_labitems, on='ITEMID')

        # save df to csv
        save_to_csv_multiple([df_patients, df_chart_events, df_output_events, df_lab_icu_specific],
                             [f'patients_{keyword}', f'chart_events_{keyword}',
                              f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])

        return (df_patients, df_chart_events, df_output_events, df_lab_icu_specific)

# aggregate events into hourly bins
def aggregate_events(keyword):
    # load dataframes
    df_patients = load_from_csv(f'patients_{keyword}')
    [df_chart_event, df_output_event, df_lab_icu_specific] = load_from_csv_multiple([f'chart_events_{keyword}', f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])

    # print(f"Unique items in CHARTEVENTS with {keyword}:")
    # unique_column_df(df_chart_event, 'ITEMID')

    # print(f"Unique items in OUTPUTEVENTS with {keyword}:")
    # unique_column_df(df_output_event, 'ITEMID')

    # print(f"Unique items in LABEVENTS with {keyword}:")
    # unique_column_df(df_lab_icu_specific, 'ITEMID')

    # aggregate events by subjects
    df_chart_patient = pd.merge(df_patients, df_chart_event, on='SUBJECT_ID', how='inner')
    # print(f"Number of unique patients with {keyword}: {df_chart_patient['SUBJECT_ID'].nunique()}")
    # print(f"Number of chart events with {keyword}: {df_chart_patient.shape[0]}")
    # print(df_chart_patient.head(5))
    df_output_patient = pd.merge(df_patients, df_output_event, on='SUBJECT_ID', how='inner')
    # print(f"Number of unique patients with {keyword}: {df_output_patient['SUBJECT_ID'].nunique()}")
    # print(f"Number of output events with {keyword}: {df_output_patient.shape[0]}")
    # print(df_output_patient.head(5))
    df_lab_patient = pd.merge(df_patients, df_lab_icu_specific, on='SUBJECT_ID', how='inner')
    # print(f"Number of unique patients with {keyword}: {df_lab_patient['SUBJECT_ID'].nunique()}")
    # print(f"Number of lab events with {keyword}: {df_lab_patient.shape[0]}")
    # print(df_lab_patient.head(5))

    