import pandas as pd
import os

from utils import read_csv, get_icd_codes, save_to_csv_multiple

# aggregate patient/event tables for a given keyword (deterioration)
def patients_events(keyword):
    # get icd codes for the given keyword
    icd_codes = get_icd_codes(keyword)

    if icd_codes is not None:
        # get subject_id and hadm_id from DIAGNOSES_ICD where icd9_code is in icd_codes
        df_diagnoses_icd = read_csv('DIAGNOSES_ICD.csv')
        df_diagnoses_icd = df_diagnoses_icd[df_diagnoses_icd['icd9_code'].isin(icd_codes)]
        # print(df_diagnoses_icd.head(5))

        # get patients from PATIENTS where subject_id is in df_diagnoses_icd
        df_patients = read_csv('PATIENTS.csv')
        df_patients = df_patients[df_patients['subject_id'].isin(df_diagnoses_icd['subject_id'])]
        # print(df_patients.head(5))
        # print(f"Number of patients with {keyword}: {df_patients.shape[0]}")

        # get icu stays from ICUSTAYS where subject_id is in df_patients
        df_icustays = read_csv('ICUSTAYS.csv')
        df_icustays = df_icustays[df_icustays['subject_id'].isin(df_patients['subject_id'])]
        # only stays < 2 days
        # df_icustays = df_icustays[df_icustays['los'] <= 2]
        print(df_icustays.head(5))
        print(f"Number of ICU stays with {keyword}: {df_icustays.shape[0]}")

        # df_icustays_hadm = df_icustays[df_icustays['hadm_id'].isin(df_diagnoses_icd['hadm_id'])]
        # print(df_icustays_hadm.head(5))
        # print(f"(using hadm)Number of ICU stays with {keyword}: {df_icustays_hadm.shape[0]}")
        
        # get lab results for selected patients' icu stays
        df_chart_events= read_csv('CHARTEVENTS.csv')
        df_chart_events = df_chart_events[df_chart_events['icustay_id'].isin(df_icustays['icustay_id'])]
        # print(df_chart_events.head(5))
        # print(f"Number of lab results with {keyword}: {df_chart_events.shape[0]}\n")

        df_output_events = read_csv('OUTPUTEVENTS.csv')
        df_output_events = df_output_events[df_output_events['icustay_id'].isin(df_icustays['icustay_id'])]
        # print(df_output_events.head(5))
        # print(f"Number of output events with {keyword}: {df_output_events.shape[0]}\n")

        df_lab_events = read_csv('LABEVENTS.csv')
        df_lab_icu_joined = pd.merge(df_lab_events, df_icustays, on=['subject_id', 'hadm_id'])
        # print(df_lab_icu_joined.head(5))
        # print(f"Number of lab events in ICU with {keyword}: {df_lab_icu_joined.shape[0]}\n")

        df_lab_icu_specific = df_lab_icu_joined[
        (df_lab_icu_joined['charttime'] >= df_lab_icu_joined['intime']) &
        (df_lab_icu_joined['charttime'] <= df_lab_icu_joined['outtime'])
        ]
        # print(df_lab_icu_specific.head(5))
        # print(f"Number of lab events in ICU with {keyword}: {df_lab_icu_specific.shape[0]}\n")

        # save df to csv
        # save_to_csv_multiple([df_patients, df_chart_events, df_output_events, df_lab_icu_specific],
        #                      [f'patients_{keyword}', f'chart_events_{keyword}',
        #                       f'output_events_{keyword}', f'lab_icu_specific_{keyword}'])

        return (df_patients, df_chart_events, df_output_events, df_lab_icu_specific)