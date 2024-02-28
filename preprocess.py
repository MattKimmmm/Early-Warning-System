import pandas as pd
import os

from utils import read_csv, get_icd_codes

df_icu_stays = read_csv('ICUSTAYS.csv')

# aggregate patient for a given keyword
def patients_aggregate(keyword):
    # get icd codes for the given keyword
    icd_codes = get_icd_codes(keyword)

    if icd_codes is not None:
        # get all patients with cardiac arrest
        patients = df_icu_stays['subject_id'][df_icu_stays['hadm_id'].isin(icd_codes)]
        print(f"Patients with cardiac arrest: {patients.head(5)}")
        return patients