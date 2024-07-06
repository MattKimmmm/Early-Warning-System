
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, rand, lit, min, max, sum, mean, count, avg
from pyspark.sql.window import Window

import time

from utils import read_csv_spark, write_csv_spark, df_analytics, analyze_df, train_test_stratified

spark = SparkSession.builder.appName('deteriorations').getOrCreate()

# aggregate patient/event tables for a given keyword (deterioration)
def patients_events(keyword):
    since = time.time()

    # read tables
    df_diagnoses = read_csv_spark('../data/DIAGNOSES_ICD.csv', spark)
    df_diagnoses_icd = read_csv_spark('../data/D_ICD_DIAGNOSES.csv', spark)
    df_patients = read_csv_spark('../data/PATIENTS.csv', spark)
    df_icustays = read_csv_spark('../data/ICUSTAYS.csv', spark)

    # icd code associated with the given deterioration keyword
    df_icd_codes = df_diagnoses_icd.filter(df_diagnoses_icd.SHORT_TITLE.contains(keyword) | df_diagnoses_icd.LONG_TITLE.contains(keyword))
    df_icd_codes_neg = df_diagnoses_icd.filter(~df_diagnoses_icd.SHORT_TITLE.contains(keyword) & ~df_diagnoses_icd.LONG_TITLE.contains(keyword))

    # get patients' diagnoses with extracted diagnoses (icd codes)
    df_dignoses_patients = df_diagnoses.join(df_icd_codes.drop('ROW_ID'), on='ICD9_CODE', how='inner')
    # df_analytics(df_dignoses_patients, "df_dignoses_patients")

    # aggregate icu stays given diagnosed patients (include stays with los <= 4)
    df_patients_stays = df_icustays.join(df_dignoses_patients.drop('HADM_ID', 'ROW_ID'), on='SUBJECT_ID', how='inner')
    window_spec = Window.partitionBy('SUBJECT_ID').orderBy(df_patients_stays.OUTTIME.desc())
    df_patients_last_visits = df_patients_stays.withColumn('row_number', row_number().over(window_spec)).filter((col('row_number') == 1) & (col('LOS') <= 4)).drop('row_number', 'ROW_ID', 'SEQ_NUM', 'SHORT_TITLE', 'LONG_TITLE', 'DBSOURCE', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'SECOND_WARDID').withColumn('IS_DIAGNOSED', lit(1))
    # df_analytics(df_patients_last_visits, "df_patients_last_visits")

    # get same number of negative samples
    df_diagnoses_patients_neg = df_diagnoses.join(df_icd_codes_neg.drop('ROW_ID'), on='ICD9_CODE', how='inner')
    df_patients_stays_neg = df_icustays.join(df_diagnoses_patients_neg.drop('HADM_ID', 'ROW_ID'), on='SUBJECT_ID', how='inner')
    window_spec_neg = Window.partitionBy('SUBJECT_ID').orderBy(df_patients_stays_neg.OUTTIME.desc())
    df_patients_last_visits_neg = df_patients_stays_neg.withColumn('row_number', row_number().over(window_spec_neg)).filter((col('row_number') == 1) & (col('LOS') <= 4)).drop('row_number', 'ROW_ID', 'SEQ_NUM', 'SHORT_TITLE', 'LONG_TITLE',  'DBSOURCE', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'SECOND_WARDID').withColumn('IS_DIAGNOSED', lit(0))
    df_diagnoses_patients_neg_sampled = df_patients_last_visits_neg.orderBy(rand()).limit(df_patients_last_visits.count())
    # df_analytics(df_diagnoses_patients_neg_sampled, "df_diagnoses_patients_neg_sampled")

    # concatenate positive and negative samples
    df_diagnoses_patients_concat = df_patients_last_visits.unionByName(df_diagnoses_patients_neg_sampled)
    df_analytics(df_diagnoses_patients_concat, "df_diagnoses_patients_concat")

    # aggregate events
    df_labevents = read_csv_spark('../data/LABEVENTS.csv', spark)
    df_outputevents = read_csv_spark('../data/OUTPUTEVENTS.csv', spark)
    df_chartevents = read_csv_spark('../data/CHARTEVENTS.csv', spark)

    # drop unnecessary columns from event tables
    df_labevents = df_labevents.drop('ROW_ID', 'HADM_ID').withColumn('EVENT_TYPE', lit('lab'))
    df_outputevents = df_outputevents.drop('ROW_ID', 'HADM_ID', 'ICUSTAY_ID', 'STORETIME', 'CGID', 'STOPPED', 'NEWBOTTLE', 'ISERROR').withColumn('EVENT_TYPE', lit('output')).withColumn('FLAG', lit('NULL')).withColumn('VALUENUM', lit(0))
    df_chartevents = df_chartevents.drop('ROW_ID', 'HADM_ID', 'ICUSTAY_ID', 'STORETIME', 'CGID', 'WARNING', 'ERROR', 'RESULTSTATUS', 'STOPPED').withColumn('EVENT_TYPE', lit('chart')).withColumn('FLAG', lit('NULL'))
    df_analytics(df_labevents, "df_labevents")
    df_analytics(df_outputevents, "df_outputevents")
    df_analytics(df_chartevents, "df_chartevents")

    events_concat = df_labevents.unionByName(df_outputevents).unionByName(df_chartevents)
    df_analytics(events_concat, 'events_concat')

    # join patient visits with event items
    patient_lab_output_chart = df_diagnoses_patients_concat.join(events_concat, on='SUBJECT_ID', how='inner')
    df_analytics(patient_lab_output_chart, 'patient_lab_output_chart')

    # save the resulting df
    # patient_lab_output_chart.write.parquet('../data/processed/patient_lab_output_chart', mode='overwrite', compression='snappy')
    write_csv_spark(patient_lab_output_chart, '../data/processed/patient_lab_output_chart.csv')
    print(f"patients_events() took {time.time() - since} seconds")
    print(f"patient_lab_output_chart was written to ../data/processed/patient_lab_output_chart.csv")

# preprocess given aggregated events
def process_items(file_path, seed):
    aggregated_events = read_csv_spark(file_path, spark)
    # aggregated_events = spark.read.parquet(file_path)
    # df_analytics(aggregated_events, "aggregated_events")
    # analyze_df(aggregated_events)

    # get rows associated with only training set for further preprocessing
    patients_train, patients_test = train_test_stratified(aggregated_events, seed)
    patients_train_rows = aggregated_events.join(patients_train, on=['SUBJECT_ID', 'IS_DIAGNOSED'], how='inner')
    # df_analytics(patients_train_rows, 'patients_train_rows')

    # group by unique item
    unique_item = patients_train_rows.groupBy('ITEMID').agg(count('VALUEUOM'),
                                                                  min('VALUE'))
    df_analytics(unique_item, 'unique_item')
    # df.groupBy('name').agg({'age': 'mean'}).collect()

    ### How to Normalize / quantize the item values?
    ### Deal with string valus (NLP, turn them into numerical vectors)