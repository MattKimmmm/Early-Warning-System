
from pyspark.sql.functions import col, row_number, rand, lit, min, max, sum, mean, count, avg, when, to_timestamp
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

import time, datetime

from utils import (read_csv_spark, write_csv_spark, df_analytics, analyze_df, train_test_stratified,
                   get_icd_codes)

# aggregate patient/event tables for a given keyword (deterioration)
def patients_events(keyword, spark):
    since = time.time()

    # read tables
    # df_diagnoses = read_csv_spark('../data/DIAGNOSES_ICD.csv', spark)
    # df_diagnoses_icd = read_csv_spark('../data/D_ICD_DIAGNOSES.csv', spark)
    # df_patients = read_csv_spark('../data/PATIENTS.csv', spark)
    # df_icustays = read_csv_spark('../data/ICUSTAYS.csv', spark)

    # icd code associated with the given deterioration keyword
    # df_icd_codes = df_diagnoses_icd.filter(df_diagnoses_icd.SHORT_TITLE.contains(keyword) | df_diagnoses_icd.LONG_TITLE.contains(keyword))
    # df_icd_codes_neg = df_diagnoses_icd.filter(~df_diagnoses_icd.SHORT_TITLE.contains(keyword) & ~df_diagnoses_icd.LONG_TITLE.contains(keyword))

    # get patients' diagnoses with extracted diagnoses (icd codes)
    # df_dignoses_patients = df_diagnoses.join(df_icd_codes.drop('ROW_ID'), on='ICD9_CODE', how='inner')
    # df_analytics(df_dignoses_patients, "df_dignoses_patients")

    # aggregate icu stays given diagnosed patients (include stays with los <= 4)
    # df_patients_stays = df_icustays.join(df_dignoses_patients.drop('HADM_ID', 'ROW_ID'), on='SUBJECT_ID', how='inner')
    # window_spec = Window.partitionBy('SUBJECT_ID').orderBy(df_patients_stays.OUTTIME.desc())
    # df_patients_last_visits = df_patients_stays.withColumn('row_number', row_number().over(window_spec)).filter((col('row_number') == 1) & (col('LOS') <= 4)).drop('row_number', 'ROW_ID', 'SEQ_NUM', 'SHORT_TITLE', 'LONG_TITLE', 'DBSOURCE', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'SECOND_WARDID').withColumn('IS_DIAGNOSED', lit(1))
    # df_analytics(df_patients_last_visits, "df_patients_last_visits")

    # get same number of negative samples
    # df_diagnoses_patients_neg = df_diagnoses.join(df_icd_codes_neg.drop('ROW_ID'), on='ICD9_CODE', how='inner')
    # df_patients_stays_neg = df_icustays.join(df_diagnoses_patients_neg.drop('HADM_ID', 'ROW_ID'), on='SUBJECT_ID', how='inner')
    # window_spec_neg = Window.partitionBy('SUBJECT_ID').orderBy(df_patients_stays_neg.OUTTIME.desc())
    # df_patients_last_visits_neg = df_patients_stays_neg.withColumn('row_number', row_number().over(window_spec_neg)).filter((col('row_number') == 1) & (col('LOS') <= 4)).drop('row_number', 'ROW_ID', 'SEQ_NUM', 'SHORT_TITLE', 'LONG_TITLE',  'DBSOURCE', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'SECOND_WARDID').withColumn('IS_DIAGNOSED', lit(0))
    # df_diagnoses_patients_neg_sampled = df_patients_last_visits_neg.orderBy(rand()).limit(df_patients_last_visits.count())
    # df_analytics(df_diagnoses_patients_neg_sampled, "df_diagnoses_patients_neg_sampled")

    # concatenate positive and negative samples
    # df_diagnoses_patients_concat = df_patients_last_visits.unionByName(df_diagnoses_patients_neg_sampled)
    # df_analytics(df_diagnoses_patients_concat, "df_diagnoses_patients_concat")

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
    # patient_lab_output_chart = df_diagnoses_patients_concat.join(events_concat, on='SUBJECT_ID', how='inner')
    # df_analytics(patient_lab_output_chart, 'patient_lab_output_chart')

    # save the resulting df
    # patient_lab_output_chart.write.parquet('../data/processed/patient_lab_output_chart', mode='overwrite', compression='snappy')
    # write_csv_spark(patient_lab_output_chart, '../data/processed/patient_lab_output_chart.csv')

    # save the events table as a whole
    write_csv_spark(events_concat, '../data/processed/events_concat.csv')
    print(f"patients_events() took {time.time() - since} seconds")
    print(f"patient_lab_output_chart was written to ../data/processed/patient_lab_output_chart.csv")

# create train/test datasets from loaded dataset, save them
def process_items(file_path, seed, spark):
    aggregated_events = read_csv_spark(file_path, spark)
    # aggregated_events = spark.read.parquet(file_path)
    # df_analytics(aggregated_events, "aggregated_events")
    # analyze_df(aggregated_events)

    # get rows associated with only training set for further preprocessing
    patients_train, patients_test = train_test_stratified(aggregated_events, seed)
    write_csv_spark(patients_train, "../data/processed/patients_train_csv")
    write_csv_spark(patients_test, '../data/processed/patients_test_csv')


# preprocess training data (quantize numerical values. vectorize discrete string values)
def process_train(patients_train_path, aggregated_events_path, spark):

    patients_train = read_csv_spark(patients_train_path, spark)
    aggregated_events = read_csv_spark(aggregated_events_path, spark)

    patients_train_rows = aggregated_events.join(patients_train, on=['SUBJECT_ID', 'IS_DIAGNOSED'], how='inner')
    # df_analytics(patients_train_rows, 'patients_train_rows')

    # group by unique item
    unique_item = patients_train_rows.groupBy('ITEMID').agg(count('VALUEUOM'),
                                                                  min('VALUE'))
    df_analytics(unique_item, 'unique_item')



    # df.groupBy('name').agg({'age': 'mean'}).collect()

    ### How to Normalize / quantize the item values?
    ### Deal with string valus (NLP, turn them into numerical vectors)

# Given full event dataset, do some analytics for preparing process steps
def full_event_analytics(events_concat_path, spark):
    events_concat = read_csv_spark(events_concat_path, spark)
    df_analytics(events_concat, 'events_concat')

    
# Given aggregated unique item df, create two new dfs with rows with values in numerical vs string
def events_in_num_string(unique_item_df_path, spark, keyword):
    unique_item_df = read_csv_spark(unique_item_df_path, spark)

    events_numerical_df = unique_item_df.filter(col('RATIO') >= 0.9)
    events_string_df = unique_item_df.filter(col('RATIO') < 0.1)

    df_analytics(events_numerical_df, 'events_numerical_df')
    df_analytics(events_string_df, 'events_string_df')

    total_count = unique_item_df.count()
    numerical_count = events_numerical_df.count()
    string_count = events_string_df.count()
    excluded_count = total_count - numerical_count - string_count

    print(f"From total of {total_count} unique items, {numerical_count} items were included as numerical and {string_count} items were included as string. {excluded_count} items were excluded due to its ambiguity in type representation.")
    print("Items with string values will be further processed to create a word embedding to assign numerical vectors to these values.")

    write_csv_spark(events_numerical_df, f'../data/processed/events_numerical_df')
    write_csv_spark(events_string_df, f'../data/processed/events_string_df')

# Join the whole event table with extracted ITEMIDs in events_string_df to prepare word embedding
def all_events_in_string(events_string_df_path, events_concat_df_path, spark):
    events_string_df = read_csv_spark(events_string_df_path, spark)
    events_concat_df = read_csv_spark(events_concat_df_path, spark)
    print("Schema of events_string_df:")
    events_string_df.printSchema()
    print("Schema of events_concat_df:")
    events_concat_df.printSchema()

    events_string_df = events_string_df.drop('NUM_NUMERIC', 'NUM_STRING', 'RATIO')

    print("Transforming VALUE column to string...")
    events_concat_df = events_concat_df.withColumn('VALUE', 
                                                   when(col('VALUE').isNull(), "null").
                                                   otherwise(col('VALUE').cast(StringType())))
    print("Schema of events_concat_df after transformation:")
    events_concat_df.printSchema()
    print("Sample data from events_concat_df after transformation:")
    events_concat_df.show(5)
    
    all_events_string = events_string_df.join(events_concat_df, on='ITEMID', how='left')
    print("DataFrames joined successfully.")
    
    df_analytics(all_events_string, 'all_events_string')

    write_csv_spark(all_events_string, '../data/processed/all_events_string_df')

# Join the whole event table with extracted ITEMIDs in events_numerical_df to prepare word embedding
def all_events_in_numeric(events_numerical_df_path, events_concat_df_path, spark):
    events_numeric_df = read_csv_spark(events_numerical_df_path, spark)
    events_concat_df = read_csv_spark(events_concat_df_path, spark)
    print("Schema of events_numeric_df:")
    events_numeric_df.printSchema()
    print("Schema of events_concat_df:")
    events_concat_df.printSchema()

    events_numeric_df = events_numeric_df.drop('NUM_NUMERIC', 'NUM_STRING', 'RATIO')

    # print("Transforming VALUE column to string...")
    # events_concat_df = events_concat_df.withColumn('VALUE', 
    #                                                when(col('VALUE').isNull(), "null").
    #                                                otherwise(col('VALUE').cast(StringType())))
    # print("Schema of events_concat_df after transformation:")
    # events_concat_df.printSchema()
    # print("Sample data from events_concat_df after transformation:")
    # events_concat_df.show(5)
    
    all_events_numeric = events_numeric_df.join(events_concat_df, on='ITEMID', how='left')
    print("DataFrames joined successfully.")
    
    df_analytics(all_events_numeric, 'all_events_numeric')

    write_csv_spark(all_events_numeric, '../data/processed/all_events_numeric_df')


# given deterioration keyword and all-event table, extract deterioration-specific items
def extract_deterioration(keyword, spark, seed):
    since = time.time()
    icd_codes = get_icd_codes(keyword, spark)

    events_concat_df = read_csv_spark('../data/processed/events_concat.csv', spark)
    df_analytics(events_concat_df, 'events_concat_df', count=True)

    # delete duplicate rows
    patients_subject_ids = patients_subject_ids.dropDuplicates()
    df_analytics(patients_subject_ids, 'patients_df - after dropping duplicates', True)

    diagnoses_icd_df = read_csv_spark('../data/DIAGNOSES_ICD.csv', spark)
    diagnoses_icd_subject_id = diagnoses_icd_df.select('SUBJECT_ID', 'ICD9_CODE')
    df_analytics(diagnoses_icd_subject_id, 'diagnoses_icd_subject_id', True)

    # get SUBJECT_ID's of patients diagnosed with the given deterioration
    icd_subject_df = icd_codes.join(diagnoses_icd_subject_id, on='ICD9_CODE', how='left').drop(
        'ROW_ID', 'HADM_ID', 'SEQ_NUM'
    )
    # negative examples
    icd_subject_df_neg = patients_subject_ids.join(icd_subject_df, on='SUBJECT_ID', how='left_anti')
    
    # df_analytics(icd_subject_df_neg, 'icd_subject_df_neg - first join', count=True)
    # icd_subject_df_neg = icd_subject_df_neg.dropDuplicates(['SUBJECT_ID'])
    # df_analytics(icd_subject_df_neg, 'icd_subject_df_neg - dropped duplicates', count=True)

    # keep the same number of patients for pos / neg samples
    sampling_fraction = icd_subject_df.count() / icd_subject_df_neg.count()
    icd_subject_df_neg = icd_subject_df_neg.sample(False, sampling_fraction, seed=seed)
    df_analytics(icd_subject_df_neg, 'icd_subject_df_neg - first join', True)

    icd_subject_df_neg = icd_subject_df_neg.join(diagnoses_icd_subject_id, on='SUBJECT_ID', how='left')
    df_analytics(icd_subject_df_neg, 'icd_subject_df_neg - second join', True)

    icd_subject_df_neg = icd_subject_df_neg.dropDuplicates(['SUBJECT_ID'])
    df_analytics(icd_subject_df_neg, 'icd_subject_df_neg - dropped duplicates', True)

    df_analytics(icd_subject_df, 'icd_subject_df', count=True)
    df_analytics(icd_subject_df_neg, 'icd_subject_df_neg - final', count=True)

    # icd_codes_neg = d_icd_diagnoses_df.select("ICD9_CODE").join(icd_codes, on='ICD9_CODE', how='left_anti')
    # sampling_fraction = icd_codes.count() / icd_codes_neg.count()
    # icd_codes_neg = icd_codes_neg.sample(False, sampling_fraction, seed=seed)

    # patients_deterioration_df = icd_subject_df.join(patients_df, on='SUBJECT_ID', how='left').drop(
    #     'DOD', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG'
    # )
    # patients_deterioration_neg_df = icd_subject_df_neg.join(patients_df, on='SUBJECT_ID', how='left').drop(
    #     'DOD', 'DOD_HOSP', 'DOD_SSN', 'EXPIRE_FLAG'
    # )
    # df_analytics(patients_deterioration_df, 'patients_deterioration_df', count=True)
    # df_analytics(patients_deterioration_neg_df, 'patients_deterioration_neg_df', count=True)

    extracted_events_df = icd_subject_df.join(events_concat_df, on='SUBJECT_ID', how='left')
    extracted_events_neg_df = icd_subject_df_neg.join(events_concat_df, on='SUBJECT_ID', how='left')
    df_analytics(extracted_events_df, 'extracted_events_df', count=True)
    df_analytics(extracted_events_neg_df, 'extracted_events_neg_df', count=True)

    write_csv_spark(extracted_events_df, f'../data/processed/{keyword}/extracted_events_df')
    write_csv_spark(extracted_events_neg_df, f'../data/processed/{keyword}/extracted_events_neg_df')

    print(f"extract_deterioration completed successfully. Time taken: {datetime.timedelta(seconds=(time.time() - since))}s.")

# Given events_numeric_df, events_string_df, and deterioration-specific df, 
def aggregate_events(events_numeric_df_path, events_string_df_path, keyword, spark):
    since = time.time()
    
    extracted_events_df_path = f'../data/processed/{keyword}/extracted_events_df'
    extracted_events_neg_df_path = f'../data/processed/{keyword}/extracted_events_neg_df'

    print(extracted_events_df_path)
    print(extracted_events_neg_df_path)
    
    events_numeric_df = read_csv_spark(events_numeric_df_path, spark).drop('NUM_NUMERIC', 'NUM_STRING', 'RATIO')
    events_string_df = read_csv_spark(events_string_df_path, spark).drop('NUM_NUMERIC', 'NUM_STRING', 'RATIO')
    extracted_events_df = read_csv_spark(extracted_events_df_path, spark)
    extracted_events_neg_df = read_csv_spark(extracted_events_neg_df_path, spark)

    # positive samples into numeric/string
    pos_numeric = events_numeric_df.join(extracted_events_df, on='ITEMID',how='left')
    pos_string = events_string_df.join(extracted_events_df, on='ITEMID', how='left')
    # negative samples
    neg_numeric = events_numeric_df.join(extracted_events_neg_df, on='ITEMID', how='left')
    neg_string = events_string_df.join(extracted_events_neg_df, on='ITEMID', how='left')

    # df_analytics(pos_numeric, 'pos_numeric', True)
    # df_analytics(pos_string, 'pos_string', True)
    # df_analytics(neg_numeric, 'neg_numeric', True)
    # df_analytics(neg_string, 'neg_string', True)

    write_csv_spark(pos_numeric, f'../data/processed/{keyword}/pos_numeric')
    write_csv_spark(pos_string, f'../data/processed/{keyword}/pos_string')
    write_csv_spark(neg_numeric, f'../data/processed/{keyword}/neg_numeric')
    write_csv_spark(neg_string, f'../data/processed/{keyword}/neg_string')

    print(f"aggregate_events completed successfully. Time taken: {datetime.timedelta(seconds=(time.time() - since))}s.")

# data cleaning for row event data for a specific deterioration
def clean_events(spark, keyword):
    # load PATIENTS and ICUSTAY tables
    # patients_df = read_csv_spark('../data/PATIENTS.csv', spark)
    # patients_subject_ids = patients_df.select('SUBJECT_ID')
    # df_analytics(patients_df, 'patients_df', True)

    icustay_df = read_csv_spark('../data/ICUSTAYS.csv', spark)
    icustay_times_df = icustay_df.select('SUBJECT_ID', 'INTIME', 'OUTTIME', 'LOS').filter('LOS <= 3')
    icustay_times_df = icustay_times_df.withColumn('INTIME', to_timestamp(col('INTIME')))
    icustay_times_df = icustay_times_df.withColumn('OUTTIME', to_timestamp(col('OUTTIME')))

    window_spec = Window.partitionBy('SUBJECT_ID').orderBy(col('OUTTIME').desc())
    icustay_times_df = icustay_times_df.withColumn('ROW_NUMBER', row_number().over(window_spec))
    icustay_times_df = icustay_times_df.filter(col('ROW_NUMBER') == 1).drop('ROW_NUMBER')

    df_analytics(icustay_times_df, 'icustay_times_df - processed', True)

    # load processed dataframes
    pos_numeric_df = read_csv_spark(f'../data/processed/{keyword}/pos_numeric', spark)
    pos_string_df = read_csv_spark(f'../data/processed/{keyword}/pos_string', spark)
    neg_numeric_df = read_csv_spark(f'../data/processed/{keyword}/neg_numeric', spark)
    pos_string_df = read_csv_spark(f'../data/processed/{keyword}/neg_string', spark)

    # create final datasets, devise the model structure (double LSTM?)
    # string/numerical values -> seperate/aggregated trainings

# Data preprocessing in one
def preprocess_data(keyword, spark):

    # extract keyword(deterioration)-specific items from all event
    

    write_csv_spark


