import os

# read a csv file
def read_csv_spark(file_path, spark, header=True, infer_schema=True):
    df = spark.read.csv(file_path, header=header, inferSchema=infer_schema)
    file_name = os.path.basename(file_path)
    print(f"{file_name}:")
    print(df.show())
    print(df.summary())
    print(f"{df.count()} X {len(df.columns)}\n")

    return df

# save spark df as csv file
def write_csv_spark(df, file_path, header=True):
    df.write.format('csv').save(file_path, header=header)

# simple analytics for a spark df
def df_analytics(df, file_name):
    print(f"For df {file_name}:")
    print(df.show())
    print(df.summary().show())
    print(f"{df.count()} X {len(df.columns)}\n")

# analyze df
def analyze_df(df):
    # groupby_patient = df.groupBy(['SUBJECT_ID', df.IS_DIAGNOSED]).count().collect()
    # print(groupby_patient)

    patient_count = df.select('SUBJECT_ID').distinct().count()
    print(f"paitient count: {patient_count}")

# create stratified train/test datasets
def train_test_stratified(df, seed):
    patient_row = df.dropDuplicates(['SUBJECT_ID']).select('SUBJECT_ID', 'IS_DIAGNOSED')
    # df_analytics(patient_row, 'patient_row')

    # stratified splits of SUBJECT_ID/IS_DIAGNOSED pair
    fraction = {0: 0.8, 1:0.8}
    patients_train_df = patient_row.sampleBy('IS_DIAGNOSED', fraction, seed)
    patients_test_df = patient_row.join(patients_train_df, on='SUBJECT_ID', how='left_anti')

    # df_analytics(patients_train_df, 'patients_train_df')
    # df_analytics(patients_test_df, 'patients_test_df')

    return patients_train_df, patients_test_df