import os, pickle, psutil, torch, time, datetime
from datasets import Dataset
from pyspark.sql.functions import lower, col

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
    try:
        df.write.format('csv').save(file_path, header=header)
    except Exception as e:
        print(f'Error in write_csv_spark: {e}')

# read/write in pickle
def read_pickle(file_path):
    with open(file_path, 'rb') as handle:
        item = pickle.load(handle)
    return item

def write_pickle(item, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)

# write a df into json
def write_as_json(df, file_path):
    df.write.mode('overwrite').json(file_path)

# simple analytics for a spark df
def df_analytics(df, file_name, count=False):
    print(f"For df {file_name}:")
    df.printSchema()

    try:
        print(df.show(20))
    except Exception as e:
        print(f"Error in show(): {e}")
    # Summary
    # try:
    #     print(df.summary().show())
    # except Exception as e:
    #     print(f"Error in summary(): {e}")
    
    # Describe
    # try:
    #     print(df.describe().show())
    # except Exception as e:
    #     print(f"Error in describe(): {e}")
    if count:
        try:
            print(f"{df.count()} X {len(df.columns)}\n")
        except Exception as e:
            print(f"Error in count(): {e}")

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

# print current memory status in CPU/GPU
def print_memory_usage(stage):
    """Function to print memory usage at different stages"""
    print(f"\n--- {stage} ---")
    print(f"CPU Memory Usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
        print(torch.cuda.memory_summary(device=torch.cuda.current_device()))

# input (sentence, label) -> (anchor, positive, negative)
def preprocess_triplets(dataset, since):
    print('preprocess_triplets')
    print_memory_usage('before forcing load')

    # force dataset to load into memory
    dataset = dataset.map(lambda x: x)

    print_memory_usage('after forcing load')

    triplets = []

    since_for = time.time()

    for i in range(len(dataset['sentence'])):
        print(f'in: {datetime.timedelta(seconds=(time.time() - since_for))}')

        print_memory_usage('in the loop')
        print(dataset[i])
        anchor_sentence = dataset['sentence'][i]
        anchor_label = dataset['label'][i]

        print(f"Type of dataset['sentence'][i]: {type(dataset['sentence'][i])}")
        print(f"Length of dataset['sentence'][i]: {len(dataset['sentence'][i])}")
        print(f"Type of dataset['label'][i]: {type(dataset['label'][i])}")

        print("Checkpoint 1: Successfully assigned anchor_sentence and anchor_label")
        print(f'anchor_sentence: {anchor_sentence}')
        print(f'anchor_label: {anchor_label}')
        
        positive_sentence = None
        negative_sentence = None
        
        print_memory_usage('before the second loop')

        since_for_j = time.time()

        # Find a positive example
        for j in range(len(dataset['sentence'])):
            print(f'in: {datetime.timedelta(seconds=(time.time() - since_for_j))}')
            print(dataset['sentence'][j])
            print(dataset['label'][j])
            if i != j and dataset['label'][j] == anchor_label:
                positive_sentence = dataset['sentence'][j]
                break
        
        # Find a negative example
        for j in range(len(dataset['sentence'])):
            if dataset['label'][j] != anchor_label:
                negative_sentence = dataset['sentence'][j]
                break
        
        if positive_sentence and negative_sentence:
            triplets.append((anchor_sentence, positive_sentence, negative_sentence))

        if (i % 10 == 0):
            print(f"{i} / {len(dataset['sentence'])} sentences processed.")
            print(f"Time elapsed: {datetime.timedelta(seconds=(time.time() - since))}")

        print(f"First Iteration done in: {datetime.timedelta(seconds=(time.time() - since))}")
        break
    
    return Dataset.from_dict({"anchor": [t[0] for t in triplets],
                              "positive": [t[1] for t in triplets],
                              "negative": [t[2] for t in triplets]})

# for renaming keys in json datasets
def rename_keys_triplet_error(example):
    return {
        'sentence': example['anchor'],
        'positive': example['positive'],
        'negative': example['negative']
    }

# Function to read CSV and get ICD codes based on a keyword search
def get_icd_codes(keyword, spark, file='../data/D_ICD_DIAGNOSES.csv'):
    # Read the CSV file into a PySpark DataFrame
    df = spark.read.csv(file, header=True, inferSchema=True)
    
    # Filter the DataFrame where either SHORT_TITLE or LONG_TITLE contains the keyword (case-insensitive)
    icd_codes = df.filter(
        (lower(col("SHORT_TITLE")).contains(keyword.lower())) | 
        (lower(col("LONG_TITLE")).contains(keyword.lower()))
    ).select("ICD9_CODE")
    
    # Show the first 5 ICD codes
    df_analytics(icd_codes, f'icd_codes for {keyword}', count=True)
    
    return icd_codes