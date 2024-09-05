from sentence_transformers import (
    SentenceTransformer
)

from preprocess import (
    patients_events, process_items, process_train, events_in_num_string, all_events_in_string,
    all_events_in_numeric
)

from finetune import (create_dataset_finetune, create_dataset_finetune_json, finetune_embedding_train, 
                      process_dataset)

from utils import read_csv_spark

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('deteriorations') \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config('spark.driver.maxResultSize', '2g') \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# MODEL = SentenceTransformer('all-mpnet-base-v2')
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

SEED = 42
BATCH_SIZE = 64
# PARTITIONS = [100000, 10000, 1000, 100, 10, 1]

# patients_events("cardiac arrest")

# process_items('../data/processed/patient_lab_output_chart.csv', SEED)

# process_train('../data/processed/patients_train_csv', '../data/processed/patient_lab_output_chart.csv')

# events_in_num_string('../data/processed/events_is_numeric')

# all_events_in_string('../data/processed/events_string_df', '../data/processed/events_concat.csv', spark)

# create_dataset_finetune('../data/processed/all_events_string_df', spark, SEED, '../data/embedding/input/train_json', '../data/embedding/input/test_json')

# finetune_embedding_train(SEED, MODEL, BATCH_SIZE, '../data/embedding/input/train_json', 
#                    '../data/embedding/input/eval_json', '../data/embedding/input/test_json')

# process_dataset('../data/embedding/input/train_json', '../data/embedding/input/test_json', SEED)

# create_dataset_finetune_json(
#     '../data/processed/all_events_string_df', spark, SEED, '../data/embedding/input/train_json', 
#     '../data/embedding/input/eval_json', '../data/embedding/input/test_json', 
#     10, 10, subset_repartition=True
# )

all_events_in_numeric('../data/processed/events_numerical_df', '../data/processed/events_concat.csv', spark)

read_csv_spark('../data/processed/all_events_numeric_df', spark)
read_csv_spark('../data/processed/all_events_string_df', spark)
