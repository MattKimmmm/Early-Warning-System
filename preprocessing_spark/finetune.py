from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import BatchAllTripletLoss, TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers
from sklearn.model_selection import train_test_split
from pyspark.sql import functions
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType
from datasets import load_dataset, Dataset

from utils import (read_csv_spark, write_csv_spark, write_pickle, write_as_json, df_analytics, print_memory_usage,
                   preprocess_triplets, rename_keys_triplet_error)
from models_embedding import TripletAccuracyEvaluator

import pprint, time, datetime, os, torch

# Given a table with string value-only items, extract the values to create a list and return it as a dataset
def create_dataset_finetune(all_events_string_path, spark, seed, train_path, test_path):
    since = time.time()
    print(f"create_dataset_finetune starting at {time.ctime()}")

    # load table
    all_events_string_df = read_csv_spark(all_events_string_path, spark).select(col('VALUE').alias('sentence'), col('ITEMID').alias('label'))
    df_analytics(all_events_string_df, 'all_events_string_df')

    # extract values (strings) and itemid to create input and label
    # values = all_events_string_df.select('VALUE').rdd.flatMap(lambda x: x).collect()
    # itemids = all_events_string_df.select('ITEMID').rdd.flatMap(lambda x: x).collect()
    # toLocalIterator
    # values = [row.VALUE for row in all_events_string_df.toLocalIterator()]
    # itemids = [row.ITEMID for row in all_events_string_df.toLocalIterator()]

    # split data
    # train, test = all_events_string_df.select(col('VALUE').alias('sentence'), col('ITEMID').alias('label')).randomSplit([0.8, 0.2], seed)
    # df_analytics(train, 'Train')
    # df_analytics(test, 'Test')

    # create train/test dataset
    # values_train, values_test, itemids_train, itemids_test = train_test_split(values, itemids, test_size=0.2, random_state=seed)
    # train = {'sentence': values_train, 'label': itemids_train}
    # test = {'sentence': values_test, 'label': itemids_test}
    # pprint.pprint(train)
    # pprint.pprint(test)

    # write_pickle(train, '../data/embedding/input/train.pickle')
    # write_pickle(test, '../data/embedding/input/test.pickle')
    # write_as_json(train, train_path)
    # write_as_json(test, test_path)

    print(f"create_dataset_finetune completed successfully. Time taken: {datetime.timedelta(seconds=(time.time() - since))}s.")

# input: all_events_string_df (aggregated event table)
# output: (anchor, positive, negative) saved in json format, to be directly fed into sentence transformer
def create_dataset_finetune_json(all_events_string_path, spark, seed, train_path, eval_path, test_path, 
                                 num_partition, num_items, subset_repartition=False):
    since = time.time()
    print(f"create_dataset_finetune_json starting at {time.ctime()}")

    # load table
    all_events_string_df = read_csv_spark(all_events_string_path, spark)
    # df_analytics(all_events_string_df, 'all_events_string_df')

    # print_memory_usage('all_events_string_df loaded')

    splits = [1 / num_partition] * num_partition
        
    subsets = all_events_string_df.randomSplit(splits, seed)

    triplet_schema = StructType([
        StructField("anchor", StringType(), True),
        StructField("positive", StringType(), True),
        StructField("negative", StringType(), True)
    ])

    final_triplets_df = spark.createDataFrame([], schema=triplet_schema)

    # print_memory_usage('created subsets/empty df')

    for i, subset in enumerate(subsets):

        # df_analytics(subset, f'subset {i+1}/{len(subsets)}', True)

        # print_memory_usage('loop starts')

        print(f"Processing subset {i+1}/{len(subsets)}")
        since_subset = time.time()

        if subset_repartition:
            subset = subset.repartition('ITEMID')

        # print_memory_usage('subset repartitioned')

        window_spec = Window.partitionBy('ITEMID').orderBy(functions.rand())
        ranked_subset = subset.withColumn('rank', row_number().over(window_spec))
        filtered_subset_a = ranked_subset.filter(col('rank') == 1).drop('rank')
        filtered_subset_b = ranked_subset.filter(col('rank') == 2).drop('rank')
        filtered_subset_c = ranked_subset.filter(col('rank') == 3).drop('rank')

        # df_analytics(filtered_subset_a, 'filtered_subset_a', True)
        # df_analytics(filtered_subset_b, 'filtered_subset_b', True)
        # df_analytics(filtered_subset_c, 'filtered_subset_c', True)

        # positive pairs
        positives = filtered_subset_a.alias('a').join(
            filtered_subset_b.alias('b'), 
            (col('a.ITEMID') == col('b.ITEMID')) & (col('a.VALUE') != ('b.VALUE')),
            'inner'
        ).select(
            col('a.VALUE').alias('anchor_pos'),
            col('b.VALUE').alias('positive'),
            col('a.ITEMID').alias('label')
        )

        # print_memory_usage('positives')
        # df_analytics(positives, 'positives', True)

        # leave only one positive sample per anchor (to reduce the size of resulting df)
        # window_spec_pos = Window.partitionBy('anchor_pos').orderBy('positive')
        # positives_with_rank = positives.withColumn('rank', row_number().over(window_spec_pos))
        # positives_final = positives_with_rank.filter(col('rank') == 1).drop('rank')
        # df_analytics(positives_final, 'positives_final')

        # print_memory_usage('positive - window')

        # negative pairs
        negatives = filtered_subset_a.alias('c').join(
            filtered_subset_c.alias('d'),
            (col('c.ITEMID') != col('d.ITEMID')),
            'inner'
        ).select(
            col('c.ITEMID'),
            col('c.VALUE').alias('anchor'),
            col('d.VALUE').alias('negative')
        )

        # df_analytics(negatives, 'negatives', True)
        # print_memory_usage('nagative')

        negatives_filtered = negatives.withColumn('rank', row_number().over(window_spec))
        negatives_filtered = negatives_filtered.filter(col('rank') <= num_items).drop('rank', 'ITEMID')

        # df_analytics(negatives_filtered, 'negatives_filtered', True)
        # print_memory_usage('nagative')

        # leave only one positive sample per anchor (to reduce the size of resulting df)
        # window_spec_neg = Window.partitionBy('anchor').orderBy('negative')
        # negatives_with_rank = negatives.withColumn('rank', row_number().over(window_spec_neg))
        # negatives_final = negatives_with_rank.filter(col('rank') == 1).drop('rank')
        # df_analytics(negatives_final, 'negatives_final')

        # print_memory_usage('negative - window')

        triplets_df = positives.join(
            negatives_filtered,
            positives.anchor_pos == negatives_filtered.anchor,
            'inner'
        ).select(
            col('anchor_pos').alias('anchor'), 
            col('positive'),
            col('negative')
        )
        # df_analytics(triplets_df, 'triplets_df', True)

        # print_memory_usage('triplets')

        final_triplets_df = final_triplets_df.union(triplets_df)

        # print_memory_usage('triplets - union')

        # num_partitions = triplets_df.rdd.getNumPartitions()
        # print(f"Number of partitions in final_triplets_df: {num_partitions}")

        print(f'subset {i+1}/{len(subsets)} completed in {datetime.timedelta(seconds=(time.time() - since_subset))}')

    num_partitions_final = final_triplets_df.rdd.getNumPartitions()
    print(f"Number of partitions in final_triplets_df: {num_partitions_final}")
    # partition_sizes = final_triplets_df.rdd.glom().map(len).collect()
    # partition_sizes = final_triplets_df.rdd.mapPartitions(lambda it: [len(list(it))]).collect()
    # print(f"Partition sizes: {partition_sizes}")
    df_analytics(final_triplets_df, 'final_triplets_df', True)

    # final_triplets_df = final_triplets_df.coalesce(100)
    final_triplets_df = final_triplets_df.repartition(100)
    
    train_df, eval_df, test_df = final_triplets_df.randomSplit([0.7, 0.15, 0.15], seed)

    write_as_json(train_df, train_path)
    write_as_json(eval_df, eval_path)
    write_as_json(test_df, test_path)

    print(f"create_dataset_finetune_json completed successfully. Time taken: {datetime.timedelta(seconds=(time.time() - since))}s.")

# takes in train_json and test_json and process them for embedding for TripletEvaluator
def process_dataset(train_path, test_path, seed):
    since = time.time()
    print(f"process_dataset starting at {time.ctime()}")

    train_org = load_dataset("json", data_dir=train_path)
    train_eval_split = train_org['train'].train_test_split(test_size=0.25, seed=seed)

    train_dataset = train_eval_split['train']
    eval_dataset = train_eval_split['test']
    test_dataset = load_dataset("json", data_dir=test_path)['train']

    since_for = time.time()
    for i in range(5):
        print(f'in: {datetime.timedelta(seconds=(time.time() - since_for))}')
        print(train_dataset[i])

    since_for = time.time()
    print("First 5 rows of the training dataset:")
    for i in range(len(train_dataset['sentence'])):
        print(f'in: {datetime.timedelta(seconds=(time.time() - since_for))}')
        print(train_dataset[i])

    print("\nFirst 5 rows of the evaluation dataset:")
    for i in range(5):
        print(eval_dataset[i])

    print("\nFirst 5 rows of the test dataset:")
    for i in range(5):
        print(test_dataset[i])

    print_memory_usage('Datasets Loaded')

    print("Train/Test datasets loaded. Starting further processing on eval/test datasets..")

    eval_dataset = preprocess_triplets(eval_dataset, since)
    test_dataset = preprocess_triplets(test_dataset, since)

    # train_dataset.save_to_disk('../data/embedding/input/train')
    # eval_dataset.save_to_disk('../data/embedding/input/eval')
    # test_dataset.save_to_disk('../data/embedding/input/test')

    print(f"process_dataset completed successfully. Time taken: {datetime.timedelta(seconds=(time.time() - since))}.")

# finetune the given model using the train/test set
def finetune_embedding_train(seed, model, batch_size, train_path, eval_path, test_path):
    since = time.time()
    print(f"finetune_embedding starting at {time.ctime()}")

    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print_memory_usage('Initial')

    print_memory_usage('Model Loaded')

    # load datasets
    # train_path = os.path.abspath("../data/embedding/input/train_json")
    # test_path = os.path.abspath("../data/embedding/input/test_json")
    # train_org = load_dataset("json", data_dir=train_path)
    # train_eval_split = train_org['train'].train_test_split(test_size=0.25, seed=seed)

    # train_dataset = train_eval_split['train']
    # eval_dataset = train_eval_split['test']
    # test_dataset = load_dataset("json", data_dir=test_path)['train']
    train_json = load_dataset('json', data_dir=train_path)['train']
    eval_json = load_dataset('json', data_dir=eval_path)['train']
    test_json = load_dataset('json', data_dir=test_path)['train']

    # train_json = train_json.select(range(100000))
    # eval_json = eval_json.select(range(100000))
    # test_json = test_json.select(range(100000))

    # train_json = train_json.map(rename_keys_triplet_error, remove_columns=['anchor'])
    # eval_json = eval_json.map(rename_keys_triplet_error, remove_columns=['anchor'])
    # test_json = test_json.map(rename_keys_triplet_error, remove_columns=['anchor'])

    print(train_json)
    print(eval_json)
    print(test_json)

    print_memory_usage('Datasets Loaded')
    
    # loss function
    loss = TripletLoss(model)

    # training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir='../models/embedding/mini/train_batch64',
        learning_rate=5e-5,
        warmup_ratio=0.1,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        # auto_find_batch_size=True,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False, # Set to True if you have a GPU that supports BF16
        # eval_accumulation_steps= # Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset, the whole predictions are accumulated on GPU/NPU/TPU before being moved to the CPU (faster but requires more memory).
        gradient_accumulation_steps=1, # Number of updates steps to accumulate the gradients for, before performing a backward/update pass. Can accelerate the process
        optim='adamw_torch',
        # batch_sampler=BatchSamplers.GROUP_BY_LABEL, # for ContrastiveTensionLoss that requires labels (and at least two rows per label)
        # eval_strategy='steps',
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch'
        # eval_steps=100,
        # logging_steps=100
    )

    # evaluator
    dev_evaluator = TripletEvaluator(
        anchors=eval_json['anchor'],
        positives=eval_json['positive'],
        negatives=eval_json['negative'],
        batch_size=batch_size,
        show_progress_bar=False,
        name='Dev Evaluation - TripletEvaluator'
    )

    # try:
    #     results = dev_evaluator(model)
    # except Exception as e:
    #     print(f"An error occurred during evaluation: {e}")

    print("Starting evaluation...")
    try:
        results = dev_evaluator(model)
        print("Evaluation results:", results)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
    print(f"Finished evaluation. Time taken: {datetime.timedelta(seconds=(time.time() - since))}")

    print('Starting training...')
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_json,
        eval_dataset=eval_json,
        loss=loss,
        evaluator=dev_evaluator
    )
    trainer.train()

    model.save_pretrained('../models/embedding/mini/train_batch64/model')

    print(f"finetune_embedding completed successfully. Time taken: {datetime.timedelta(seconds=(time.time() - since))}.")


