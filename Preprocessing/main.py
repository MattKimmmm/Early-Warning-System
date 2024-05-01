from utils import unique_column_values, get_icd_codes, column_analytics, unique_column, load_np
from preprocess import patients_events, aggregate_events_output_lab, patients_events_nagative, create_dataset
from preprocess import aggregate_events_output_lab_negative, aggregate_events_output_lab_bin, aggregate_events_output_lab_bin_negative
from preprocess import create_dataset_bin
import time

RANDOM_SEED = 42
BATCH_SIZE = 16

# Unique Column Values
# unique_column_values('ICUSTAYS.csv', 'first_careunit')

# get_icd_codes("cardiac arrest")

# Column Analytics
# column_analytics('ICUSTAYS.csv', 'LOS', 100)
# Mean: 4.452456617647059
# Median: 2.1114499999999996
# Standard Deviation: 6.19682833639683
# Minimum: 0.1059
# Maximum: 35.4065
# -> most patients stay in the ICU for 2 days or less (median: 2.11 days)

# Patients Aggregate
# since = time.time()
# patients_events("cardiac arrest")
# patients_events_nagative("cardiac arrest", 1355, RANDOM_SEED)
# patients_events("sepsis")
# patients_events_nagative("sepsis", 4781, RANDOM_SEED)
# print(f"patients_event took {time.time() - since}")

# Aggregate Events for given keyword
# since = time.time()
# aggregate_events_output_lab("cardiac arrest")
# aggregate_events_output_lab_negative("cardiac arrest")
# aggregate_events_output_lab("sepsis")
# aggregate_events_output_lab_negative("sepsis")

since = time.time()
aggregate_events_output_lab_bin("cardiac arrest", 10)
aggregate_events_output_lab_bin_negative("cardiac arrest", 10)
print(f"aggregate_events took {time.time() - since}")

since = time.time()
aggregate_events_output_lab_bin("cardiac arrest", 5)
aggregate_events_output_lab_bin_negative("cardiac arrest", 5)
print(f"aggregate_events took {time.time() - since}")

since = time.time()
aggregate_events_output_lab_bin("cardiac arrest", 2)
aggregate_events_output_lab_bin_negative("cardiac arrest", 2)
print(f"aggregate_events took {time.time() - since}")

# create dataloaders
# create_dataset("cardiac arrest", BATCH_SIZE, RANDOM_SEED)
# create_dataset("sepsis", BATCH_SIZE, RANDOM_SEED)

# create_dataset_bin("cardiac arrest", BATCH_SIZE, RANDOM_SEED, 10)
# create_dataset_bin("cardiac arrest", BATCH_SIZE, RANDOM_SEED, 5)
# create_dataset_bin("cardiac arrest", BATCH_SIZE, RANDOM_SEED, 2)