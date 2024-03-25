

from utils import unique_column_values, get_icd_codes, column_analytics
from preprocess import patients_events, aggregate_events
import time

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
# print(f"patients_event took {time.time() - since}")

since = time.time()
# Aggregate Events for given keyword
aggregate_events("cardiac arrest")
print(f"aggregate_events took {time.time() - since}")