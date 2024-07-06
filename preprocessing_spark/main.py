from preprocess import patients_events, process_items


SEED = 42

# patients_events("cardiac arrest")

process_items('../data/processed/patient_lab_output_chart.csv', SEED)