DATA_PATH = "data"

# Features
CATEGORICAL_FEATURES = ["anaemia", "diabetes", "high_blood_pressure", "smoking", "sex"]
NUMERICAL_FEATURES = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]
LABEL = "DEATH_EVENT"
RANDOM_STATE = 42

# MLFlow
PRODUCTION_RUN_ID = "7d6795fa67e34bb9b1622499f64794ed"
PRODUCTION_EXPERIMENT_ID = "2"
MLFLOW_URI = "http://localhost:5000"
