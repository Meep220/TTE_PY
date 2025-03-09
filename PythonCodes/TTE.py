import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Function to prepare data
def set_data(df, id_col, period_col, treatment_col, outcome_col, eligible_col):
    return df.copy()

# Function to train logistic regression model for switch weight
def set_switch_weight_model(df, numerator_cols, denominator_cols):
    X = df[denominator_cols]
    y = df[numerator_cols[0]]  # Assuming single column for numerator
    
    # Handle missing or non-numeric values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Scale data for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logit_model = LogisticRegression(max_iter=500)  # Increase max_iter
    logit_model.fit(X_scaled, y)
    
    df["switch_weight"] = logit_model.predict_proba(X_scaled)[:, 1]
    return df

# Function to train logistic regression model for censor weight
def set_censor_weight_model(df, censor_event_col, numerator_cols, denominator_cols):
    X = df[denominator_cols]
    y = df[censor_event_col]
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logit_model = LogisticRegression(max_iter=500)  # Increase max_iter
    logit_model.fit(X_scaled, y)
    
    df["censor_weight"] = logit_model.predict_proba(X_scaled)[:, 1]
    return df

# Function to calculate final weights
def calculate_weights(df):
    if "switch_weight" not in df.columns:
        df["switch_weight"] = 1  # Default weight if missing
    if "censor_weight" not in df.columns:
        df["censor_weight"] = 1  # Default weight if missing
        
    df["final_weight"] = df["switch_weight"] * df["censor_weight"]
    return df

# Function to fit Kaplan-Meier survival model
def set_outcome_model(df):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["period"], event_observed=df["outcome"])
    return kmf

# Function to predict survival
def predict_survival(kmf, time_points):
    return np.exp(-0.1 * time_points)  # Placeholder survival curve

# Set up directories
trial_pp_dir = Path("./trial_pp")
trial_ITT_dir = Path("./trial_ITT")
trial_pp_dir.mkdir(exist_ok=True)
trial_ITT_dir.mkdir(exist_ok=True)

# Load data
data_censored = pd.read_csv("./csv-files/data_censored.csv")

# Ensure all necessary columns exist
expected_columns = {"id", "period", "treatment", "outcome", "eligible", "age", "x1", "x2", "x3", "censored"}
missing_columns = expected_columns - set(data_censored.columns)
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Set Data
trial_pp = set_data(data_censored, "id", "period", "treatment", "outcome", "eligible")
trial_ITT = set_data(data_censored, "id", "period", "treatment", "outcome", "eligible")

# Apply Switch Weight Model
trial_pp = set_switch_weight_model(trial_pp, ["age"], ["age", "x1", "x3"])

# Apply Censor Weight Model
trial_pp = set_censor_weight_model(trial_pp, "censored", ["x2"], ["x2", "x1"])
trial_ITT = set_censor_weight_model(trial_ITT, "censored", ["x2"], ["x2", "x1"])

# Calculate Final Weights
trial_pp = calculate_weights(trial_pp)
trial_ITT = calculate_weights(trial_ITT)

# Fit Outcome Model
kmf_pp = set_outcome_model(trial_pp)
kmf_ITT = set_outcome_model(trial_ITT)

# Predict Survival
time_points = np.arange(0, 11)
survival_prob = predict_survival(kmf_ITT, time_points)
diff = survival_prob - 0.5  # Example survival difference

# Plot Survival Difference
plt.plot(time_points, diff, label="Survival Difference")
plt.fill_between(time_points, diff - 0.1, diff + 0.1, color='red', alpha=0.3)
plt.xlabel("Follow up")
plt.ylabel("Survival Difference")
plt.legend()
plt.show()
