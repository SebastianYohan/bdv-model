import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib

file_path = "df_fix_trimmed.xlsx"
df_bdv = pd.read_excel(file_path, sheet_name=0)

X_bdv = df_bdv[['moisture','temp','estimated_temp','interaction']]
y_bdv = df_bdv['bdv']

# model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf_model.fit(X_bdv, y_bdv)
joblib.dump(rf_model, "rf-model.pkl")

#saat deploy, jalankan:
#rf_model = joblib.load("rf-model.pkl")