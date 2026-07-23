import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
df=pd.read_csv("ndvi_lst_dataset.csv")
print(df.head())
print(df.describe())
df.info()
print(df.shape)
print(df.columns)
print(df.isna().sum())
df["humidity_percent"]=df["humidity_percent"].fillna(df["humidity_percent"].mean())
df["windspeed_mps"]=df["windspeed_mps"].fillna(df["windspeed_mps"].mean())
df["ndvi"]=df["ndvi"].fillna(df["ndvi"].mean())
df["builtup_area_percent"]=df["builtup_area_percent"].fillna(df["builtup_area_percent"].mean())
X=df[["humidity_percent", "windspeed_mps", "builtup_area_percent", "ndvi", "longitude", "latitude"]];
y=df["lst_celsius"]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
model1=RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
prediction1=model1.predict(X_test)
error1_rf=mean_squared_error(y_test, prediction1)
error2_rf=mean_absolute_error(y_test, prediction1)
error3_rf=math.sqrt(mean_squared_error(y_test, prediction1))
error4_rf=r2_score(y_test, prediction1)
print("for Random Forest Regressor : ")
print("Mean Squared Error:", error1_rf)
print("Mean Absolute Error:", error2_rf)
print("Root Mean Square Error:", error3_rf)
print("R2 Score:", error4_rf)
rf_importance=model1.feature_importances_
for feature, imp in zip(X.columns, rf_importance):
    print(f"{feature}: {imp}")
model2=AdaBoostRegressor(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
prediction2=model2.predict(X_test)
error1_gb=mean_squared_error(y_test, prediction2)
error2_gb=mean_absolute_error(y_test, prediction2)
error3_gb=math.sqrt(mean_squared_error(y_test, prediction2))
error4_gb=r2_score(y_test, prediction2)
print("for Adaboost regressor")
print("Mean Squared Error:", error1_gb)
print("Mean Absolute Error:", error2_gb)
print("Root Mean Square Error:", error3_gb)
print("R2 Score:", error4_gb)
adaboost_importance=model2.feature_importances_
for feature, imp in zip(X.columns, adaboost_importance):
    print(f"{feature}: {imp}")
model3=XGBRegressor(n_estimators=100, random_state=42)
model3.fit(X_train, y_train)
prediction3=model3.predict(X_test)
error_xgb=mean_squared_error(y_test, prediction3)
error2_xgb=mean_absolute_error(y_test, prediction3)
error3_xgb=math.sqrt(mean_squared_error(y_test, prediction3))
error4_xgb=r2_score(y_test, prediction3)
print("for XGBoost regressor")
print("Mean Squared Error:", error_xgb)
print("Mean Absolute Error:", error2_xgb)
print("Root Mean Square Error:", error3_xgb)
print("R2 Score:", error4_xgb)
labels=["Humidity", "Wind", "Built-up", "NDVI", "Longitude", "Latitude"]
rows=["mean_squared_error", "mean_absolute_error", "root_mean_square_error", "r2_score"]
columns=["Random Forest", "AdaBoost", "XGBoost"]
results_df=pd.DataFrame([[error1_rf, error1_gb, error_xgb], [error2_rf, error2_gb, error2_xgb], [error3_rf, error3_gb, error3_xgb], [error4_rf, error4_gb, error4_xgb]], index=rows, columns=columns)
print(results_df)
plt.figure(figsize=(18, 6)) 
plt.subplot(231)
plt.bar(labels, rf_importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance for random forests algorithm")
plt.subplot(232)
plt.bar(labels, adaboost_importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance for AdaBoost algorithm")
plt.subplot(234)
plt.scatter(y_test, prediction1, color='blue', label='Random Forest')
plt.xlabel("y_test")
plt.ylabel("prediction")
plt.title("y_test vs random forestprediction")
plt.legend()
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--"
)
plt.subplot(235)
plt.scatter(y_test, prediction2, color='red', label='AdaBoost')
plt.xlabel("y_test")
plt.ylabel("prediction")
plt.title("y_test vs adaboost prediction")
plt.legend()
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--"
)
plt.subplot(236)
plt.scatter(y_test, prediction3, color='green', label='XGBoost')
plt.xlabel("y_test")
plt.ylabel("prediction")
plt.title("y_test vs xgboost prediction")
plt.legend()
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--"
)
plt.tight_layout()
plt.show()