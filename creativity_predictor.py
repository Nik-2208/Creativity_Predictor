import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

df = pd.read_csv(r"C:\Users\nikhi\OneDrive\Desktop\Creativity_Predictor\ai_dev_productivity.csv")
# df.info()
# df.describe
# print(df.columns)
# print(df.isna().sum())

df['coffee_adjusted'] = df['coffee_intake_mg'].apply(lambda x: x if x>0 else 50)  # Replace 0 coffee with neutral baseline
df['creative_burst'] = 0.5*df['hours_coding'] + 0.8*df['coffee_adjusted'] + 0.2*df['ai_usage_hours'] + np.random.randint(0,100,size=len(df))
X = df[['hours_coding', 'coffee_adjusted', 'sleep_hours',
        'commits', 'bugs_reported', 'ai_usage_hours',
        'cognitive_load', 'task_success']]
y = df['creative_burst']

# X = df['hours_coding','coffee_intake_mg','ai_usage_hours','commits']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

en = ElasticNet(alpha=1,l1_ratio=0.5,random_state=42)
en.fit(X_train_scaled,y_train)
# print(f"Label: {df.columns}, Coefficients: {en.coef_}")
# print("Intercept: ",en.intercept_)

# print("Best coef: ",en.coef_!=0)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Linear Regression")
print("MAE: ", mean_absolute_error(y_test, y_pred_lr))
print("MSE: ", mean_squared_error(y_test, y_pred_lr))
print("R2 Score: ", r2_score(y_test, y_pred_lr))
scores = cross_val_score(LinearRegression(), ss.transform(X), y, cv=5, scoring='r2')
print("LR Mean R²:", scores.mean())

rfg = RandomForestRegressor(n_estimators=10,random_state=42)
rfg.fit(X_train_scaled,y_train)
y_pred = rfg.predict(X_test_scaled)

print("Random Forest Regressor")
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))
scores = cross_val_score(RandomForestRegressor(n_estimators=10, random_state=42), ss.transform(X), y, cv=5, scoring='r2')
print("RFG Mean R²:", scores.mean())

# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel("True creative_burst")
# plt.ylabel("Predicted creative_burst")
# plt.title("RandomForest Predictions vs Truth")
# plt.show()

# print(df.head())

joblib.dump((lr, ss), "lr_model.pkl")
print("Model saved as lr_model.pkl")
