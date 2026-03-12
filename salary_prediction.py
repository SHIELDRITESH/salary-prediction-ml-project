import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib
data = pd.read_csv("salary_prediction_dataset.csv")

print(data)

print(data.info())
print(data.describe())

print(data.isnull().sum())

X = data[['Experience', 'TestScore', 'InterviewScore']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Salary:", y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

joblib.dump(model, "salary_model.pkl")

new_data = pd.DataFrame([[5, 8, 9]], columns=['Experience','TestScore','InterviewScore'])

prediction = model.predict(new_data)

print("Predicted Salary for new candidate:", prediction)