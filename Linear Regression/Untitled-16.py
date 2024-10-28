import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib, datetime
# Create a training dataset
df = pd.read_csv(r'C:\Users\aksudhak\Downloads\DSML\ML Model\trainingdata.csv')
X = df[['COUNT','WEIGHT']].values  #data features
y = df['PRIORITY'].values  #data target

# Split the data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(datetime.datetime.now())
# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print(datetime.datetime.now())
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Save the model to a file
joblib.dump(model, r'C:\Users\aksudhak\Downloads\DSML\ML Model\linear_model.joblib')

count_value = np.array([[30,100]])
new_predictions = model.predict(count_value)
new_predictions = np.clip(new_predictions, 1, 10)
print(f"Predicted PRIORITY for COUNT value : {new_predictions[0]:.2f}")