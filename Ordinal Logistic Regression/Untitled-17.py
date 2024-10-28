import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import pickle

# Create a training dataset
df = pd.read_csv(r'C:\Users\aksudhak\Downloads\DSML\ML Model\trainingdata.csv')
X = df[['COUNT', 'WEIGHT']]  # data features
y = df['PRIORITY']  # data target

# Split the data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the ordinal logistic regression model
model = OrderedModel(y_train, X_train, distr='logit')

# Train the model
res = model.fit(method='bfgs', maxiter=10000)  # You may need to adjust the optimization method and iterations

# Summary of the model
print(res.summary())

# Make predictions on the test set
y_pred_prob = res.predict(X_test)  # These are the probabilities for all classes

# The predicted class will be the one with the highest cumulative probability that is still less than 0.5
y_pred = (y_pred_prob.cumsum(axis=1) < 0.5).sum(axis=1) + 1

print(f"Predicted PRIORITY values on test set: {y_pred}")

# Save the model to a file
with open(r'C:\Users\aksudhak\Downloads\DSML\ML Model\ordinal_logistic_model.pkl', 'wb') as file:
    pickle.dump(res, file)

# Load the model and make a prediction on a new data point
with open(r'C:\Users\aksudhak\Downloads\DSML\ML Model\ordinal_logistic_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

count_value = np.array([[30, 100]])
new_prediction_prob = loaded_model.predict(count_value)  # These are the probabilities for all classes

# The predicted class will be the one with the highest cumulative probability that is still less than 0.5
new_prediction = (new_prediction_prob.cumsum(axis=1) < 0.5).sum(axis=1) + 1

print(f"Predicted PRIORITY for COUNT value: {new_prediction[0]}")