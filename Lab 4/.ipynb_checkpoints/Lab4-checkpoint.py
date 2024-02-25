#use decision tree on diabetes data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Explore the dataset
print(df.head())

#plot
sns.countplot(x='Outcome',data=df)
plt.show()


X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# Perform decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
adjusted = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
f1 = (r2/(1-r2))*((len(y)-X.shape[1]-1)/X.shape[1])


print("Decision Tree")
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Adjusted R-squared (Adj R2): {adjusted}')
print(f'F-statistic (F1): {f1}')
