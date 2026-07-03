import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv('advertising.csv')
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()  
model.fit(X_train, y_train) 

# Save the model to a file
pickle.dump(model,open('model.pkl', 'wb'))