import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(42) 
num_samples = 1000
temperature = np.random.rand(num_samples) * 30 + 20  
humidity = np.random.rand(num_samples) * 100 
other_sensor_data = np.random.rand(num_samples, 3)  
features = np.column_stack((temperature, humidity, other_sensor_data))
X_train, X_test, y_train, y_test = train_test_split(features, temperature, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
new_sensor_readings = np.array([[25, 60, 0.5, 0.2, 0.8]]) 
predicted_temperature = model.predict(new_sensor_readings)
print(f"Predicted Temperature: {predicted_temperature[0]}")




