```cpp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Collection (Reading from CSV)
# Replace 'sensor_data.csv' with your actual file path
data = pd.read_csv(r'sensordata.csv')

# Step 2: Data Preprocessing
# Check for missing values and handle them if necessary
data.dropna(inplace=True)

# Step 3: Feature Selection
features = data[['Temperature', 'Humidity']]
target = data['SoilMoisture']

# Step 4: Model Selection (Linear Regression)
model = LinearRegression()

# Step 5: Training the Model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 7: Prediction
# Predict soil moisture for new data (replace this with your actual new data)
```
new_data = np.array([[25, 50], [30, 60]])  # New data: [[Temperature, Humidity], ...]
predictions = model.predict(new_data)
print(f"Predicted Soil Moisture: {predictions}")
