import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'F:\Nairobi Office Price Ex.csv')


# Extract relevant columns
x = data['SIZE'].values
y = data['PRICE'].values

# Define the Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    # Number of observations
    n = len(y)
    
    # Predicted values
    y_pred = m * x + c
    
    # Calculate gradients
    m_grad = (-2/n) * np.sum(x * (y - y_pred))
    c_grad = (-2/n) * np.sum(y - y_pred)
    
    # Update weights
    m = m - learning_rate * m_grad
    c = c - learning_rate * c_grad
    
    return m, c

# Normalize the feature x and set initial values for m and c
x_normalized = (x - np.mean(x)) / np.std(x)
m, c = np.random.rand(), np.random.rand()
learning_rate = 0.0001
epochs = 10
errors = []

# Training loop for 10 epochs
for epoch in range(epochs):
    # Predict using current m and c
    y_pred = m * x_normalized + c
    # Calculate and record the error
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    
    # Update m and c using gradient descent
    m, c = gradient_descent(x_normalized, y, m, c, learning_rate)
    
    # Print error for the current epoch
    print(f"Epoch {epoch + 1}: Mean Squared Error = {error}")

# Plotting the line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, m * ((x - np.mean(x)) / np.std(x)) + c, color='red', label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft)")
plt.ylabel("Office Price")
plt.title("Linear Regression: Line of Best Fit with Normalized Data")
plt.legend()
plt.show()

# Predict office price for a 100 sq. ft office
x_100_normalized = (100 - np.mean(x)) / np.std(x)
predicted_price = m * x_100_normalized + c
print(f"Predicted office price for 100 sq. ft: {predicted_price}")
