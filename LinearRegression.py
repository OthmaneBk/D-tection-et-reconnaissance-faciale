#Data Initialization
import numpy as np

x_train = np. array ([1000 , 1500 , 2000 , 2200 , 2500 , 4500 , 5000 ,
5500])
y_train = np. array ([100 , 120 , 150 , 180 , 210 , 400 , 450 , 480])

#Model Definition
def compute_model_output(x,w,b):
  return w*x+b

#Model Fitting
w=0.09 #w is Weight parameter
b=-11.914414009999999999999999999999999 #b is Weight parameter

# Compute predictions
predictions=compute_model_output(x_train,w,b)
print(f"Predictions: {predictions}")

#Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, marker='x', c='r', label='Data points')
plt.plot(x_train, predictions, c='b', label='Linear model')
plt.title('House Prices vs. Size')
plt.xlabel('Size (sq. ft.)')
plt.ylabel('Price (1000s of dollars)')
plt.legend()
plt.grid(True)
plt.show()

#Prediction
x_new=3000
predicted_price=compute_model_output(x_new,w,b)
print(f"Predicted price for a house with size {x_new} sq. ft.: ${predicted_price:.2f} thousand dollars")

#Cost Function
m=len(x_train)
def Cost_function(x_train,w,b,y_train):
  cost = 0
  for i in range(m):
    f_wb = w * x_train[i] + b
    cost +=(f_wb - y_train[i])**2
  cost = cost / (2 * m)
  return cost
cost = Cost_function(x_train, w, b, y_train)
print(f"The cost function using w={w} and b={b} is={cost:.3f}")