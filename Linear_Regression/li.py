import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('Advertising.csv')

# collection 2 column TV and Radio

# y = mx + b
# sales  = weight.radio +  bias
# Radio la bien doc lap
# Weight la he so bien doc lap
# Bias la gia tri de bu dap nhung sai so, tuong tu nhu Error

X = dataframe.values[:, 2]
Y = dataframe.values[:, 4]

# show gia tri 2 bang 
# plt.scatter(X, Y, marker ='o')
# plt.show()

# du doan 
def predict(new_radio, weight, bias):
    return weight*new_radio + bias

# tinh chi phi MSE
def cost_function(X, Y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error  += (Y[i] - (weight*X[i] + bias))**2
    return sum_error/n


# update weight, bias, Gradient descent 
# Lay weight va bias hien tai - (learning docs speed)

def update_weight(X, Y, weight, bias, learning_rate):
    n  = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2*X[i]*(Y[i] - (weight*X[i] + bias))
        bias_temp += -2*(Y[i] - (weight*X[i] + bias))
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias_temp/n)*learning_rate

    return weight, bias

# training function
def train(X, Y, weight, bias, learning_rate, iter):
    cost_his = []

    for i in range(iter):
        weight, bias = update_weight(X, Y, weight, bias, learning_rate)
        cost = cost_function(X, Y, weight, bias)
        cost_his.append(cost)

    return weight, bias, cost_his

# excute
weight, bias, cost = train(X, Y, 0.03, 0.0014, 0.001, 60)
print(weight)
print(bias)
print(cost)
print('gia tri du doan:')
print(predict(19, weight, bias))

solanlap = [i for i in range(60)]
plt.plot(solanlap, cost)
plt.show()