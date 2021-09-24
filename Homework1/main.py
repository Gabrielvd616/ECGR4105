import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

'''
Computes gradient descent model parameters and cost for linear regression

Input parameters:
x : m x n np array of training samples where m is the number of training samples and n
    is the number of model parameters
y : m x 1 np array of training labels
theta : 1 x n np array of model parameters
alpha : scalar value for learning rate
iterations : scalar value for the number of iterations

Output parameters:
theta : 1 x n np array of final model parameters
cost_history : 1 x iterations array of cost values for each iteration
'''


def gradient_descent(x, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        # Computes ML model prediction using column vector theta and x values using
        # matrix vector product
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * x.transpose().dot(errors)
        theta = theta - sum_delta

        # Computes value of cost function J for each iteration
        sqr_error = np.square(errors)
        cost_history[i] = 1 / (2 * m) * np.sum(sqr_error)

    return theta, cost_history


'''
Computes gradient descent model parameters and cost for linear regression using
regularization method of parameter penalization

Input parameters:
x : m x n np array of training samples where m is the number of training samples and n
    is the number of model parameters
y : m x 1 np array of training labels
theta : 1 x n np array of model parameters
alpha : scalar value for learning rate
iterations : scalar value for the number of iterations
reg_param : scalar value for the regularization parameter

Output parameters:
theta : 1 x n np array of final model parameters
cost_history : 1 x iterations array of cost values for each iteration
'''


def gradient_descent_reg(x, y, theta, alpha, iterations, reg_param):
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        # Computes ML model prediction using column vector theta and x values using
        # matrix vector product
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * (x.transpose().dot(errors) + reg_param * theta)
        theta = theta - sum_delta

        # Computes value of cost function J for each iteration
        sqr_error = np.square(errors)
        cost_history[i] = 1 / (2 * m) * (np.sum(sqr_error) + reg_param *
                                         np.sum(np.square(theta)))

    return theta, cost_history


# Defines the map function to map strings to numbers in table
def binary_map(x):
    return x.map({'yes': 1, 'no': 0})


# Problem 1, Part a
# Initializes the number of iterations and the learning rate alpha
iterations = 1500
alpha = 0.07

# Reads labelled training data
df = pd.read_csv(r'Housing.csv')

# List of variables to map
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
           'prefarea']

# Applies binary_map function to df
df[varlist] = df[varlist].apply(binary_map)

# Splits the data into training and test sets
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size=0.7, test_size=0.3,
                                     random_state=np.random)

# Formats training set
# x1 : area
# x2 : bedrooms
# x3 : bathrooms
# x4 : stories
# x10: parking
num_vars = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
m = len(df_train)
y = df_train.values[:, 0]
x = df_train[num_vars].values[:, 2:6]  # x1 was observed to cause loss to diverge
x = np.hstack((np.ones((m, 1)), x.reshape(m, 4)))

# Retrains ML model with x2, x3, x4, x10 (x1 not usable)
theta = np.zeros(5)
theta, train_cost = gradient_descent(x, y, theta, alpha, iterations)

# Formats test set
m = len(df_test)
y_0 = df_test.values[:, 0]
x_0 = df_test[num_vars].values[:, 2:6]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 4)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x2, x3, x4, x10 (x1 not usable)
plt.figure(1)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history for $x_{2}$, $x_{3}$, $x_{4}$,'
          ' and $x_{10}$')
plt.legend()

# Problem 1, Part b
# Formats training set
# x1 : area             x7 : basement
# x2 : bedrooms         x8 : hotwaterheating
# x3 : bathrooms        x9 : airconditioning
# x4 : stories          x10: parking
# x5 : mainroad         x11: prefarea
# x6 : guestroom
m = len(df_train)
x = df_train.values[:, 2:12]
x = np.hstack((np.ones((m, 1)), x.reshape(m, 10)))

# Retrains ML model with x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 (x1 not usable)
theta = np.zeros(11)
theta, train_cost = gradient_descent(x, y, theta, alpha, iterations)

# Formats test set
m = len(df_test)
x_0 = df_test.values[:, 2:12]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 10)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x2, x3, x4, x5, x6, x7, x8, x9, x10, x11
# (x1 not usable)
plt.figure(2)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history for all input variables except $x_{1}$')
plt.legend()

# Problem 2, Part a
# Defines MIN MAX scaler
scaler1 = MinMaxScaler()
df_train_norm = df_train[num_vars].values[:, :6]
df_test_norm = df_test[num_vars].values[:, :6]
df_train_norm = scaler1.fit_transform(df_train_norm)
df_test_norm = scaler1.fit_transform(df_test_norm)

# Formats training set
m = len(df_train_norm)
y = df_train_norm[:, 0]
x = df_train_norm[:, 1:6]
x = np.hstack((np.ones((m, 1)), x.reshape(m, 5)))

# Retrains ML model with x1, x2, x3, x4, x10
theta = np.zeros(6)
theta, train_cost = gradient_descent(x, y, theta, alpha, iterations)

# Formats test set
m = len(df_test_norm)
y_0 = df_test_norm[:, 0]
x_0 = df_test_norm[:, 1:6]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 5)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x1, x2, x3, x4, x10
plt.figure(3)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history with input normalization for $x_{1}$,'
          ' $x_{2}$, $x_{3}$, $x_{4}$, and $x_{10}$')
plt.legend()

# Defines standardization scaler
scaler2 = StandardScaler()
df_train_stand = df_train[num_vars].values[:, :6]
df_test_stand = df_test[num_vars].values[:, :6]
df_train_stand = scaler2.fit_transform(df_train_stand)
df_test_stand = scaler2.fit_transform(df_test_stand)

# Formats training set
m = len(df_train_stand)
y = df_train_stand[:, 0]
x = df_train_stand[:, 1:6]
x = np.hstack((np.ones((m, 1)), x.reshape(m, 5)))

# Retrains ML model with x1, x2, x3, x4, x10
theta = np.zeros(6)
theta, train_cost = gradient_descent(x, y, theta, alpha, iterations)

# Formats test set
m = len(df_test_stand)
y_0 = df_test_stand[:, 0]
x_0 = df_test_stand[:, 1:6]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 5)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x1, x2, x3, x4, x10
plt.figure(4)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history with input standardization for $x_{1}$,'
          ' $x_{2}$, $x_{3}$, $x_{4}$, and $x_{10}$')
plt.legend()

# Problem 2, Part b
# Defines MIN MAX scaler
df_train_norm = df_train.values[:, :12]
df_test_norm = df_test.values[:, :12]
df_train_norm = scaler1.fit_transform(df_train_norm)
df_test_norm = scaler1.fit_transform(df_test_norm)

# Formats training set
m = len(df_train_norm)
y = df_train_norm[:, 0]
x = df_train_norm[:, 1:12]
x = np.hstack((np.ones((m, 1)), x.reshape(m, 11)))

# Retrains ML model with x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11
theta = np.zeros(12)
theta, train_cost = gradient_descent(x, y, theta, alpha, iterations)

# Formats test set
m = len(df_test_norm)
y_0 = df_test_norm[:, 0]
x_0 = df_test_norm[:, 1:12]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 11)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
# x11
plt.figure(5)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history with input normalization for all input'
          ' variables')
plt.legend()

# Defines standardization scaler
df_train_stand = df_train.values[:, :12]
df_test_stand = df_test.values[:, :12]
df_train_stand = scaler2.fit_transform(df_train_stand)
df_test_stand = scaler2.fit_transform(df_test_stand)

# Formats training set
m = len(df_train_stand)
y = df_train_stand[:, 0]
x = df_train_stand[:, 1:12]
x = np.hstack((np.ones((m, 1)), x.reshape(m, 11)))

# Retrains ML model with x1, x2, x3, x4, x10
theta = np.zeros(12)
theta, train_cost = gradient_descent(x, y, theta, alpha, iterations)

# Formats test set
m = len(df_test_stand)
y_0 = df_test_stand[:, 0]
x_0 = df_test_stand[:, 1:12]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 11)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
# x11
plt.figure(6)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history with input standardization for all input'
          ' variables')
plt.legend()

# Problem 3, Part a
# Initializes lambda to 0.001
reg_param = 0.01

# Uses MIN MAX scaler since it results in the least loss
df_train_norm = df_train[num_vars].values[:, :6]
df_test_norm = df_test[num_vars].values[:, :6]
df_train_norm = scaler1.fit_transform(df_train_norm)
df_test_norm = scaler1.fit_transform(df_test_norm)

# Formats training set
m = len(df_train_norm)
y = df_train_norm[:, 0]
x = df_train_norm[:, 1:6]
x = np.hstack((np.ones((m, 1)), x.reshape(m, 5)))

# Retrains regularized ML model with x1, x2, x3, x4, x10
theta = np.zeros(6)
theta, train_cost = gradient_descent_reg(x, y, theta, alpha, iterations, reg_param)

# Formats test set
m = len(df_test_norm)
y_0 = df_test_norm[:, 0]
x_0 = df_test_norm[:, 1:6]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 5)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x1, x2, x3, x4, x10
plt.figure(7)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history with input normalization and regularization'
          ' for $x_{1}$, $x_{2}$, $x_{3}$, $x_{4}$, and $x_{10}$')
plt.legend()

# Problem 3, part b
# Uses MIN MAX scaler since it results in the least loss
df_train_norm = df_train.values[:, :12]
df_test_norm = df_test.values[:, :12]
df_train_norm = scaler1.fit_transform(df_train_norm)
df_test_norm = scaler1.fit_transform(df_test_norm)

# Formats training set
m = len(df_train_norm)
y = df_train_norm[:, 0]
x = df_train_norm[:, 1:12]
x = np.hstack((np.ones((m, 1)), x.reshape(m, 11)))

# Retrains regularized ML model with x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11
theta = np.zeros(12)
theta, train_cost = gradient_descent_reg(x, y, theta, alpha, iterations, reg_param)

# Formats test set
m = len(df_test_norm)
y_0 = df_test_norm[:, 0]
x_0 = df_test_norm[:, 1:12]
x_0 = np.hstack((np.ones((m, 1)), x_0.reshape(m, 11)))

# Computes test cost
test_cost = gradient_descent(x_0, y_0, theta, alpha, iterations)[1]
print('Final value of theta =', theta)

# Plots training and test cost history for x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
# x11
plt.figure(8)
plt.plot(np.linspace(1, iterations, iterations), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, iterations, iterations), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history with input normalization and regularization'
          ' for all input variables')
plt.legend()
plt.show()
