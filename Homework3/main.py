import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

'''
Computes gradient descent model parameters and cost for logistic regression

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
acc_history : 1 x iterations array of accuracy for each iteration
'''


def gradient_descent(x, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    acc_history = np.zeros(iterations)
    m = x.shape[0]
    x = np.hstack((np.ones((m, 1)), x.reshape(m, n)))

    for i in range(iterations):
        # Computes ML model prediction using column vector theta and x values with
        # matrix vector product
        predictions = np.divide(1, 1 + np.exp(-1 * x.dot(theta)))
        theta = theta - (alpha / m) * x.transpose().dot(predictions - y)

        # Computes value of cost function J for each iteration
        log1 = np.multiply(y, np.log(predictions))
        log2 = np.multiply(1 - y, np.log(1 - predictions))
        cost_history[i] = -1 / m * np.sum(log1 + log2)

        # Computes accuracy for each iteration
        acc_history[i] = (m - np.sum(np.abs(np.round(predictions) - y))) / m

    return theta, cost_history, acc_history


# Problem 1
# Loads breast labelled training data
breast = load_breast_cancer()

# Formats np array since breast object contains separate fields for data and labels
breast_data = breast.data
labels = np.reshape(breast.target, (breast_data.shape[0], 1))
df = pd.DataFrame(np.concatenate([breast_data, labels], axis=1))

n = breast_data.shape[1]
x = df.values[:, :n]
y = df.values[:, n]

# Performs MIN MAX scaling
mms = MinMaxScaler()
x = mms.fit_transform(x)

# Performs standardization
ss = StandardScaler()
x = ss.fit_transform(x)

# Performs 80% and 20% split of the labelled data into training and test sets
np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=np.random)

# Initializes theta, the learning rate alpha, and the number of iterations
theta = np.zeros(x.shape[1] + 1)
alpha = 0.07
it = 1500

# Trains ML model with all input variables and evaluates the model on the test set
theta, train_cost, train_acc = gradient_descent(x_train, y_train, theta, alpha, it)
test_cost, test_acc = gradient_descent(x_test, y_test, theta, alpha, it)[1:]

# Evaluates model using accuracy evaluation metric
print('Accuracy:', test_acc[0])

# Plots training and test cost history for the logistic regression
plt.figure(1)
plt.plot(np.linspace(1, it, it), train_cost, color='orange',
         label='Training cost')
plt.plot(np.linspace(1, it, it), test_cost, color='red',
         label='Test cost')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training and test cost history for all input variables of the logistic'
          ' regression')
plt.legend()

# Plots training and test accuracy history for the logistic regression
plt.figure(2)
plt.plot(np.linspace(1, it, it), train_acc, color='orange',
         label='Training accuracy')
plt.plot(np.linspace(1, it, it), test_acc, color='red',
         label='Test accuracy')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Training and test accuracy history for all input variables of the logistic'
          ' regression')
plt.legend()

# Problem 2
# Performs PCA on the data
pca = PCA()
pcs = pca.fit_transform(x)

# Performs 80% and 20% split of the labelled data into training and test sets
x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(pcs, y, train_size=0.8,
                                                            test_size=0.2,
                                                            random_state=np.random)

# Initializes evaluation metrics for logistic regression model PCA
k = pcs.shape[1]
accuracy = np.zeros(k)
precision = np.zeros(k)
recall = np.zeros(k)

# Iteratively trains and evaluates model for principal components
acc_max = 0
k_opt = 0
for i in range(k):
    # Performs logistic regression by instantiating LogisticRegression object
    lr = LogisticRegression()
    lr.fit(x_train_p[:, :i + 1], y_train_p)
    y_pred = lr.predict(x_test_p[:, :i + 1])

    # Evaluates model using accuracy, precision, and recall evaluation metrics
    accuracy[i] = metrics.accuracy_score(y_test_p, y_pred)
    precision[i] = metrics.precision_score(y_test_p, y_pred)
    recall[i] = metrics.recall_score(y_test_p, y_pred)

    if accuracy[i] > acc_max:
        acc_max = accuracy[i]
        k_opt = i + 1

# Displays optimal K and corresponding accuracy, precision, and recall
print('Optimal value of K:', k_opt)
print('Accuracy:', acc_max)
print('Precision:', precision[k_opt - 1])
print('Recall:', recall[k_opt - 1])

# Plots accuracy, precision, and recall for varying numbers of principal components
plt.figure(3)
plt.plot(np.linspace(1, k, k), accuracy, color='red',
         label='Accuracy')
plt.plot(np.linspace(1, k, k), precision, color='green',
         label='Precision')
plt.plot(np.linspace(1, k, k), recall, color='blue',
         label='Recall')
plt.rcParams['figure.figsize'] = (10, 6)
plt.grid()
plt.xlabel('K')
plt.ylabel('Metric value')
plt.title('Accuracy, precision, and recall for logistic regression with K principal'
          ' components')
plt.legend()

# Problem 3
# Performs LDA on the training data
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)

# Evaluates naive Bayes model using accuracy, precision, and recall evaluation metrics
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Precision:', metrics.precision_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))

# Problem 4
# Transforms input data using linear discriminant function from LDA
lds = lda.fit_transform(x, y)

# Performs 80% and 20% split of the labelled data into training and test sets
x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(lds, y, train_size=0.8,
                                                            test_size=0.2,
                                                            random_state=np.random)

# Performs logistic regression by instantiating LogisticRegression object
lr = LogisticRegression()
lr.fit(x_train_l, y_train_l)
y_pred = lr.predict(x_test_l)

# Evaluates model using accuracy, precision, and recall evaluation metrics
print('Accuracy:', metrics.accuracy_score(y_test_l, y_pred))
print('Precision:', metrics.precision_score(y_test_l, y_pred))
print('Recall:', metrics.recall_score(y_test_l, y_pred))
plt.show()
