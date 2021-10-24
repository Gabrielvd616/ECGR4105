import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Problem 1
# Reads labelled training data
df = pd.read_csv(r'diabetes.csv')

x = df.values[:, :7]
y = df.values[:, 8]

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

# Performs logistic regression by instantiating LogisticRegression object
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# Generates confusion matrix to evaluate accuracy of model
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# Evaluates model using accuracy, precision, and recall evaluation metrics
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Precision:', metrics.precision_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))

# Plots the results of the binary classifier model
class_names = [0, 1]
tick_marks = np.arange(len(class_names))
plt.subplots()
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Generates heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues', fmt='g')
plt.title('Confusion matrix')
plt.xlabel('Predicted class')
plt.ylabel('Actual class')

# Problem 2
# Generates Gaussian naive bayes model by instantiating GaussianNB object
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

# Generates confusion matrix to evaluate accuracy of model
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# Evaluates model using accuracy, precision, and recall evaluation metrics
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Precision:', metrics.precision_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))

# Plots the results of the binary classifier model
tick_marks = np.arange(len(class_names))
plt.subplots()
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Generates heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues', fmt='g')
plt.title('Confusion matrix')
plt.xlabel('Predicted class')
plt.ylabel('Actual class')

# Problem 3
# Performs K-fold cross-validation of logistic regression for K = 5 folds
metrics = ['accuracy', 'precision', 'recall']
cv5 = KFold(n_splits=5, random_state=1, shuffle=True)
scores = cross_validate(lr, x, y, scoring=metrics, cv=cv5, n_jobs=-1)
print('Accuracy:', scores['test_accuracy'])
print('Precision:', scores['test_precision'])
print('Recall:', scores['test_recall'])

# Performs K-fold cross-validation of logistic regression for K = 10 folds
cv10 = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_validate(lr, x, y, scoring=metrics, cv=cv10, n_jobs=-1)
print('Accuracy:', scores['test_accuracy'])
print('Precision:', scores['test_precision'])
print('Recall:', scores['test_recall'])

# Problem 4
# Performs K-fold cross-validation of Gaussian naive bayes model for K = 5 folds
scores = cross_validate(gnb, x, y, scoring=metrics, cv=cv5, n_jobs=-1)
print('Accuracy:', scores['test_accuracy'])
print('Precision:', scores['test_precision'])
print('Recall:', scores['test_recall'])

# Performs K-fold cross-validation of Gaussian naive bayes model for K = 10 folds
scores = cross_validate(gnb, x, y, scoring=metrics, cv=cv10, n_jobs=-1)
print('Accuracy:', scores['test_accuracy'])
print('Precision:', scores['test_precision'])
print('Recall:', scores['test_recall'])
plt.show()
