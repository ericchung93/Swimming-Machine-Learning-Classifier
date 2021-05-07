'''Final Project for Eric Chung
CptS 437 Intro to Machine Learning
Support Vector Regression and Artificial Neural Network Models
To Predict Swimming Improvement Trajectory'''
# Support Vector Regression:

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

'''Neural Network Class Definition (To Be Used Later)'''
class NeuralNetwork(object):

    def __init__(self, n_input, n_hidden, n_output, learning_rate):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.learning_rate = learning_rate

        self.wih = np.random.normal(0.0, pow(self.n_hidden, -0.5), (self.n_hidden, self.n_input))
        self.who = np.random.normal(0.0, pow(self.n_output, -0.5), (self.n_output, self.n_hidden))

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.learning_rate * np.dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)
        self.wih += self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def activation_function(self, x):
        return scipy.special.expit(x)

'''SUPPORT VECTOR REGRESSION MODEL'''
# importing the dataset
dataset = pd.read_csv('SVR_train_test.csv')

X = np.array(dataset['age']).reshape(-1, 1)
y = np.array(dataset['time']).reshape(-1, 1)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
y_train = y_train.reshape(len(y_train), )
y_test = y_test.reshape(len(y_test), )

# first perform a linear regression for comparison
lm = LinearRegression()
lm.fit(X_train, y_train)

print("Intercept: {:,.3f}".format(lm.intercept_))
print("Coefficient: {:,.3f}".format(lm.coef_[0]))

mae = mean_absolute_error(y_test, lm.predict(X_test))
print("MAE = {:,.2f}".format(1 * mae))

# Plot outputs
plt.figure(figsize=(10, 7))
plt.scatter(x=dataset['age'], y=dataset['time'])
plt.plot(X_test, lm.predict(X_test), color='red')
plt.xlabel('Age of Athlete')
plt.ylabel('100 Freestyle Time (seconds)')
plt.title('Linear Regression Prediction')
plt.show()

# support vector regression function
def svr_results(X_test, y_test, fitted_svr_model):
    print("C: {}".format(fitted_svr_model.C))
    print("Epsilon: {}".format(fitted_svr_model.epsilon))

    print("Intercept: {:,.2f}".format(fitted_svr_model.intercept_[0]))
    print("Coefficient: {:,.2f}".format(fitted_svr_model.coef_[0]))

    mae = mean_absolute_error(y_test, fitted_svr_model.predict(X_test))
    print("MAE = {:,.2f}".format(1 * mae))

    perc_within_eps = 100 * np.sum(y_test - fitted_svr_model.predict(X_test) < eps) / len(y_test)
    print("Percentage within Epsilon = {:,.2f}%".format(perc_within_eps))

    # Plot outputs
    plt.figure(figsize=(10, 7))
    plt.scatter(x=dataset['age'], y=dataset['time'])
    plt.plot(X_test, fitted_svr_model.predict(X_test), color='red')
    plt.plot(X_test, fitted_svr_model.predict(X_test) + eps, color='black')
    plt.plot(X_test, fitted_svr_model.predict(X_test) - eps, color='black')
    plt.xlabel('Age of Swimmer')
    plt.ylabel('100 Freestyle Time (seconds)')
    plt.title('SVR Prediction')
    plt.show()

# Based on the linear regression, a linear SVR will work best
# No evidence shown that the relationship of age vs time is
# polynomial or exponential
eps = 3
svr = LinearSVR(epsilon=eps, C=10000)
svr.fit(X_train, y_train)
svr_results(X_test, y_test, svr)

'''COMPARISON TO ACTUAL DATA'''
input_nodes = 5
hidden_nodes = 1
output_nodes = 1
learning_rate = 0.1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

def train_network(neural_net, epochs):
    dataset2 = open('ANN_train.csv', 'r')
    datalist2 = dataset2.readlines()
    dataset2.close()
    print(datalist2)
    train_11 = []
    train_12 = []
    train_13 = []
    train_14 = []
    train_15 = []
    train_18 = []
    for i in range(epochs):
        print('Training epoch {}/{},'.format(i + 1, epochs))
        for record in datalist2:
            all_values = record.split(',')
            targets = float(all_values[5])
            print(targets)
            inputs = (np.asfarray(all_values[0:5]))
            print(inputs)
            neural_net.train(inputs, targets)
    print('complete')

def test_network(neural_net):
    print('Testing the neural network.')
    test_data_file = open('ANN_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    for i, record in enumerate(test_data_list):
        all_values = record.split(',')
        correct_label = all_values[5]
        inputs = np.asfarray(all_values[0:5])
        print('Actual: ' + str(correct_label))
        outputs = neural_net.query(inputs)
        print(outputs)
        label = np.argmax(outputs)
        print('Predicted: ' + str(label))
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print('testing complete')
    return scorecard

# Time to train!
train_network(nn, 1)
# Time to test!
test_results = np.asarray(test_network(nn))
# Printing Results
print("Neural Network is {}% accurate at predicting".format(test_results.sum() / float(test_results.size) * 100.0))


