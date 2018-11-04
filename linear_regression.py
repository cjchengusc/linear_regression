#!/usr/bin/env python
#encoding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt

class gradient_descent_algorithm:
    def __init__(self,x,y,alpha,total_iteration):
        self.x = x    # x represents input features of each training examples. x is a nxm matrix. 
        self.y = y    # y represents outputs of each training examples. y is a real number. 
        self.alpha = alpha
        self.total_iteration = total_iteration

    def get_cost_function(self,theta):
        m = self.x.shape[1]    # m represents number of training examples. Amount of columns in matrix x is passed to m.
        squared_error = 0
        squared_error_sum = 0
        for mm in range(0,m):
            hypothesis = theta.transpose() * self.x[:,mm]
            squared_error = (hypothesis[0,0] - self.y[0,mm]) ** 2
            squared_error_sum += squared_error
        cost_function = squared_error_sum / (2 * m)
        return cost_function

    def gradient_descent(self):
        n = self.x.shape[0]    # n represents number of features. Amount of rows in matrix x is passed to n. 
        m = self.x.shape[1]    # m represents number of training examples. Amount of columns in matrix x is passed to m. 
        iteration_count = 0
        cost_function   = 0
        theta = np.matrix([[None]]*n)    # theta represents parameters of x of each training examples. theta is a nx1 matrix. 
        for nn in range(0,n):
            theta[nn,0] = 0
        cost_function_vs_iteration_count_plot = plt.subplot(212)
        cost_function_vs_iteration_count_plot.set_xlabel('Iterations of gradient descent')
        cost_function_vs_iteration_count_plot.set_ylabel(r'Cost function J($\Theta$)')
        for iteration in range(0,self.total_iteration):
            cost_function = self.get_cost_function(theta)
            partial_derivative_of_squared_error_sum = np.matrix([[None]]*n)
            for nn in range(0,n):
                partial_derivative_of_squared_error_sum[nn,0] = 0
            for mm in range(0,m):
                hypothesis = theta.transpose() * self.x[:,mm]
                cost = (hypothesis - self.y[0,mm])[0,0]
                partial_derivative_of_squared_error = self.x[:,mm] * cost * 2
                partial_derivative_of_squared_error_sum += partial_derivative_of_squared_error
            partial_derivative_of_cost_function = partial_derivative_of_squared_error_sum / (2 * m)
            theta = theta - partial_derivative_of_cost_function * self.alpha
            iteration_count += 1
            print '========================================'
            print 'iteration_count = ' , iteration_count
            print 'cost_function = '   , cost_function
            print 'theta = '           , theta
            cost_function_vs_iteration_count_plot.plot(iteration_count, cost_function, color='black', marker='o', markersize=2)
        print '========================================'
        print 'Learning rate = ', self.alpha
        print 'Final theta = ', theta
        print '========================================'
        return theta

class linear_random_number_generator:
    def __init__(self,number_of_training_examples,number_of_features):
        self.number_of_training_examples = number_of_training_examples
        self.number_of_features = number_of_features

    def linear_random_number_generator_method(self):
        x = np.matrix([[None]*self.number_of_training_examples]*self.number_of_features)
        y = np.matrix([[None]*self.number_of_training_examples])
        for training_example in range(0,self.number_of_training_examples):
            x[0,training_example] = 1
            x[1,training_example] = training_example
            y[0,training_example] = x[1,training_example] + random.uniform(0,1) * 10
        return x, y

class plot_hypothesis_and_linear_random_number:
    def __init__(self,x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

    def plot_hypothesis_and_linear_random_number_method(self):
        number_of_training_examples = self.x.shape[1]
        plot_generator = plt.subplot(211)
        plot_generator.set_xlabel('Input x of training examples')
        plot_generator.set_ylabel('Output y of training examples')
        for training_example in range(0,number_of_training_examples):
            plot_generator.plot(self.x[1,training_example], self.y[0,training_example], color='red', marker='o', markersize=6)
        theta_1 = self.theta[1,0]
        theta_0 = self.theta[0,0]
        x_axis = np.arange(0.,100.,0.2)
        hypothesis = theta_1 * x_axis + theta_0
        y_axis = hypothesis
        plot_generator.plot(x_axis, y_axis, color='black', linestyle='solid', linewidth=2)

# Main function
plt.figure()
L = linear_random_number_generator(number_of_training_examples=100, number_of_features=2)
input_x, output_y = L.linear_random_number_generator_method()
G = gradient_descent_algorithm(x=input_x, y=output_y, alpha=0.000001, total_iteration=4000)
final_theta = G.gradient_descent()
P = plot_hypothesis_and_linear_random_number(x=input_x, y=output_y, theta=final_theta)
P.plot_hypothesis_and_linear_random_number_method()
plt.show()
