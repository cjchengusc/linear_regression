# Machine Learning: Linear Regression

## Description
- Gradient descent algorithm is applied to linear regression.
- In gradient descent algorithm, parameters corresponding to each input features are simultaneously and repeatedly updated until they are convergent.
- Parameters converge when partial derivative of squared error cost function of each parameters reaches the minimum value.
- In other words, gradient descent algorithm is used to get parameters that minimize the cost function.
- Plotting *cost function - number of iterations* curve is a sufficient way to monitor functionality of gradient descent algorithm. Gradient descent algorithm works correctly when cost function decreases on every iteration. Also a reasonable learning rate in gradient descent algorithm is set based on performance of *cost function - number of iterations* curve.

## Execution
Simply run the `linear_regression.py` file using Python  
```
python linear_regression.py
```
The output will look like this
```
......
========================================
iteration_count =  3998
cost_function =  6.37708837651
theta =  [[0.020515526218523258]
 [1.069225545291617]]
========================================
iteration_count =  3999
cost_function =  6.37708716933
theta =  [[0.02051662488932839]
 [1.069225535547381]]
========================================
iteration_count =  4000
cost_function =  6.37708596216
theta =  [[0.020517723559517188]
 [1.0692255257807557]]
========================================
Learning rate =  1e-06
Final theta =  [[0.020517723559517188]
 [1.0692255257807557]]
========================================
```
![image](https://github.com/cjchengusc/linear_regression/blob/master/linear_regression_convergent.png)

