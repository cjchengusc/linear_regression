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
========================================
iteration_count =  1
cost_function =  1884.39487012
theta =  [[5.4045850823284475e-05]
 [0.0035118078522766694]]
========================================
iteration_count =  2
cost_function =  1872.07941154
theta =  [[0.00010791781311203043]
 [0.007012082008200774]]
========================================
iteration_count =  3
cost_function =  1859.84471411
theta =  [[0.00016161645795809582]
 [0.010500860347271768]]
========================================
iteration_count =  4
cost_function =  1847.69024822
theta =  [[0.00021514235457773236]
 [0.013978180624583501]]
========================================
......
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

Learning rate alpha and number of total iterations can be set in `Main function` in the `linear_regression.py` file. 
```python
# Main function
plt.figure()
L = linear_random_number_generator(number_of_training_examples=100, number_of_features=2)
input_x, output_y = L.linear_random_number_generator_method()
G = gradient_descent_algorithm(x=input_x, y=output_y, alpha=0.000001, total_iteration=4000)
final_theta = G.gradient_descent(G)
P = plot_hypothesis_and_linear_random_number(x=input_x, y=output_y, theta=final_theta)
P.plot_hypothesis_and_linear_random_number_method()
plt.show()
```

## Reference
Andrew Ng, [Machine Learning](https://www.coursera.org/learn/machine-learning), Stanford University Coursera
