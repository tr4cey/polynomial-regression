# Polynomial Regression from scratch in Python

This project is a demonstration of using Polynomial Regression on a simple data set and plotting the results.
For polynomial regression a matrix of theta values will be calculated like the following: 

$$\theta(x)  =  \begin{bmatrix}  
 1\\  
 x\\  
 x^2\\  
 ...\\  
 x^k\\ 
\end{bmatrix} \mathbb{R}^k$$

Using that theta matrix a label prediction can be calculated given a test input of $x$

$$h_{\theta}(x)  =  1  +  \theta_{1}x  +  \theta_{2}x^2  +  ...  +  \theta_{k}x^k$$

To check which value of $k$ is the most effective for this data set, the same experiment is run with the following values $k = [3, 5, 10, 25, 50]$ and the following plot is produced.

![polynomial regression plot image](/poly_plot.png)

Seeing the results of the plot show that higher values of $k$ fit the data set much better (like $k = 10$ or $k = 25$) but with the value $k = 50$ there is significant over fitting with the function.

Also looking at the plot of the data set it looks like a $sine$ or $cosine$ function would fit this data set well. So the above theta matrix and prediction function can be updated to the following:

$$\theta(x)  =  \begin{bmatrix}  
 1\\  
 x\\  
 x^2\\  
 ...\\  
 x^k\\  
 cos(x)\\ 
\end{bmatrix}  \mathbb{R}^k$$

$$h_{\theta}(x)  =  1 +  cos(x) +  \theta_{1}x  +  \theta_{2}x^2  +  ...  +  \theta_{k}x^k$$

Adding the cosine function improves the results slightly, especially for lower values of k. So the model will preform faster since a lower value of k is needed with this updated regression function.

![polynomial regression plot with cos function image](/cosine_plot.png)

## Dependencies
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## Running the project

Install the list of dependencies above and run the project in your terminal with:
`python polynomial_regression.py`
