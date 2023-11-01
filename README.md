# ADAMOptimizationShowcase
ADAM is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. It is quite easy to implement, and while it comes at a slight increase in computational cost, it its extremely more efficent at optimization problems

For this demonstration, I want to showcase the superiority of ADAM when compared to typical gradient descent.

I want to optimize:

$$f(x,y) = asin(kxy) + bcos(hy)$$

Where my loss function is the Mean-Squared Error of the residuals:

$$L = \frac{1}{n} \sum_{i=1}^{n} (\hat{f}_i(x,y) - f_i(x,y))^2$$


