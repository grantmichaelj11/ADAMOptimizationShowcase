# ADAM Optimization Showcase
ADAM is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. It is quite easy to implement, and while it comes at a slight increase in computational cost, it its extremely more efficent at optimization problems

For this demonstration, I want to showcase the superiority of ADAM when compared to typical gradient descent.

I want to optimize:

$$f(x,y) = asin(kxy) + bcos(hy)$$

Where my loss function is the Mean-Squared Error of the residuals:

$$L = \frac{1}{n} \sum_{i=1}^{n} (\hat{f}_i(x,y) - f_i(x,y))^2$$

The gradient descent formula is:
$$f(\theta)_{i+1} = f(\theta) _{i} - \alpha \nabla L(\theta) _{i}$$
where $\alpha$ is our "training rate" ensures too large of steps are not taken.

For $f(x,y)$ the gradient is defined defined:

$$ \frac{\partial L}{\partial \alpha} = \frac{-2}{n} sin(kxy) * [f(x,y) - asin(kxy) - bcos(hy)]$$

$$ \frac{\partial L}{\partial \beta} = \frac{-2}{n}cos(hy) * [f(x,y) - asin(kxy) - bcos(hy)]$$

$$ \frac{\partial L}{\partial k} = \frac{-2}{n} * axy * cos(kxy) * [f(x,y) - asin(kxy) - bcos(hy)]$$

$$ \frac{\partial L}{\partial h} = \frac{2}{n} * by * sin(hy) * [f(x,y) - asin(kxy) - bcos(hy)]$$

ADAM Optimization differs as two new hyperparameters are introduced, $\beta _1$ and $\beta _2$ which are the exponential decay rates for the moment estimates. These parameters are used to update the fist and second biased movement respectively. The algorithm for ADAM is:

1: Solve the gradients of the loss function $$g_t = \nabla _\theta f_t (\theta _{t-1})$$
2: Update biased first moment estimate $$m_t = \beta _1 * m _{t-1} + (1-\beta _1) * g_t$$
3: Update biased second moment estimate $$v_t = \beta _2 * v _{t-1} + (1-\beta _2) * g_t^2$$
4: Compute bias-corrected first moment estimate $$\hat{m}_t = m_t / (1- \beta _1 ^t)$$
5: Compute bias-corrected second moment estimate $$\hat{v}_t = v_t / (1 - \beta _2 ^t) $$
6: Update parameters $$\theta_t = \theta _{t-1} - \alpha * \hat{m}_t / (\sqrt{\hat{v} _t} + \epsilon)$$
7: Update iteration step $$t = t+1$$
8: Repeat until convergence or iteration count is met

The code in this repository demonstrates how ADAM is able to optimize this test function faster, and demonstrates how its behavior differs drastically from gradient descent. For more information on ADAM Optimization, the interested reader is referred to: arXiv:1412.6980v9 
