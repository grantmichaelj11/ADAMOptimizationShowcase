# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:18:50 2023

@author: Grant
"""

import numpy as np
import matplotlib.pyplot as plt


##################################### Define model we are trying to fit #######################################
def func(x, y, k, h, a, b):
    
    return a * np.sin(k*x*y) + b * np.cos(h*y)
###############################################################################################################


##################################### Define MSE Loss Function Gradients ######################################
def dLa(fx, x, y, k, h, a, b):
    
    return -2 * np.sin(k*x*y) * (fx - a*np.sin(k*x*y) - b*(np.cos(h*y)))/len(x)

def dLb(fx, x, y, k, h, a, b):
    
    return -2 * np.cos(h*y) * (fx - a*np.sin(k*x*y) - b*(np.cos(h*y)))/len(x)

def dLk(fx, x, y, k, h, a, b):
    
    return -2 * a * x * y * np.cos(k*x*y) * (fx - a*np.sin(k*x*y) - b*(np.cos(h*y)))/len(x)

def dLh(fx, x, y, k, h, a, b):
    
    return 2 * b * y * np.sin(h*y) * (fx - a*np.sin(k*x*y) - b*(np.cos(h*y)))/len(x)
##############################################################################################################


##################################### Optimization methods ####################################################
def gradient_descent_optimizaiton(fx, x, y, initial_guesses, loss=0.005, iterations=10000, tolerance=0.001):
    
    k0, h0, a0, b0 = initial_guesses['k0'], initial_guesses['h0'], initial_guesses['a0'], initial_guesses['b0']
    iteration = []
    error_list = []
    
    for i in range(iterations):
        
        grad_k = np.sum(dLk(fx, x, y, k0, h0, a0, b0))
        grad_h = np.sum(dLh(fx, x, y, k0, h0, a0, b0))
        grad_a = np.sum(dLa(fx, x, y, k0, h0, a0, b0))
        grad_b = np.sum(dLb(fx, x, y, k0, h0, a0, b0))
        
        k = k0 - grad_k*loss
        h = h0 - grad_h*loss
        a = a0 - grad_a*loss
        b = b0 - grad_b*loss
        
        fx_estimated = func(x, y, k, h, a, b)
        
        error = np.sum(np.abs(fx_estimated - fx)**2)
        
        k0 = k
        h0 = h
        a0 = a
        b0 = b
        
        iteration.append(i+1)
        error_list.append(error)
        
        if error <= tolerance:
            
            print("Converged solution found after: " + str(i+1) + " iterations. % Error: " + str(round(error*100,4)))
            return k, h, a, b, iteration, error_list
        
    print("Failed to location solution within " + str(i+1) + " iterations. Current % error: " + str(round(error*100,4)))
    return k, h, a, b, iteration, error_list

def ADAM_optimization(fx, x, y, initial_guesses, beta1=0.9, beta2=0.9, loss=0.005, iterations=10001, tolerance=0.001):
    
    #Initilization starts the same as a gradient descent - find the gradients
    k0, h0, a0, b0 = initial_guesses['k0'], initial_guesses['h0'], initial_guesses['a0'], initial_guesses['b0']
    
    initial_guesses = np.array([k0, h0, a0, b0])
    
    iteration = []
    error_list = []
    
    #Initialize 1st and 2nd order moments for moving averages
    gradients = np.zeros((iterations, 4))
    
    mt_history = np.zeros((iterations,4))
    vt_history = np.zeros((iterations,4))
    
    for i in range(iterations-1):
        
        t = i + 1
        
        grad_k = np.sum(dLk(fx, x, y, k0, h0, a0, b0))
        grad_h = np.sum(dLh(fx, x, y, k0, h0, a0, b0))
        grad_a = np.sum(dLa(fx, x, y, k0, h0, a0, b0))
        grad_b = np.sum(dLb(fx, x, y, k0, h0, a0, b0))
        
        gradients[t] = np.array([grad_k, grad_h, grad_a, grad_b])
        
        mt = (beta1 * mt_history[t-1] + (1-beta1) * gradients[t])
        mt_history[t] = (mt)
    
        vt = (beta2 * vt_history[t-1] + (1-beta2) * gradients[t]**2)
        vt_history[t] = (vt)
        
        mhat_t = mt / (1-beta1**t)
        
        vhat_t = vt / (1-beta2**t)
        
        k, h, a, b = initial_guesses - loss * mhat_t/(np.sqrt(vhat_t) + tolerance)
        
        fx_estimated = func(x, y, k, h, a, b)
        
        error = np.sum(np.abs(fx_estimated - fx)**2)
        
        k0, h0, a0, b0 = k, h, a, b
        
        initial_guesses = np.array([k0, h0, a0, b0])
        
        iteration.append(i+1)
        error_list.append(error)
        
        if error <= tolerance:
            
            print("Converged solution found after: " + str(i+1) + " iterations. % Error: " + str(round(error*100,4)))
            return k, h, a, b, iteration, error_list
        
    print("Failed to location solution within " + str(i+1) + " iterations. Current % error: " + str(round(error*100,4)))
    return k, h, a, b, iteration, error_list

##################################################################################################################

######################################### Initialization #########################################################
# Randomly Generate 1000 different datapoints for x and y coordinates
x = np.random.uniform(-5,5, 1000)
y = np.random.uniform(-5,5, 1000)


# We will create a known function for the sake of this exercise - we can change
# k, h, a, b to known values to see how well our gradient descent performs

k_solution = 0.05
h_solution = 1.5
a_solution = 1.7
b_solution = 1.247

fx = func(x, y, k_solution, h_solution, a_solution, b_solution)

#Initial Guesses similar to what scipy.curve_fit would default to

initial_guesses = {'k0': 0.01, 
                   'h0': 1,
                   'a0': 1,
                   'b0': 1} 

# Run a gradient descent with our intital guesses until either:
# 1) a tolerance is met 2) we run n iterations

#Gradient Descent Samples
k_estimated, h_estimated, a_estimated, b_estimated, grad_iter, grad_error = gradient_descent_optimizaiton(fx, x, y, initial_guesses, loss=0.01)
k_grad_loss2, h_grad_loss2, a_grad_loss2, b_grad_loss2, grad_iter2, grad_error2 = gradient_descent_optimizaiton(fx, x, y, initial_guesses, loss=0.005)
k_grad_loss3, h_grad_loss3, a_grad_loss3, b_grad_loss3, grad_iter3, grad_error3 = gradient_descent_optimizaiton(fx, x, y, initial_guesses, loss=0.001)

#ADAM Optimization Samples
k_adam, h_adam, a_adam, b_adam, adam_iter, adam_error = ADAM_optimization(fx, x, y, initial_guesses, loss=0.01)
k_adam2, h_adam2, a_adam2, b_adam2, adam_iter2, adam_error2 = ADAM_optimization(fx, x, y, initial_guesses, loss = 0.005)
k_adam3, h_adam3, a_adam3, b_adam3, adam_iter3, adam_error3 = ADAM_optimization(fx, x, y, initial_guesses, loss=0.001)


#Graph error loss
fig, ax = plt.subplots()

ax.loglog(grad_iter, grad_error, label='Gradient Descent L=1e-2', color='b')
ax.loglog(grad_iter2, grad_error2, label='Gradient Descent L=5e-3', color='b', linestyle='--')
ax.loglog(grad_iter3, grad_error3, label='Gradient Descent L=1e-3', color='b', linestyle=':')

ax.loglog(adam_iter, adam_error, label='ADAM L=1e-3', color=(0.5,0,0.5), linestyle='-')
ax.loglog(adam_iter2, adam_error2, label='ADAM L=5e-3', color=(0.5,0,0.5), linestyle='--')
ax.loglog(adam_iter3, adam_error3, label='ADAM L=1e-4', color=(0.5,0,0.5), linestyle=':')


ax.set_xlabel('Iterations')
ax.set_ylabel('Loss Function')
ax.set_title('Optimization of: f(x,y) = asin(kxy) + bcos(hy)')

ax.legend(fontsize='small')

plt.show()



