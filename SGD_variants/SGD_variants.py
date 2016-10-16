import numpy as np


class SGDvariants(object):
	
	def __init__(self, exp_base = 10, r = 2.5*1e8, c = 0.75, momentum_gamma = 0.9, adag_epsilon = 1, rms_gamma = 0.9, adam_beta1 = 0.9, adam_beta2 = 0.999, adam_epsilon = 1e-8):
		self.exp_base = exp_base
		self.r = r
		self.c = c
		self.momentum_gamma = momentum_gamma
		self.adag_epsilon = adag_epsilon
		self.rms_gamma = rms_gamma
		self.adam_beta1 = adam_beta1
		self.adam_beta2 = adam_beta2
		self.adam_epsilon = adam_epsilon

	def exponential_scheduling(self, lr, i):
		return lr / (self.exp_base**(i/self.r))

	def power_scheduling(self, lr, i):
		return lr / ((1.0 + (i/self.r)) ** self.c)

	def sgd(self, gradient, lr, weights, params):
		weights = weights - lr * (gradient)
		return params, weights

	def momentum_sgd(self, gradient, lr, weights, params):
		new_v = (params["prev_v"] * self.momentum_gamma) - (lr * gradient)
		weights = weights + new_v
		params["prev_v"] = new_v
		return params, weights

	

	def AdaGrad(self, new_gradient, lr, weights, params):
		params["gradient_sum"] += np.square(new_gradient)
		gradient = np.true_divide(new_gradient,np.sqrt(self.adag_epsilon + params["gradient_sum"]))
		weights = weights - lr * gradient
		
		return params, weights

	def RMSprop(self, new_gradient, lr, weights, params):
		params["decaying_grad_sum"] = self.rms_gamma * params["decaying_grad_sum"] + (1-self.rms_gamma) * np.square(new_gradient)
		gradient = np.true_divide(new_gradient, np.sqrt(self.adag_epsilon + params["decaying_grad_sum"]))
		weights = weights - lr * gradient
		
		return params, weights

	def Adam(self, gradient, lr, weights, params):
		params["first_moment"] = self.adam_beta1 * params["first_moment"] + (1-self.adam_beta1) * gradient
		# print params["first_moment"]
		# params["first_moment"] = params["first_moment"] / (1 - self.adam_beta1)
		# print params["first_moment"]
		params["second_moment"] = self.adam_beta2 * params["second_moment"] + (1 - self.adam_beta2) * np.square(gradient)
		# print params["second_moment"]
		# params["second_moment"] = params["second_moment"] / (1 - self.adam_beta2)
		# print params["second_moment"]
		gradient = np.true_divide(params["first_moment"], np.sqrt(params["second_moment"]) + self.adam_epsilon)
		weights = weights - lr * gradient

		return params, weights




		
		
