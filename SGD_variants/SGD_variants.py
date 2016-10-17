import numpy as np


class SGDvariants(object):
	"""
	This class implements the sgd and learning rate schedules 
	"""
	
	def __init__(self, exp_base = 10, r = 2.5*1e8, c = 0.75, momentum_gamma = 0.9, adag_epsilon = 1, rms_gamma = 0.9, adam_beta1 = 0.9, adam_beta2 = 0.999, adam_epsilon = 1e-8):
		"""

		exp_base : This is required in exponential scheduling. This is the base to which iteration/time is raised to
					get the new learning rate

		r : This is a parameter for scaling in the time/iteration number in power scheduling of learning rate.

		c : This is the power that is raised to scaled iteration, in power scheduling of learning rate.

		momentum_gamma : This is the decay factor for accmulating the gradients over time. This id required in momentum
						schedule of sgd

		adag_epsilon : This is required in AdaGrad sgd, to prevent division by zerro

		rms_gamma : This is required in RMSprop sgd. This acts as a decay constant

		adam_beta1 : This is required in Adaptive Moment Estimation. This is known as the first moment.

		adam_beta2 : This is required in Adaptive Moment Estimation. This is known as the second moment.

		adam_epsilon : This is required in AdaGrad sgd, to prevent division by zerro

		"""
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
		# params["first_moment"] = params["first_moment"] / (1 - self.adam_beta1)
		params["second_moment"] = self.adam_beta2 * params["second_moment"] + (1 - self.adam_beta2) * np.square(gradient)
		# params["second_moment"] = params["second_moment"] / (1 - self.adam_beta2)
		gradient = np.true_divide(params["first_moment"], np.sqrt(params["second_moment"]) + self.adam_epsilon)
		weights = weights - lr * gradient

		return params, weights




		
		
