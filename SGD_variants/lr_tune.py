import numpy as np


class SGD(object):
	"""
	Exponential Scheduling
	Power Scheduling
	Performance Scheduling
	AdaGrad
	AdaDec
	RProp

	"""
	def __init__(self, mini_batch_size = 200, momentum = 0.9, c = 0.75, K = 1, r = 2.5*1e8, exp_base = 10.0, gamma = 0.999, tau = 5):
		self.mini_batch_size = mini_batch_size
		self.momentum = momentum
		self.c = c
		self.exp_base = exp_base
		self.r = float(r)
		self.gamma = gamma
		self.tau = tau

	def _get_discounted_grad_sum(grad_array, i):
		if i == 0:
			return grad_array[i]
		else:
			return self.gamma*(self._get_discounted_grad_sum(grad_array,i-self.tau)) 
									+ np.sum(grad_array[i-self.tau:])/self.tau

	def sgd_momentum(params, gradient, lr, v_prev):
		"""
		momentum tern increases the dimensions of the gradient points in same direction
		gamma = 0.9
		"""

		v_new = 0.9 * v_prev + lr * gradient
		params = params - v_new
		return params, v_new

	def sgd_acc_momentum(params, gradient, lr, v_prev):
		


	def exponential_scheduling(self, lr, i):
		return lr / (exp_base**(i/self.r))

	def power_scheduling(self, lr, i):
		return lr / ((1.0 + (i/self.r)) ** self.c)

	def AdaGrad(self, lr, grad_sum):
		return (lr / (np.sqrt(self.K + grad_sum)))

	def AdaDec(self, lr, grad_array, i):
		discounted_grad_sum = _get_discounted_grad_sum(grad_array, i)
		return self.power_scheduling(lr, i) / (np.sqrt(self.K + discounted_grad_sum))






