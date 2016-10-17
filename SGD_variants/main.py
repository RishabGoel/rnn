from SGD_variants import SGDvariants
import numpy as np
"""
This class handles
- the assignment of sgd and learning rate schedules for learning task
- learn the weights of the model
"""
class SGD(object):
	
	def __init__(self, batch_size = 100, iterations = 1000, exp_base = 10, r = 2.5*1e8, c = 0.75, momentum_gamma = 0.9, adag_epsilon = 1, rms_gamma = 0.9, adam_beta1 = 0.9, adam_beta2 = 0.999, adam_epsilon = 1e-8, algo = "vanilla_sgd", lr_schedule = None):
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

		algo : SGD schedule that one wants to try
		Possible values are -
		- vanilla_sgd
		- momentum
		- adagrad
		- rms_prop
		- adam

		lr_schedule : The kind of decay schedule for learning rate that one wants
		- exponential_lr_decay
		- power_lr_decay

		"""
		self.batch_size = batch_size
		self.iterations = iterations
		self.exp_base = exp_base
		self.r = r
		self.c = c
		self.momentum_gamma = momentum_gamma
		self.adag_epsilon = adag_epsilon
		self.rms_gamma = rms_gamma
		self.adam_beta1 = adam_beta1
		self.adam_beta2 = adam_beta2
		self.adam_epsilon = adam_epsilon
		self.sgd_algos = SGDvariants(exp_base = 10, r = 2.5*1e8, c = 0.75, momentum_gamma = 0.9, adag_epsilon = 1, rms_gamma = 0.9, adam_beta1 = 0.9, adam_beta2 = 0.999, adam_epsilon = 1e-8)
		self.type = algo
		self.lr_scheduling = lr_schedule
		print "batch_size is %d, iterations are %d, type of sgd is %s, lr_schedule is %s" % (batch_size, iterations, algo, lr_schedule)

		if lr_schedule == "exponential_lr_decay":
			self.lr_schedule = self.sgd_algos.exponential_scheduling
		elif lr_schedule == "power_lr_decay":
			self.lr_schedule = self.sgd_algos.power_scheduling

	def set_algo(self, x_dim):
		"""
		This function sets the lr and sgd routines and initialse any accumulating parameters required for sgd 

		x_dim : This the input dimension of the feature vector
		"""
		if self.type == "vanilla_sgd":
			self.algo = self.sgd_algos.sgd
			self.params = {}
		elif self.type == "momentum":
			self.algo = self.sgd_algos.momentum_sgd
			self.params = {}
			self.params["prev_v"] = np.zeros((x_dim,1))
		elif self.type == "adagrad":
			self.algo = self.sgd_algos.AdaGrad
			self.params = {}
			self.params["gradient_sum"] = np.zeros((x_dim,1))
		elif self.type == "rms_prop":
			self.algo = self.sgd_algos.RMSprop
			self.params = {}
			self.params["decaying_grad_sum"] = np.zeros((x_dim,1))
		elif self.type == "adam":
			self.algo = self.sgd_algos.Adam
			self.params={}
			self.params["first_moment"] = np.zeros((x_dim,1))
			self.params["second_moment"] = np.zeros((x_dim,1))

	def train(self, data = None, labels = None, initial_lr = 0.01):
		"""
		This function learns the weights of the selected model
		"""
		batch_num = (data.shape[0] / self.batch_size)
		# print "no of batches are %d"%batch_num
		# print data.shape, labels.shape
		self.set_algo(data.shape[1])
		weights = np.random.randn(data.shape[1],1)
		# print "initial weights are", weights
		for iters in xrange(self.iterations):	
		# for iters in xrange(1):	
			lr = initial_lr
			# print "%d iters of %d"%(iters, self.iterations)
			for batch in xrange(batch_num):
			# for batch in xrange(1):
				data_batch = data[batch * self.batch_size : (batch + 1) * self.batch_size]
				labels_batch = labels[batch * self.batch_size : (batch + 1) * self.batch_size]
				if self.lr_scheduling != None:
					lr = self.lr_schedule(lr, batch)
				y_hat = np.dot(data_batch, weights)
				diff = y_hat - labels_batch
				
				gradient = (np.dot(np.transpose(data_batch), diff))/float(self.batch_size)
				# print gradient, self.params
				self.params, weights = self.algo(gradient, lr, weights, self.params)

			# print "loss after ",iters," iterations is"
		# print "final weights are ", weights
		return weights






		

		
		

			



		