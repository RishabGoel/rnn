import numpy as np
import matplotlib.pyplot as plt
from main import SGD



"""

This script
- generates dummy data
- compares various Stochastic Gradient Descent routines and learning rate schedules

"""
def generate_test_data(train_num = 10000, test_num = 100, inp_dimen = 2):
	"""
	This function generates the dummy data for learning

	noise:

	Noise is generally assumed to be normally distributed. So sampling it from normal distribution to be added to 
	target variable

	x, x_test: These are n x d matrix where n is number of training samples and d is thee dimension of the data

	y, y_test : These are n x 1 dimensional matrices with just one output dimension
	"""

	noise = np.random.normal(0, 0.5, train_num).reshape((train_num,1))
	x = np.random.rand(train_num, inp_dimen)
	x_test = np.random.rand(test_num, inp_dimen)
	weights = np.random.rand(inp_dimen, 1)
	y = np.dot(x, weights) + noise
	y_test = np.dot(x_test, weights)
	return x, x_test, y, y_test, weights

def main():
	"""
	This function compares the vaarious schedules for the sgd
	"""
	#generating dummy data ...
	x, x_test, y, y_test, weights = generate_test_data()

	# creating the SGD Objects for sgd comparisons ...
	sgds_exp = [SGD(algo = "vanilla_sgd", lr_schedule = "exponential_lr_decay"),
			 SGD(algo = "momentum", lr_schedule = "exponential_lr_decay"), 
			 SGD(algo = "adagrad", lr_schedule = "exponential_lr_decay"), 
			 SGD(algo = "rms_prop", lr_schedule = "exponential_lr_decay"), 
			 SGD(algo = "adam", lr_schedule = "exponential_lr_decay")]
	sgds_power = [SGD(algo = "vanilla_sgd", lr_schedule = "power_lr_decay"),
			 SGD(algo = "momentum", lr_schedule = "power_lr_decay"), 
			 SGD(algo = "adagrad", lr_schedule = "power_lr_decay"), 
			 SGD(algo = "rms_prop", lr_schedule = "power_lr_decay"), 
			 SGD(algo = "adam", lr_schedule = "power_lr_decay")]
	
	costs = [] 						# Accumulates the cost for the sgd routines in the above arrays
	types = [] 						# x axis labels for the type of routine
	# print x.shape
	sgds_exp[0].train(x,y)
	for i in sgds_exp:
		types.append(i.type)
		costs.append(np.sum(np.square(np.dot(x_test,i.train(x,y)))))

	for i in sgds_power:
		costs[sgds_power.index(i)] -= np.sum(np.square(np.dot(x_test,i.train(x,y))))
	type_idx = range(len(types))
	plt.xticks(type_idx, types)
	plt.scatter(type_idx, np.log(np.abs(costs)))
	# type_idx = range(len(types2))
	# plt.xticks(type_idx, types2)
	# plt.scatter(type_idx, costs[5:], color = "r")
	plt.show()
	print costs
	# sgd = SGD(algo = "adam", lr_schedule = "exponential_lr_decay")
	# sgd = SGD(lr_schedule = "exponential_lr_decay")
	# wts = sgd.train(x, y)
	# print "learned_weights ", wts

	# print "actual weights", weights

if __name__ == '__main__':
	main()
