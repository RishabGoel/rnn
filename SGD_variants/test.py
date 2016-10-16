import numpy as np
import matplotlib.pyplot as plt
from main import SGD

def generate_test_data(train_num = 10000, test_num = 100):
	noise = np.random.normal(0, 0.5, train_num).reshape((train_num,1))
	x = np.random.rand(train_num, 2)
	x_test = np.random.rand(test_num, 2)
	weights = np.random.rand(2, 1)
	y = np.dot(x, weights) + noise
	y_test = np.dot(x_test, weights)
	return x, x_test, y, y_test, weights

def main():
	"generating dummy data ..."
	x, x_test, y, y_test, weights = generate_test_data()

	"creating the SGD Objects ..."
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
	costs = []
	types = []
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
