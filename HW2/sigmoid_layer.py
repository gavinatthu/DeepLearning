""" Sigmoid Layer """

import numpy as np


class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.

		self.input = Input
		self.input = 1. / (1. + np.exp(-Input))
		return self.input


	    ############################################################################
	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta
		return self.input * (1. - self.input) * delta
		

	    ############################################################################

'''
a = SigmoidLayer()
Input = np.random.random((100, 10)) - 0.5
delta = np.random.random((100, 10)) + 0.5
fw = a.forward(Input)
bw = a.backward(delta)
print(bw.shape)
'''