""" Dropout Layer """

import numpy as np

class DropoutLayer():
	def __init__(self, p = 0.2):
		self.p = p
		self.trainable = False

	def forward(self, Input, is_training=True):

		############################################################################
	    # TODO: Put your code here
		mu = 1 - self.p
		if is_training:
			self.mask = np.random.uniform(size=Input.shape) > self.p
			mu = self.mask
		
		return Input * mu
	    #pass # delete before implement
	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here

		return delta * self.mask
	    
		#pass # delete before implement
	    ############################################################################
'''
layer = DropoutLayer()

input = np.random.random((100, 1, 14, 14)) - 0.5
delta = np.random.random((100, 1, 14, 14)) - 0.5

fw = layer.forward(input)
bw = layer.backward(delta)
print(bw.shape)
'''