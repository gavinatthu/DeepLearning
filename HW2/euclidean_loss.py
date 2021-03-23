""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.acc and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		self.logit = logit
		self.gt = gt
		self.loss = 1 / 2 / logit.shape[0] * np.linalg.norm(logit - gt)**2
		self.prediction = np.argmax(logit, axis=1)
		gt = np.argmax(gt, axis=1)
		self.acc = np.sum(self.prediction == gt, axis=0) / float(logit.shape[0])
	    ############################################################################
		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		self.gradient = (self.logit - self.gt)
		return self.gradient

	    ############################################################################
'''
a = EuclideanLossLayer()
X = np.random.random((500, 10)) - 10
W = np.random.random((500, 10)) + 10
loss = a.forward(X,W)
bw = a.backward()
acc = a.acc
print(bw)
'''