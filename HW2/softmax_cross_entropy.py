""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		
		# Store values
		self.input = logit
		self.gt = gt

		# Loss
		self.loss1 = - np.sum(logit * gt, axis=1) 
		self.loss2 = np.log(np.sum(np.exp(logit), axis=1))
		self.loss = np.sum((self.loss1 + self.loss2), axis=0)/logit.shape[0]
        
		# Accurary
		self.prediction = np.argmax(self.input, axis=1)
		gt = np.argmax(gt, axis=1)
		self.acc = np.sum(self.prediction == gt) / float(logit.shape[0])
	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)

		self.h = np.exp(self.input) / np.sum(np.exp(self.input), axis=1)[:,None]
		return (self.h - self.gt)
	    ############################################################################
'''
a = SoftmaxCrossEntropyLossLayer()
X = np.random.random((50, 10)) - 10
W = np.random.random((50, 10)) + 10
loss = a.forward(X,W)
bw = a.backward()
acc = a.acc
print(bw)
'''