# -*- encoding: utf-8 -*-
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=2, pad=0):

	N, C, H, W = input_data.shape  
	out_h = (H + 2*pad - filter_h)//stride + 1  # 输出数据的高
	out_w = (W + 2*pad - filter_w)//stride + 1  # 输出数据的长
	img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
	return col


def col2im(col, input_shape, filter_h, filter_w, stride=2, pad=0):
	N, C, H, W = input_shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1
	col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

	img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

	return img[:, :, pad:H + pad, pad:W + pad]




class MaxPoolingLayer():
	def __init__(self, kernel_size, pad):
		'''
		This class performs max pooling operation on the input.
		Args:
			kernel_size: The height/width of the pooling kernel.
			pad: The width of the pad zone.
		'''

		self.kernel_size = kernel_size
		self.pad = pad
		self.trainable = False

	def forward(self, Input, **kwargs):
		'''
		This method performs max pooling operation on the input.
		Args:
			Input: The input need to be pooled.
		Return:
			The tensor after being pooled.
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		N ,C, H, W = Input.shape
		H_new = int((H + 2*self.pad) / self.kernel_size)
		W_new = int((W + 2*self.pad) / self.kernel_size)

		input_col = im2col(Input, self.kernel_size, self.kernel_size).reshape(-1, self.kernel_size**2)
		
		index = np.argmax(input_col, axis=1)
		output = np.max(input_col, axis=1)
		output = output.reshape(N, H_new, W_new, C).transpose(0, 3, 1, 2)


		self.Input = Input
		self.index = index
		return output
		
		'''
		stride = self.kernel_size
		Input = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		 #shape = (#sample) x (#input channel) x (#height) x (#width)


		shape = (C, self.kernel_size, self.kernel_size, N, H_new, W_new)
		strides = Input.itemsize * np.array([H * W, W, 1, C * H * W, W * stride, stride])
		y_col = np.lib.stride_tricks.as_strided(Input, shape=shape, strides=strides)
		y_col = y_col.reshape(self.kernel_size ** 2, -1)

		## Max pooling
		output = y_col.max(axis=0).reshape(N, C, H_new, W_new)

		self.gamma = np.zeros_like(y_col.T)
		index = (np.array(range(y_col.T.shape[0])),np.argmax(y_col.T, axis = 1))
		self.gamma[index] = 1
		self.gamma = self.gamma.T.reshape(N, C, H_new*self.kernel_size, W_new*self.kernel_size)
		'''
	    ############################################################################



	def backward(self, delta):
		'''
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the new delta.

		# upsample(delta)
		delta = delta.transpose(0, 2, 3, 1)
		delta_max = np.zeros((delta.size, self.kernel_size**2))
		delta_max[np.arange(self.index.size), self.index.flatten()] = delta.flatten()
		delta_max = delta_max.reshape(delta.shape + (self.kernel_size**2,))
		
		delta_col = delta_max.reshape(delta_max.shape[0] * delta_max.shape[1] * delta_max.shape[2], -1)
		
		delta_out = col2im(delta_col, self.Input.shape, self.kernel_size, self.kernel_size)
		
		'''
		delta = np.kron(delta, np.ones((self.kernel_size, self.kernel_size)))
		delta_out = self.gamma * delta
		#print(delta_out)
		'''
		return delta_out
	    ############################################################################


'''
layer = MaxPoolingLayer(2, 0)
input = np.random.random((100, 1, 28, 28)) - 0.5
delta = np.random.random((100, 1, 14, 14)) - 0.5

fw = layer.forward(input)
bw = layer.backward(delta)
print(bw.shape)

delta = np.array([[0,1],[5,3],[1,2]])

#print(delta)
index = (np.array(range(delta.shape[1])),np.argmax(delta, axis = 0))
#print(index)
new = np.zeros_like(delta)
new[index] = 1

#print(new)

#index = index.tolist()


#bw = layer.backward(delta)
#print(fw.shape)
'''