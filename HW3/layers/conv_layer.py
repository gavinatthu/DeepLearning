# -*- encoding: utf-8 -*-
import numpy as np
# if you implement ConvLayer by convolve function, you will use the following code.

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
	'''
	Parameters
	----------
	input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
	filter_h : 卷积核的高
	filter_w : 卷积核的长
	stride : 步幅
	pad : 填充

	Returns
	-------
	col : 2维数组
	'''
	# 输入数据的形状
	# N：批数目，C：通道数，H：输入数据高，W：输入数据长
	N, C, H, W = input_data.shape  
	out_h = (H + 2*pad - filter_h)//stride + 1  # 输出数据的高
	out_w = (W + 2*pad - filter_w)//stride + 1  # 输出数据的长
	# 填充 H,W
	img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
	# (N, C, filter_h, filter_w, out_h, out_w)的0矩阵
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
	# 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
	return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
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





class ConvLayer():
	"""
	2D convolutional layer.
	This layer creates a convolution kernel that is convolved with the layer
	input to produce a tensor of outputs.
	Arguments:
		inputs: Integer, the channels number of input.
		filters: Integer, the number of filters in the convolution.
		kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
		pad: Integer, the size of padding area.
		trainable: Boolean, whether this layer is trainable.
	"""
	def __init__(self, inputs,
	             filters,
	             kernel_size,
	             pad,
	             trainable=True):
		self.inputs = inputs
		self.filters = filters
		self.kernel_size = kernel_size
		self.pad = pad
		assert pad < kernel_size, "pad should be less than kernel_size"
		self.trainable = trainable

		self.XavierInit()

		self.grad_W = np.zeros_like(self.W)
		self.grad_b = np.zeros_like(self.b)


	def XavierInit(self):
		raw_std = (2 / (self.inputs + self.filters))**0.5
		init_std = raw_std * (2**0.5)

		self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))

		self.b = np.random.normal(0, init_std, (self.filters,))


	def forward(self, Input, **kwargs):
		'''
		forward method: perform convolution operation on the input.
		Agrs:
			Input: A batch of images, shape-(batch_size, channels, height, width)
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		#print("input.shape = ", Input.shape)
		stride = 1

		N, C, H, W = Input.shape
		k_num, _, k_H, k_W = self.W.shape
		H_new = int((H + 2 * self.pad - k_H)/stride) + 1
		W_new = int((W + 2 * self.pad - k_W)/stride) + 1
		input_col = im2col(Input, k_H, k_W, stride = 1, pad = self.pad)
		'''此处rot180'''
		#W_rot = np.rot90(self.W, 2, axes=(2, 3))
		W_rot = self.W
		W_col = W_rot.reshape(k_num, -1)
		
		output = np.dot(input_col, W_col.T) + self.b
		output = output.reshape(N, H_new, W_new, -1).transpose(0, 3, 1, 2)
		
		self.Input = Input
		self.input_col = input_col
		self.W_col = W_col

		return output
		############################################################################


	def backward(self, delta):
		'''
		backward method: perform back-propagation operation on weights and biases.
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate self.grad_W, self.grad_b, and return the new delta.
		#print("W.shape=", self.W.shape)
		k_num, C, k_H, k_W = self.W.shape
		self.grad_b = np.sum(delta, axis = (0, 2, 3))

		#delta = np.rot90(delta, 2, axes=(2, 3))
		delta = delta.transpose(0, 2, 3, 1).reshape(-1, k_num)
		
		#self.grad_b = np.sum(delta, axis = 0)
		#print("input_col.T:", self.input_col.T.shape, "delta:", delta.shape)
		self.grad_W = np.dot(self.input_col.T, delta)
		self.grad_W = self.grad_W.transpose(1, 0).reshape(k_num, C, k_H, k_W)

		output_col = np.dot(delta, self.W_col)
		output = col2im(output_col, self.Input.shape, k_H, k_W, stride = 1, pad = self.pad)
		return output
		#delta = np.pad(delta, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		
		'''
		#img2coll method
		self.grad_W = self.img2col(Input.transpose((1, 0, 2, 3)), delta.transpose((1, 0, 2, 3))).transpose((1, 0, 2, 3))
		delta_pad = np.pad(delta, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), 'constant')


		W_rot = np.rot90(self.W, 2, axes=(2, 3)) # rot180
		#print("delta_pad.shape = ", delta_pad.shape)
		#print("W.shapee = ", W_rot.shape)
		delta_out = self.img2col(delta_pad, W_rot)
		delta_out = delta_out[:, :, self.pad:-self.pad, self.pad:-self.pad]

		return delta_out
		'''
	    ############################################################################


	'''
	def img2col(self, Input, w, b = None, stride = 1):
		N, C, H, W = Input.shape
		k_num, _, k_H, k_W = w.shape
		H_new = int((H - k_H)/stride) + 1
		W_new = int((W - k_W)/stride) + 1
		shape = (C, k_H, k_W, N, H_new, W_new)
		strides = Input.itemsize * np.array([H * W, W, 1, C * H * W, W * stride, stride])
		y_col = np.lib.stride_tricks.as_strided(Input, shape=shape, strides=strides)
		y_col = y_col.reshape(C * k_H * k_W, N * H_new * W_new)
		output = w.reshape(k_num, -1).dot(y_col)
		if b is not None:
			output += b.reshape(-1, 1)
		output = output.reshape(k_num, N, H_new, W_new).transpose((1, 0, 2, 3))		
		return output
	'''
'''
input = np.random.random((100, 8, 14, 14)) - 0.5
delta = np.random.random((100, 16, 14, 14)) - 0.5
w = np.random.normal(0, 1, (8, 1, 3, 3))
layer = ConvLayer(1, 8, 3, 1)
fw = layer.forward(input)
bw = layer.backward(delta)


'''


