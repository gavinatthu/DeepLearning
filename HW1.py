import mnist_data_loader
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
max_epoch = 5
lr = 0.1

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def cost_function(t, h):
    J = -1.0 / batch_size * np.sum(t * np.log(h)+ (1 - t) * np.log(1 - h))
    return J

def d_cross_function(x, t, h):
    return 1.0 / batch_size * np.dot(x.T, (h - t))

def err(predict,label):
    return np.mean(0.5*(label-predict)**2)

mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)
# training dataset
train_set = mnist_dataset.train
# test dataset
test_set = mnist_dataset.test
print("Training dataset size:", train_set.num_examples)
print("Test dataset size:", test_set.num_examples)


example_id = 1201
image = train_set.images[example_id] # shape = 784 (28*28)
label = train_set.labels[example_id] # shape = 1
print("label:", label)
plt.imshow(np.reshape(image,[28,28]),cmap='gray')
plt.show()


w = np.zeros((784, ))
cost_record = []
err_record = []

# train
for epoch in range(0, max_epoch):
    print("epoch:", epoch)
    iter_per_batch = train_set.num_examples // batch_size
    for batch_id in range(0, iter_per_batch):
    # get the data of next minibatch (have been shuffled)
        batch = train_set.next_batch(batch_size)
        input, label = batch
        label = label/3 - 1
        # prediction
        h = sigmoid(np.dot(input, w))
        h = np.squeeze(h)
        cost = cost_function(label, h)
        cost = np.squeeze(cost)
        predict = h > 0.5
        E = err(label, predict)
        # calculate the loss (and accuracy)
        dw = d_cross_function(input, label, h)
        #print(label.shape,h.shape,dw.shape)
        w_next = w - lr * dw         # Gradient descent
        # update weights
        w = w_next
        if batch_id % 1 == 0:
            err_record.append(E)
            cost_record.append(cost)
            #print("Cost in batch:", batch_id, " = ", cost)
            #print("Err in batch:", batch_id, " = ", E)

# test     
w = w.reshape(-1, 1)
label_train = np.squeeze(train_set.labels)/3 -1
label_test = np.squeeze(test_set.labels)/3 - 1
h_train = np.squeeze(sigmoid(np.dot(train_set.images, w)))
h_test = np.squeeze(sigmoid(np.dot(test_set.images, w)))
predict_train = h_train > 0.5
predict_test = h_test > 0.5

print("Final error=", err(label_train, predict_train))

Y1 = cost_record
Y2 = err_record
X1 = np.linspace(1, len(Y1), len(Y1))
#plt.plot(X1,Y1,  linewidth=1.0, linestyle="-",label="Full Connection Network")
plt.plot(X1,Y1,  linewidth=1.0, linestyle="-",label="Cost Record")
plt.legend()
plt.xlabel('Iteration Times')
plt.ylabel('Value')
plt.savefig('./cost.jpg')
plt.show()

plt.plot(X1,Y2,  linewidth=1.0, linestyle="-",color="red",label="Error(MSE) Record")
plt.legend()
plt.xlabel('Iteration Times')
plt.ylabel('Value')
plt.savefig('./err.jpg')
plt.show()