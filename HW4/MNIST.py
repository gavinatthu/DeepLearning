import time
import torch
import torchvision
import matplotlib.pyplot as plt
from network import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



torch.manual_seed(24)

DOWNLOAD_PATH = '/data/mnist'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000
N_EPOCHS = 25
L_RATE = 0.003


# Normalize MNIST
transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history, acc_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            
            data = data.to(device)
            target = target.to(device)

            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    acc_history.append(correct_samples / total_samples)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')



start_time = time.time()

model = Conv().to(device)
model2 = Conv2().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=L_RATE)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=L_RATE, momentum=0.9, weight_decay=0.01)

train_loss_history, test_loss_history, acc_history = [], [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_epoch(model, optimizer, train_loader, train_loss_history)
    evaluate(model, test_loader, test_loss_history, acc_history)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


start_time = time.time()
train_loss_history2, test_loss_history2, acc_history2 = [], [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_epoch(model2, optimizer2, train_loader, train_loss_history2)
    evaluate(model2, test_loader, test_loss_history2, acc_history2)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

Y1 = train_loss_history
Y12 = train_loss_history2
X1 = range(0, len(Y1))

plt.plot(X1,Y1,  linewidth=1.0, linestyle="-",label="Adam")
plt.plot(X1,Y12,  linewidth=1.0, linestyle="-",label="SGD")
plt.legend()
plt.title('Train Loss vs. Inters')
plt.xlabel('Iteration Times')
plt.ylabel('Cross Entropy Loss')
plt.show()
plt.savefig('./train2.jpg')

Y2 = acc_history
Y22 = acc_history2
X2 = range(0, len(Y2))
Y3 = test_loss_history
Y32 = test_loss_history2
X3 = range(0, len(Y3))

plt.subplot(2,1,1)
plt.plot(X2,Y2,  linewidth=1.0, linestyle="-",label="Adam")
plt.plot(X2,Y22,  linewidth=1.0, linestyle="-",label="SGD")
plt.legend()
plt.title('Test Accuracy&Loss vs. Epoches')
plt.ylabel('Accuracy')
plt.subplot(2,1,2)
plt.plot(X3,Y3,  linewidth=1.0, linestyle="-",label="Adam")
plt.plot(X3,Y32,  linewidth=1.0, linestyle="-",label="SGD")
plt.legend()
plt.xlabel('Epoches')
plt.ylabel('Cross Entropy Loss')
plt.show()
plt.savefig('./test2.jpg')
