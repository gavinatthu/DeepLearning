import time
import torch
import numpy as np

from models import *
import os

os.environ["CUDA_VISIBLE_DEVICES"]="7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 50
N_EPOCHS = 50
L_RATE = 0.001
H_SIZE = 256
DROPOUT = 0.2

TEXT, LABEL, train_loader, val_loader, test_loader = DataLoader(BATCH_SIZE)

output_size = 5



train_loss_his, train_acc_his, test_loss_his, test_acc_his = [], [], [], []

start_time = time.time()

for (L_RATE, H_SIZE, DROPOUT) in [(0.001, 256, 0),(0.001, 128, 0),(0.001, 256, 0.2),(0.001, 128, 0.2),(0.003, 256, 0),(0.003, 256, 0.2)]:
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    model = RNN(TEXT.vocab.vectors, DROPOUT, H_SIZE, output_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=L_RATE)
    
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss, val_loader, train_acc)
        evaluate(model, test_loader, test_loss, test_acc)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    train_loss_his.append(train_loss)
    train_acc_his.append(train_acc)
    test_loss_his.append(test_loss)
    test_acc_his.append(test_acc)

'''
plot_result(train_loss, train_acc,0)
plot_result(test_loss, test_acc,1)
'''
plot_his(train_loss_his, train_acc_his, 0)
plot_his(test_loss_his, test_acc_his, 1)