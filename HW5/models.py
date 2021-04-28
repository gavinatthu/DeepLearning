import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import os

os.environ["CUDA_VISIBLE_DEVICES"]="7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################
# DataLoader
################################

# set up fields


def plot_his(train_loss_his, train_acc_his, mode):
    X1 = range(0, len(train_loss_his[0]))
    plt.figure(figsize=(15,15))
    plt.subplot(2,1,1)
    plt.plot(X1,train_loss_his[0],  linewidth=1.0, linestyle="-",label="(0.001, 256, 0)")
    plt.plot(X1,train_loss_his[1],  linewidth=1.0, linestyle="-",label="(0.001, 128, 0)")
    plt.plot(X1,train_loss_his[2],  linewidth=1.0, linestyle="-",label="(0.001, 256, 0.2)")
    plt.plot(X1,train_loss_his[3],  linewidth=1.0, linestyle="-",label="(0.001, 128, 0.2)")
    plt.plot(X1,train_loss_his[4],  linewidth=1.0, linestyle="-",label="(0.003, 256, 0)")
    plt.plot(X1,train_loss_his[5],  linewidth=1.0, linestyle="-",label="(0.003, 256, 0.2)")
    plt.legend()
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss & Acc of Results')
    plt.subplot(2,1,2)
    plt.plot(X1,train_acc_his[0],  linewidth=1.0, linestyle="-",label="(0.001, 256, 0)")
    plt.plot(X1,train_acc_his[1],  linewidth=1.0, linestyle="-",label="(0.001, 128, 0)")
    plt.plot(X1,train_acc_his[2],  linewidth=1.0, linestyle="-",label="(0.001, 256, 0.2)")
    plt.plot(X1,train_acc_his[3],  linewidth=1.0, linestyle="-",label="(0.001, 128, 0.2)")
    plt.plot(X1,train_acc_his[4],  linewidth=1.0, linestyle="-",label="(0.003, 256, 0)")
    plt.plot(X1,train_acc_his[5],  linewidth=1.0, linestyle="-",label="(0.003, 256, 0.2)")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlabel('Iteration Times(Epochs for test)')
    plt.savefig('./result'+str(mode)+'.jpg', dpi=400)
    plt.close()

def train_epoch(model, optimizer, data_loader, loss_history, val_loader, acc_history):
    total_samples = len(data_loader.dataset)
    correct_samples = 0.0
    model.train()
    for i in range(1, len(data_loader) + 1):
    #for i, (input, labels) in enumerate(data_loader):
        batch = next(iter(data_loader))
        input = batch.text.to(device)
        target = batch.label - 1 #chanege the scale from 1~5 to 0~4
        target = target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(input), dim=1)
        
        loss = F.nll_loss(output, target)
        _, pred = torch.max(output, dim=1)
        correct_samples += pred.eq(target).sum()
        acc = correct_samples / (40*len(batch))

        loss.backward()
        optimizer.step()

        #evaluate(model, val_loader, val_loss, val_acc)

        if i % 40 == 0:
            print('[' +  '{:5}'.format(i * len(batch)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) + '  Acc: ' + '{:5}'.format(correct_samples) + '/' +
                  '{:5}'.format(40*len(batch)))
            correct_samples = 0.0
            loss_history.append(loss.item())
            acc_history.append(acc.item())



def evaluate(model, data_loader, loss_history, acc_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    for i in range(len(data_loader)):
        batch = next(iter(data_loader))
        input = batch.text.to(device)
        target = batch.label -1 #chanege the scale from 1~5 to 0~4
        target = target.to(device)

        output = F.log_softmax(model(input), dim=1)
        loss = F.nll_loss(output, target, reduction='sum')
        _, pred = torch.max(output, dim=1)
        
        total_loss += loss.item()
        correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    acc = correct_samples / total_samples
    loss_history.append(avg_loss)
    acc_history.append(acc)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * acc) + '%)\n')

          
def DataLoader(BATCH_SIZE):
    TEXT = data.Field()
    LABEL = data.Field(sequential=False,dtype=torch.long)

    # make splits for data
    train, val, test = datasets.SST.splits(TEXT, LABEL, fine_grained=True, train_subtrees=False)

    # build the vocabulary
    TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
    LABEL.build_vocab(train)

    # make iterator for splits
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=BATCH_SIZE)
    return TEXT, LABEL, train_iter, val_iter, test_iter



'''Attention: batch.label in the range [1,5] not [0,4] !!!'''

class RNN(nn.Module):
    def __init__(self, pretrained_emb, dropout, hidden_dim, output_dim):
        
        super().__init__()
        self.d = dropout
        #self.bn = nn.BatchNorm1d()
        input_dim, embedding_dim = pretrained_emb.size()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.data.copy_(pretrained_emb)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        #x = self.bn(input)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(hidden[-1,:,:])
        x = self.fc(hidden)
        return x


class RNN1(nn.Module):
    def __init__(self, pretrained_emb, dropout, hidden_size, output_size):
        super(RNN, self).__init__()
        emb_size, emb_dimension = pretrained_emb.size()
        num_layers = 2
        self.hidden_size = hidden_size
        self.sup = num_layers * 2             # num_layers * num_directions
        # Loading pre-trained embeddings
        self.embeddings = nn.Embedding(emb_size, emb_dimension)
        self.embeddings.weight = nn.Parameter(pretrained_emb, requires_grad=False)
        
        # Bi-directional layer
        self.rnn = nn.RNN(emb_dimension, hidden_size, num_layers=2, bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear Layers
        self.fc1 = nn.Linear(self.sup*hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), output_size)
        
        # Cross-Entropy Loss
        self.softmax = nn.CrossEntropyLoss()
        
        
    def linear_block(self, x, linear_layer):
        linear_out = linear_layer(x)
        linear_out = F.relu(linear_out)
        linear_out = self.dropout(linear_out)
        return linear_out
        
        
    def forward(self, input_sentences):
        # Get embeddings of each of the words in the sentences
        sentence_of_emb = self.embeddings(input_sentences)
        # [sentence_length, batch_size, emb_dim]
        h_0 = torch.zeros(self.sup, input_sentences.size(0), self.hidden_size).cuda()
        # out : [sentence_length, batch_size, 2 * hidden_size]
        out, h_n = self.rnn(sentence_of_emb)

        h_n = h_n.contiguous().view(h_n.size(0), -1)
        linear_out_1 = self.linear_block(h_n, self.fc1)
        logits = self.fc2(linear_out_1)
        print(logits)
        return logits

'''

################################
# After build your network 
################################


# Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

# you should maintain a nn.embedding layer in your network
model.embedding.weight.data.copy_(pretrained_embeddings)
'''



