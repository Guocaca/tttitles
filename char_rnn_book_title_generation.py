from __future__ import print_function, division
import sys
import random
import time
import math
import string

import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Some parameters to adjust
learning_rate = 0.0005
dropout = 0.1
criterion = nn.NLLLoss()
n_iters = 100000
print_every = 5000
plot_every = 500
max_length = 60

# Get all the possible letters and all training samples in a list
def readFile(file_name):
    all_letters = set()
    all_titles = open(file_name).read().strip().split('\n')
    for title in all_titles:
        all_letters = all_letters.union(set(title))
    return ''.join(list(all_letters)), all_titles

file_name = sys.argv[1]
all_letters, all_titles = readFile(file_name)
n_letters = len(all_letters) + 1 # For EOS

# Random choice from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Timer
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes) 

# Generate random training example
def randomTrainingExample():
    line = randomChoice(all_titles)
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return input_line_tensor, target_line_tensor

# Initiate some parameters for training
rnn = RNN(n_letters, 128, n_letters)

# Training the neural network
def train(input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0] / input_line_tensor.size()[0]

# Start training here
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

plt.plot(all_losses)

# Sample from a starting letter
def sample(start_letter = 'a'):
    input = Variable(inputTensor(start_letter))
    hidden = rnn.initHidden()

    output_title = start_letter

    for i in range(max_length):
        output, hidden = rnn(input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_title += letter
        input = Variable(inputTensor(letter))

    return output_title

def samples(times):
    for _ in range(times):
        start_letter = randomChoice(string.ascii_lowercase)
        print(sample(start_letter))

samples(5)