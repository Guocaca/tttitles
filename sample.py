import torch

from model_rnn import *
from read import *
from train_snd_save import *

# The maximum possible length of generated book title
max_length = 30

rnn = torch.load(model_save_name)

# Random choice from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

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

if __name__ == '__main__':
    samples(5)
