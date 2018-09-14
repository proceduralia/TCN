import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from utils import data_generator
from model import TCN
import time

parser = argparse.ArgumentParser(description='Sequence Modeling - Delta Task')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=8,
                    help='kernel size (default: 8)')
parser.add_argument('--iters', type=int, default=100,
                    help='number of iters per epoch (default: 100)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=1000, metavar='N',
                    help='The size of the blank (i.e. T) (default: 1000)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 50')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 10)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


batch_size = args.batch_size
epochs = args.epochs
iters = args.iters
T = args.seq_len
n_steps = args.seq_len
n_train = 10000
n_test = 1000

print(args)
print("Preparing data...")
train_x, train_y = data_generator(T, n_train)
test_x, test_y = data_generator(T, n_test)


channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(1, 1, channel_sizes, kernel_size, dropout=dropout)

if args.cuda:
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

criterion = nn.MSELoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate():
    model.eval()
    out = model(test_x.unsqueeze(1).contiguous())
    loss = criterion(out.view(-1, 1), test_y.view(-1, 1))
    print('\nTest set: Average loss: {:.8f}\n'.format(loss.data[0]))
    return loss.data[0]


def train(ep):
    global batch_size, seq_len, iters, epochs
    model.train()
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    for batch_idx, batch in enumerate(range(0, n_train, batch_size)):
        start_ind = batch
        end_ind = start_ind + batch_size

        x = train_x[start_ind:end_ind]
        y = train_y[start_ind:end_ind]
        
        optimizer.zero_grad()
        out = model(x.unsqueeze(1).contiguous())
        loss = criterion(out.view(-1, 1), y.view(-1, 1))
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        total_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f}'.format(
                ep, batch_idx, n_train // batch_size+1, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss.data[0]))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0


for ep in range(1, epochs + 1):
    train(ep)
    evaluate()
