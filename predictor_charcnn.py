import os
import argparse
import sys
import numpy as np

from charcnn.model import CharCNN
from charcnn.data_loader import AGNEWs
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from charcnn.metric import print_f_score


def get_args():
    parser = argparse.ArgumentParser(description='Character level CNN text classifier testing',
                                     formatter_class=argparse.RawTextHelpFormatter)
    # model
    parser.add_argument('--model_path', default=None,
                        help='Path to pre-trained acouctics model created by DeepSpeech training')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--kernel_num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--channel_size', type=int, default=256, help='size of channels')
    parser.add_argument('--pool_size', type=int, default=3, help='size of channels')
    parser.add_argument('--fc_size', type=int, default=1024, help='size of channels')
    parser.add_argument('--kernel_sizes', nargs="+", type=int, default=[7, 7, 3, 3, 3, 3],
                        help='space-separated kernel sizes to use for convolution')
    # data
    parser.add_argument('--test_path', metavar='DIR',
                        help='path to testing data csv', default='charcnn/data/ag_news_csv/test.csv')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size for training [default: 128]')
    parser.add_argument('--alphabet_path', default='charcnn/alphabet.json', help='Contains all characters for prediction')
    # device
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
    # logging options
    parser.add_argument('--save_folder', default='charcnn/Results/', help='Location to save epoch models')

    args = parser.parse_args()
    return args


class Predictor_charcnn(object):
    def __init__(self, cuda=True, alphabet_path='charcnn/alphabet.json', batch_size=20, num_workers=4,
                 model_path=None, kernel_sizes=[7, 7, 3, 3, 3, 3],
                 channel_size=256, pool_size=3, fc_size=1024, dropout=0.5):
        self.cuda = cuda
        self.alphabet_path = alphabet_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_path = model_path
        self.kernel_sizes = kernel_sizes
        self.channel_size = channel_size
        self.pool_size = pool_size
        self.fc_size = fc_size
        self.dropout = dropout

        self._init_model()

    def _init_model(self):
        print("=> loading weights from '{}'".format(self.model_path))
        assert os.path.isfile(self.model_path), "=> no checkpoint found at '{}'".format(self.model_path)
        checkpoint = torch.load(self.model_path)

        model = CharCNN(n_class=checkpoint['n_class'],
                        n_char=checkpoint['n_char'], max_seq_len=checkpoint['max_seq_len'],
                        kernel_sizes=self.kernel_sizes,
                        channel_size=self.channel_size, pool_size=self.pool_size,
                        fc_size=self.fc_size, dropout=self.dropout)

        model.load_state_dict(checkpoint['state_dict'])

        # using GPU
        if self.cuda:
            model = torch.nn.DataParallel(model).cuda()

        model.eval()
        self.model = model
        self.n_char = checkpoint['n_char']
        self.l0 = checkpoint['max_seq_len']

    def _get_data(self, test_path, alphabet_path, l0, label_n_txt=None):
        # load testing data
        print("\nLoading testing data...")
        test_dataset = AGNEWs(label_data_path=test_path, alphabet_path=alphabet_path,
                              l0=l0, label_n_txt=label_n_txt)
        print("Transferring testing data to iterator...")
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

        _, num_class_test = test_dataset.get_class_weight()
        print('\nNumber of testing samples: ' + str(test_dataset.__len__()))
        for i, c in enumerate(num_class_test):
            print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))

        assert self.n_char == len(test_dataset.alphabet)
        return test_loader

    def pred(self, test_path=None, label_n_txt=None):
        test_loader = self._get_data(test_path, self.alphabet_path, self.l0, label_n_txt=label_n_txt)

        corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
        predicates_all, target_all, prob_all = [], [], []
        print('\nTesting...')
        for i_batch, (data) in enumerate(test_loader):
            inputs, target = data
            target.sub_(1)
            size += len(target)

            prob, predicates, loss, corr = self.pred_batch(inputs, target)

            accumulated_loss += loss
            corrects += corr
            predicates_all += predicates.cpu().numpy().tolist()
            target_all += target.data.cpu().numpy().tolist()
            prob_all += [prob.data.cpu()]

        prob_all = torch.cat(prob_all, 0)
        avg_loss = accumulated_loss / size
        accuracy = 100.0 * corrects / size
        print('\rEvaluation - loss: {:.6f}  acc: {:.3f}%({}/{}) '.format(avg_loss,
                                                                         accuracy,
                                                                         corrects,
                                                                         size))
        print_f_score(predicates_all, target_all)
        return prob_all.numpy()

    def pred_batch(self, inputs, target):
        if self.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        with torch.no_grad():
            inputs = Variable(inputs)
        target = Variable(target)
        logit = self.model(inputs)
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        prob = torch.exp(logit)

        loss = F.nll_loss(logit, target, size_average=False).data[0]
        corr = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        return prob, predicates, loss, corr


if __name__ == '__main__':
    args = get_args()
    predictor = Predictor_charcnn(cuda=args.cuda, alphabet_path=args.alphabet_path,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  model_path=args.model_path, kernel_sizes=args.kernel_sizes,
                                  channel_size=args.channel_size, pool_size=args.pool_size,
                                  fc_size=args.fc_size, dropout=args.dropout)
    prob = predictor.pred(args.test_path, None)
