import csv
import sys
import os.path as op
import re
import torch
import codecs
import json
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd


class AGNEWs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0=1014, label_n_txt=None):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        # read alphabet
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        self.alphabet = alphabet
        self.l0 = l0
        self.load(label_n_txt)
        self.y = torch.LongTensor(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def load(self, label_n_txt=None, lowercase=True):

        self.label = []
        self.data = []
        file = self.label_data_path

        if label_n_txt is None:

            with open(self.label_data_path, 'r') as f:
                rdr = csv.reader(f, delimiter=',', quotechar='"')
                try:
                    content = [row for row in rdr]
                except csv.Error:
                    csv.field_size_limit(sys.maxsize)
                    content = [row for row in rdr]

                # num_samples = len(content)

                for row in content:
                    try:
                        lbl = int(row[0])
                    except ValueError:
                        continue

                    self.label.append(lbl)
                    txt = ' '.join(row[1:])
                    if lowercase:
                        txt = txt.lower()
                    self.data.append(txt)
        else:
            self.label, self.data = list(zip(*label_n_txt))
            self.label = list(self.label)
            self.data = [txt.lower() if lowercase else txt for txt in self.data]

    def get_num_class(self):
        label_set = set(self.label)
        return len(label_set)

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in zip(range(self.l0), sequence[::-1]):
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def get_class_weight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples / float(self.label.count(c)) for c in label_set]
        return class_weight, num_class


if __name__ == '__main__':

    label_data_path = '/Users/ychen/Documents/TextClfy/data/ag_news_csv/test.csv'
    alphabet_path = '/Users/ychen/Documents/TextClfy/alphabet.json'

    train_dataset = AGNEWs(label_data_path, alphabet_path)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, drop_last=False)
    # print(len(train_loader))
    # print(train_loader.__len__())

    # size = 0
    for i_batch, sample_batched in enumerate(train_loader):
        # len(i_batch)
        # print(sample_batched['label'].size())
        inputs = sample_batched['data']
        print(inputs.size())
        # print('type(target): ', target)
