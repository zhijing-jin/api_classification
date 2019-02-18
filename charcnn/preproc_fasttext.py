import csv
import sys
import os
import random
from efficiency.log import fwrite


def ft_format():
    folders = ['ag', 'fake', 'yelp']

    csv.field_size_limit(sys.maxsize)

    for train_test in ['train', 'test']:
        for fol in folders:
            file = 'data/{}/{}.csv'.format(fol, train_test)
            with open(file) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                contents = [(row[0], ' '.join(row[1:])) for row in readCSV]
                print(f'[Info] Obtained {len(contents)} lines from CSV file.')

            file_fasttext = 'data/{}/{}_ft.txt'.format(fol, train_test)
            writeout = ['__label__{}\n'.format(' '.join(row)) for row in contents]
            fwrite(''.join(writeout), file_fasttext)


ft_format()
