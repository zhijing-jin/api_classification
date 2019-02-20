from __future__ import division
import csv
import sys
import random


def read_mr():
    contents = []
    for ix, pos_neg in enumerate(['neg', 'pos']):
        file = '../data/mr/rt-polarity.{}'.format(pos_neg)
        with open(file, 'rb') as f:
            contents += [[str(ix + 1), row] for row in f]

    split_len = len(contents) // 10
    random.shuffle(contents)
    splits = (contents[:-split_len], contents[-split_len:])

    for split, train_test in zip(splits, ['train', 'test']):
        file = '../data/mr/{}.csv'.format(train_test)

        with open(file, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in split:
                writer.writerow(row)
            print('[Info] Written {} lines into {}'.format(len(split), file))


def save_tok():
    csv.field_size_limit(sys.maxsize)

    import spacy
    nlp = spacy.load('en')

    folders = ['ag', 'fake', 'yelp', 'mr']
    folders = ['mr']
    train_tests = ['train', 'test']

    for train_test in train_tests:
        for fol in folders:
            file = '../data/{}/{}.csv'.format(fol, train_test)
            file_tok = '../data/{}/{}_tok.csv'.format(fol, train_test)

            with open(file) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                contents = [(row[0], ' '.join(row[1:]))
                            for row in reader]
                print(f'[Info] Obtained {len(contents)} lines from CSV file {file}.')

            contents = [(row[0], tokenize(nlp, row[1])) for row in contents]

            with open(file_tok, mode='w') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in contents:
                    writer.writerow(row)
            print('[Info] Written {} lines into {}'.format(len(contents), file_tok))


def tokenize(nlp, sent_str, lowercase=True):
    doc = nlp(sent_str, disable=['parser', 'tagger', 'ner'])
    s = ' '.join([token.text for token in doc])
    if lowercase:
        s = s.lower()
    return s


def read_fakenews():
    csv.field_size_limit(sys.maxsize)
    n_test = 1000

    file = 'data/fake_news/train_raw.csv'
    fields = ['label', 'title', 'author', 'text']

    with open(file) as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        data = [{f: row[f] for f in fields} for row in csv_reader]
        print(f'[Info] Obtained {len(data)} lines from CSV file.')

    assert all(row[fields[0]] in '01' for row in data)

    data0 = [row for row in data if row[fields[0]] == '0']
    data1 = [row for row in data if row[fields[0]] == '1']

    random.shuffle(data0)
    random.shuffle(data1)

    data0_train, data0_test = data0[:-n_test], data0[-n_test:]
    data1_train, data1_test = data1[:-n_test], data1[-n_test:]

    train = data0_train + data1_train
    test = data0_test + data1_test

    random.shuffle(train)
    random.shuffle(test)

    def _writecsv(file, dic_list):
        with open(file, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            # writer.writeheader()
            for row in dic_list:
                row[fields[0]] = str(int(row[fields[0]]) + 1)
                writer.writerow(row)
            print("[Info] Written {} rows".format(len(dic_list)))

    _writecsv('data/fake_news/train.csv', train)
    _writecsv('data/fake_news/test.csv', test)


read_mr()
save_tok()
