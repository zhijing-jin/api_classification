import csv
import sys
import random

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





read_fakenews()