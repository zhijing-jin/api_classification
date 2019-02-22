import numpy as np
from efficiency.function import shell
from efficiency.log import fwrite

import spacy


def _tokenize(nlp, sent_str, lowercase=True):
    doc = nlp(sent_str, disable=['parser', 'tagger', 'ner'])
    s = ' '.join([token.text for token in doc])
    if lowercase:
        s = s.lower()
    return s


class Predictor_fasttext:
    def __init__(self, model_path='fasttext/model_yelp.bin', batch_size=None):
        self.model_name = 'fasttext'
        self.command = '{model_name}/{model_name} predict-prob \
                {model_path} {{test_path}} {{k_most}}' \
            .format(model_name=self.model_name, model_path=model_path)
        self.n_lbl = 1  # k_most = 1

        self.test_path_default = '{}/temp_test.txt'.format(self.model_name)
        self.lbl_pref = '__label__'

        self.nlp = spacy.load('en')

    def pred(self, test_path=None, label_n_txt=None, lowercase=True):
        if label_n_txt is not None:
            test_path = self.test_path_default
            self.label, self.data = list(zip(*label_n_txt))
            self.data = [txt.lower() if lowercase else txt for txt in self.data]

            writeout = ['{}{} {}\n'.format(self.lbl_pref, lbl, _tokenize(self.nlp, dat))
                        for lbl, dat in zip(self.label, self.data)]
            fwrite(''.join(writeout), test_path)

        self.n_lbl = self._get_n_lbl(test_path)
        command = self.command.format(test_path=test_path, k_most=self.n_lbl)
        stdout, stderr = shell(command, stderr=True)

        return self._stdout_to_preds(stdout)

    def _get_n_lbl(self, file):
        if not file:
            return self.n_lbl

        with open(file) as f:
            content = [row for row in f if row]
            lbls = [int(row.split()[0].replace(self.lbl_pref, ''))
                    for row in content]
            n_lbl = max(lbls)
        return n_lbl

    def _stdout_to_preds(self, stdout):
        content = stdout.split(b'\n')
        n_lbl = self.n_lbl
        preds_tuple = []
        for row_ix, row in enumerate(content):
            if not row:
                continue

            k_pred = []
            toks = row.split()
            assert len(toks) / 2 == n_lbl

            for tok_ix in range(n_lbl):
                lbl, prob = toks[tok_ix * 2: tok_ix * 2 + 2]
                b_lbl_pref = str.encode(self.lbl_pref)
                lbl = int(lbl.replace(b_lbl_pref, b''))
                prob = float(prob)

                k_pred += [(lbl, prob)]

            preds_tuple += [k_pred]

        n_data = len(preds_tuple)

        preds = np.zeros((n_data, n_lbl))
        for pred_ix, k_pred in enumerate(preds_tuple):
            for lbl, prob in k_pred:
                preds[pred_ix, lbl - 1] = prob

        return preds
