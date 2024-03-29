from predictor_charcnn import Predictor_charcnn
from predictor_fasttext import Predictor_fasttext


def get_clf_pred(model, dataset, label_n_txt=None, test_path=None, batch_size=20):
    model_path_format = {
        'fasttext': '{}/model_{}_tok.bin',
        'charcnn': '{}/model_{}_tok/CharCNN_best.pth.tar'
    }
    model_path = model_path_format[model].format(model, dataset)

    ft_str = {'fasttext': '_ft.txt',
              'charcnn': '_tok.csv'}

    test_path = 'data/{}/test{}'.format(dataset, ft_str[model]) \
        if not (label_n_txt and test_path) else None

    predictors = {'fasttext': Predictor_fasttext,
                  'charcnn': Predictor_charcnn}
    batch_size = batch_size if label_n_txt is None else min(batch_size, len(label_n_txt))
    predictor = predictors[model](model_path=model_path, batch_size=batch_size)

    prob = predictor.pred(test_path=test_path, label_n_txt=label_n_txt)

    return prob


def main():
    model = 'charcnn'  # ['charcnn', 'fasttext']
    dataset = 'ag'  # ['fake', 'yelp', 'ag']

    label_n_txt = [(2, 'this is on sports'),
                   (1, 'this is on music'),
                   (3,
                    "E-mail scam targets police chief Wiltshire Police warns about ""phishing"" after its fraud squad chief was targeted."),
                   (4,
                    "Card fraud unit nets 36,000 cards In its first two years, the UK's dedicated card fraud unit, has recovered 36,000 stolen cards and 171 arrests - and estimates it saved 65m.")]

    # if you want to test out `label_n_txt`, just put a list for label_n_txt
    preds = get_clf_pred(model, dataset, label_n_txt=label_n_txt)

    # if you want to test out the file in `test_path`, just put None for label_n_txt.
    # if you put '' in test_path, it will use the default test_path 'data/{}/test{}.txt'
    preds = get_clf_pred(model, dataset, label_n_txt=None, test_path='')

    print("[Info] Preds:", preds)


if __name__ == '__main__':
    main()
