from predictor_charcnn import Predictor_charcnn
from predictor_fasttext import Predictor_fasttext


def get_prediction(model, dataset, label_n_txt=None):
    model_path_format = {
        'fasttext': '{}/model_{}.bin',
        'charcnn': '{}/model_{}/CharCNN_best.pth.tar'
    }
    model_path = model_path_format[model].format(model, dataset)

    ft_str = {'fasttext': '_ft',
              'charcnn': ''}
    test_path = 'data/{}/test{}.txt'.format(dataset, ft_str[model]) \
        if label_n_txt is None else None

    predictors = {'fasttext': Predictor_fasttext,
                  'charcnn': Predictor_charcnn}
    predictor = predictors[model](model_path=model_path)

    prob = predictor.pred(test_path=test_path, label_n_txt=label_n_txt)

    return prob


if __name__ == '__main__':
    model = 'fasttext' # ['charcnn', 'fasttext']
    dataset = 'ag' # ['fake', 'yelp', 'ag']

    label_n_txt = [(2, 'this is on sports'),
                   (1, 'this is on music'),
                   (3,
                    "E-mail scam targets police chief Wiltshire Police warns about ""phishing"" after its fraud squad chief was targeted."),
                   (4,
                    "Card fraud unit nets 36,000 cards In its first two years, the UK's dedicated card fraud unit, has recovered 36,000 stolen cards and 171 arrests - and estimates it saved 65m.")]

    # if you want to test out the file in `test_path`, just put None for label_n_txt
    preds = get_prediction(model, dataset, label_n_txt=None)

    # if you want to test out `label_n_txt`, just put a list for label_n_txt
    preds = get_prediction(model, dataset, label_n_txt=label_n_txt)
    import pdb;

    pdb.set_trace()
