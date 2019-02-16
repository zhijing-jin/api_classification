
from predictor import Predictor


if __name__ == '__main__':
    model_path = 'model_ag/CharCNN_best.pth.tar'
    test_path='data/ag_news_csv/test.csv'
    label_n_txt = [(2, 'this is on sports'),
                   (1, 'this is on music'),
                   (3,"E-mail scam targets police chief Wiltshire Police warns about ""phishing"" after its fraud squad chief was targeted."),
                   (4,"Card fraud unit nets 36,000 cards In its first two years, the UK's dedicated card fraud unit, has recovered 36,000 stolen cards and 171 arrests - and estimates it saved 65m.")]


    predictor = Predictor(cuda=True, alphabet_path='alphabet.json', l0=1014, batch_size=2, num_workers=4,
                 model_path=model_path, kernel_sizes=[7, 7, 3, 3, 3, 3],
                 channel_size=256, pool_size=3, fc_size=1024, dropout=0.5)

    prob = predictor.pred(test_path, label_n_txt=label_n_txt)
    prob = predictor.pred(test_path, label_n_txt=None)

