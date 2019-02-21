
# How to run
Install packages (Python=3)
```
pip install -r charcnn/requirements.txt
pip install
```
Copy the data and code on CSAIL server
```
cp -a /data/rsg/nlp/zhijing/proj/1902attack .
```
Run the Prediction codes.
```
CUDA_VISIBLE_DEVICES='0' python run_clf.py
```
Run the LM to get perplexity of a sentence.
```
CUDA_VISIBLE_DEVICES='0' python run_ppl.py
```

## How to use
get perplexity
```python
from run_ppl import LM
lm = LM()
sentence = 'it is a beautiful day'
ppl = lm.get_ppl(sentence, printout=True)

```

get classifier result
```python
from run_clf import get_clf_pred
model = 'fasttext'  # ['charcnn', 'fasttext']
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
    
```
## Message:
In order to use the interface, just check out `run_clf.py`.

## Ready-To-Use Models
| | MR| AG | Fake| Yelp|  
|---|---|---|---|---|
|Fasttext| 72.7|91.4|99.4|93.7|
|CharCNN| 70.0|89.0|98.0|93.0|



# The following is for Zhijing's Use (to train the model)
### Obtain Data
```
# data is at: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

/data/rsg/nlp/zhijing/tool/gdown/gdown.pl https://drive.google.com/open?id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg data/yelp_polar_csv.tar.gz
cd data
tar -xvf yelp_polar_csv.tar.gz
cd ..
```

### Run
```
# without _tok, the result is higher
cd /data/rsg/nlp/zhijing/proj/temp/charcnn
CUDA_VISIBLE_DEVICES='2' python train.py --cuda \
--save_folder model_temp \
--checkpoint_per_batch 10

cd /data/rsg/nlp/zhijing/proj/1902attack/charcnn
CUDA_VISIBLE_DEVICES='2' python train.py \
--cuda \
--save_folder model_fake_tok \
--train_path ../data/fake/train_tok.csv \
--val_path ../data/fake/test_tok.csv --l0 5000

cd /data/rsg/nlp/zhijing/proj/1902attack/charcnn
CUDA_VISIBLE_DEVICES='1' python train.py \
--cuda \
--save_folder model_yelp_tok \
--train_path ../data/yelp/train_tok.csv \
--val_path ../data/yelp/test_tok.csv --l0 5300


cd /data/rsg/nlp/zhijing/proj/1902attack/charcnn
CUDA_VISIBLE_DEVICES='0' python train.py \
--cuda \
--save_folder model_yelp_tok \
--train_path ../data/yelp/train_tok.csv \
--val_path ../data/yelp/test_tok.csv 

cd /data/rsg/nlp/zhijing/proj/1902attack/charcnn
CUDA_VISIBLE_DEVICES='3' python train.py \
--cuda \
--save_folder model_mr \
--train_path ../data/mr/train.csv \
--val_path ../data/mr/test.csv --l0 300 --lr 0.00001 \
--save_interval 100

--train_path ../data/yelp/train.csv \
--val_path ../data/yelp/test.csv --l0 5300

# ag_news
python predictor.py --test_path='data/ag/test.csv' \
--model_path='charcnn/model_ag/CharCNN_best.pth.tar'

# yelp
python predictor.py --test_path='data/yelp/test.csv' \
--model_path='charcnn/model_yelp/CharCNN_epoch_1.pth.tar'

# fake_news
python predictor.py --test_path='data/fake/test.csv' \
--model_path='charcnn/model_fake/CharCNN_best.pth.tar'

# mr
python predictor_charcnn.py --test_path='data/mr/test.csv' \
--model_path='charcnn/model_mr/CharCNN_best.pth.tar' # 68.0

# ag_news
python predictor_charcnn.py --test_path='data/ag/test_tok.csv' \
--model_path='charcnn/model_ag_tok/CharCNN_best.pth.tar'

# yelp
python predictor_charcnn.py --test_path='data/yelp/test_tok.csv' \
--model_path='charcnn/model_yelp_tok/CharCNN_epoch_1.pth.tar'

# fake_news
python predictor_charcnn.py --test_path='data/fake/test_tok.csv' \
--model_path='charcnn/model_fake_tok/CharCNN_best.pth.tar'
# mr
python predictor_charcnn.py --test_path='data/mr/test_tok.csv' \
--model_path='charcnn/model_mr_tok/CharCNN_best.pth.tar'
```

### fasttext
```
cd /data/rsg/nlp/zhijing/proj/1902attack/fasttext
./fasttext supervised \
-output model_fake_tok -input ../data/fake/train_ft.txt 

./fasttext supervised \
-output model_yelp_tok -input ../data/yelp/train_ft.txt 

./fasttext supervised \
-output model_ag_tok -input ../data/ag/train_ft.txt 

./fasttext supervised \
-output model_mr_tok -input ../data/mr/train_ft.txt 

./fasttext supervised \
-output model_temp -input ../data/fake/train_ft.txt 


./fasttext test \
 model_fake_tok.bin ../data/fake/test_ft.txt 1 # P@1	0.994

./fasttext predict-prob \
 model_fake_tok.bin ../data/fake/test_ft.txt 1
 
./fasttext test \
 model_ag_tok.bin ../data/ag/test_ft.txt 1 # P@1	0.914

./fasttext predict-prob \
 model_ag_tok.bin ../data/ag/test_ft.txt 1

./fasttext test \
 model_yelp_tok.bin ../data/yelp/test_ft.txt 1 # P@1	0.937

./fasttext predict-prob \
 model_yelp_tok.bin ../data/yelp/test_ft.txt 1
 
./fasttext test \
 model_mr_tok.bin ../data/mr/test_ft.txt 1 # P@1	0.727
 
#imdb 0.86

```

### Work done on 20190221
- coded the interface for bert LM
- adapted bert for perplexity scores

### Acknowledgement
The code for charcnn is adapted from https://github.com/srviest/char-cnn-text-classification-pytorch
The code for fasttext is from https://github.com/facebookresearch/fastText