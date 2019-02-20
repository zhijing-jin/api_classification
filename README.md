
# How to run
Install packages (Python=3)
```
pip install -r charcnn/requirements.txt
```
Copy the data and code on CSAIL server
```
cp -a /data/rsg/nlp/zhijing/proj/1902attack .
```
Run the Prediction codes.
```
CUDA_VISIBLE_DEVICES='0' python predict.py
```

## Message:
In order to use the interface, just check out `predict.py`.

## Ready-To-Use Models
| | MR| AG | Fake| Yelp|  
|---|---|---|---|---|
|Fasttext| 72.7|91.4|99.4|93.7|
|CharCNN| 70.0|89.0|98.0||

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

### Work done on 20190218
- coded the interface for fasttext
- restructured folders

### Acknowledgement
The code for charcnn is adapted from https://github.com/srviest/char-cnn-text-classification-pytorch
The code for fasttext is from https://github.com/facebookresearch/fastText