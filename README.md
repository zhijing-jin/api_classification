
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
cd /data/rsg/nlp/zhijing/proj/temp/charcnn
CUDA_VISIBLE_DEVICES='0' python train.py --cuda \
--save_folder model_temp \
--checkpoint_per_batch 10

cd /data/rsg/nlp/zhijing/proj/temp/charcnn
CUDA_VISIBLE_DEVICES='0' python -m pdb -c continue train.py \
--cuda \
--save_folder model_fake \
--train_path ../data/fake/train.csv \
--val_path ../data/fake/test.csv --l0 5000


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

```

### fasttext
```
cd /data/rsg/nlp/zhijing/proj/temp/fasttext
./fasttext supervised \
-output model_fake -input ../data/fake/train_ft.txt 

./fasttext supervised \
-output model_yelp -input ../data/yelp/train_ft.txt 

./fasttext supervised \
-output model_ag -input ../data/ag/train_ft.txt 

./fasttext supervised \
-output model_temp -input ../data/fake/train_ft.txt 


./fasttext test \
 model_fake.bin ../data/fake/test_ft.txt 1

./fasttext predict-prob \
 model_fake.bin ../data/fake/test_ft.txt 1
 
./fasttext test \
 model_ag.bin ../data/ag/test_ft.txt 1

./fasttext predict-prob \
 model_ag.bin ../data/ag/test_ft.txt 1

./fasttext test \
 model_yelp.bin ../data/yelp/test_ft.txt 1

./fasttext predict-prob \
 model_yelp.bin ../data/yelp/test_ft.txt 1

```

### Work done on 20190218
- coded the interface for fasttext

### Acknowledgement
The code for charcnn is adapted from https://github.com/srviest/char-cnn-text-classification-pytorch
The code for fasttext is from https://github.com/facebookresearch/fastText