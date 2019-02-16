
# How to run
Install packages (Python=3)
```
pip install -r requirements.txt
```
Copy the data and code on CSAIL server
```
cp /data/rsg/nlp/zhijing/proj/temp/char-cnn-text-classification-pytorch .
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
cd /data/rsg/nlp/zhijing/proj/temp/char-cnn-text-classification-pytorch
CUDA_VISIBLE_DEVICES='0' python train.py --cuda \
--save_folder model_temp \
--checkpoint_per_batch 10

cd /data/rsg/nlp/zhijing/proj/temp/char-cnn-text-classification-pytorch
CUDA_VISIBLE_DEVICES='0' python -m pdb -c continue train.py \
--cuda \
--save_folder model_fake \
--train_path data/fake_news/train.csv \
--val_path data/fake_news/test.csv --l0 10000


--train_path data/yelp_review_polarity_csv/train.csv \
--val_path data/yelp_review_polarity_csv/test.csv --l0 5300

# ag_news
python predictor.py --test_path='data/ag_news_csv/test.csv' \
--model_path='model_ag/CharCNN_best.pth.tar'

# yelp
python predictor.py --test_path='data/yelp_review_polarity_csv/test.csv' \
--model_path='model_yelp/CharCNN_epoch_1.pth.tar'

--model_path=model_temp/CharCNN_epoch_1.pth.tar
```

### Work done on 20190216
- debugged code for Yelp dataset
- changed the interface import order
- debugged pytorch version incompatibility
- downloaded fake_news dataset
- preprocessed fake_news dataset
- debugged data_loading for fake_news dataset

### Acknowledgement
The code is adapted from https://github.com/srviest/char-cnn-text-classification-pytorch

