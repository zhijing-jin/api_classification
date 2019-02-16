
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
CUDA_VISIBLE_DEVICES='1' python train.py --cuda \
--save_folder model_ag \
--checkpoint_per_batch 10

--train_path data/yelp_review_polarity_csv/train.csv \
--val_path data/yelp_review_polarity_csv/test.csv --l0 1052

python predictor.py --test_path='data/ag_news_csv/test.csv' \
--model_path='model_ag/CharCNN_best.pth.tar'

--model_path=model_temp/CharCNN_epoch_1.pth.tar
```

- debugged code for Yelp dataset
- changed the interface import order
- debugged pytorch version incompatibility

### Acknowledgement
The code is adapted from https://github.com/srviest/char-cnn-text-classification-pytorch

