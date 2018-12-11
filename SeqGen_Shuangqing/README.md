# SeqGen_Shuangqing
This is the code created by Shuangqing. It is based on the Seq2Seq framework and incorporates the techniques of *Global Encoding*.

***********************************************************

## Requirements
* Ubuntu 16.0.4
* Python >= 3.5
* Pytorch >= 0.4.1

**************************************************************

## Preprocessing
```
python3 preprocess.py --load_data path_to_data --save_data path_to_store_data 
```
Remember to put the data into a folder and name them *train.src*, *train.tgt*, *valid.src*, *valid.tgt*, *test.src* and *test.tgt*, and make a new folder inside called *data*.

***************************************************************

## Training
```
python3 train.py --log log_name --config config_yaml --gpus id
```

****************************************************************

## Evaluation
```
python3 train.py --log log_name --config config_yaml --gpus id --restore checkpoint --mode eval
```

```
