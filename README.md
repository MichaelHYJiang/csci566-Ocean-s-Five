# Learning-to-See-Moving-Objects-in-the-Dark

## Modified from Master Branch
Files change: network.py, train_batch.py

## Usage
Download dataset from Google Cloud first. Put it in 0_data directory and unzip it.

### Generate file lists
```Shell
python generate_file_list.py
```

### Training
```Shell
python train_batch.py
```

### Testing
```Shell
python test_batch.py [test_case]
```
test_case can be:

0 test on training set

1 test on validation set

2 test on test set(save npy results)

3 test on customized set

All cases save mp4 output videos, while case 2 saves extra npy results.
