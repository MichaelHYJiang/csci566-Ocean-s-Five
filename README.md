# Extreme Dark Video Enhancement
#### USC 2019 Fall CSCI 566 *Ocean's Five* Course Project


## Baseline Version

### Download dataset from Google cloud first.
### Put it in *0_data* directory and unzip it.

### Generate file lists
```Shell
python generate_file_list.py
```

### Config parameters and then run training
```Shell
python train.py
```

### After training, run test command
```Shell
python test.py [test_case]
```
test_case can be:

0   test on training set

1   test on validation set

2   test on test set(save npy results)

3   test on customized set

All cases save mp4 output videos, while case 2 saves extra npy results.

Quantative measurements on current experiments:

|   Experiment Name  |                   Learning Rate                  | Group Number |  Frame Frequence  | Network Depth |        Adjustment        | Final Loss | Final Validation Loss |        PSNR        |        SSIM        |        MSE(MABD)       |
|:------------------:|:------------------------------------------------:|:------------:|:-----------------:|:-------------:|:------------------------:|:----------:|:---------------------:|:------------------:|:------------------:|:----------------------:|
|      Baseline      |    0-30epoch: LR = 1e-4 31-60epoch: LR = 1e-5    |      12      |         4         |       3       |                          | 0.02925395 |       0.03194653      |  27.20341440836589 | 0.8399437169233958 |  0.0007276190425069668 |
| Baseline Finetuned | same as above 61-75: LR = 1e-5 76-100: LR = 1e-4 |      12      | 0-75: 4 76-100: 1 |       3       |                          | 0.02811074 |       0.03246053      |                    |                    |                        |
|  Hist Layer in2he  |    0-30epoch: LR = 1e-4 31-60epoch: LR = 1e-5    |      12      |         4         |       3       |        hist layer        | 0.04175019 |       0.05246153      | 22.929300082171405 | 0.7788759288964449 |  0.0008090686585943601 |
|  Hist Layer he2he  |    0-30epoch: LR = 1e-4 31-60epoch: LR = 1e-5    |      12      |         4         |       3       |        hist layer        | 0.02953422 |       0.03511760      | 26.969168140270085 | 0.8392954715976009 | 0.00036061441545966536 |
|       Depth 2      |    0-30epoch: LR = 1e-4 31-60epoch: LR = 1e-5    |      12      |         4         |       2       | down-sampling number = 2 | 0.03208632 |       0.03496898      |  26.30993395911323 | 0.8239608135488299 | 0.00015883494476721382 |
|     FrameFreq2     |    0-30epoch: LR = 1e-4 31-60epoch: LR = 1e-5    |      12      |         2         |       3       |    frame frequency = 2   |            |                       |                    |                    |                        |
|       ResNet       |                                                  |              |                   |               |  conv -> Residual blocks |            |                       |                    |                    |                        |

### Detailed baseline results: 
https://drive.google.com/file/d/1DLVLR_MRh65gQV7GYVfXjNUma0v7J0PI/view?usp=sharing

