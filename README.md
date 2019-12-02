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

|   Experiment Name  | Final Loss | Final Validation Loss |        PSNR        |        SSIM        |        MSE(MABD)       |                 Learning Rate                | Group Number |    Frame Frequence   | Network Depth |                  Adjustment                 | People |
|:------------------:|:----------:|:---------------------:|:------------------:|:------------------:|:----------------------:|:--------------------------------------------:|:------------:|:--------------------:|:-------------:|:-------------------------------------------:|:------:|
|      Baseline      | 0.02925395 |       0.03194653      |  27.20341440836589 | 0.8399437169233958 |  0.0007276190425069668 |      0-30epoch: 1e-4<br>31-60epoch: 1e-5     |      12      |           4          |       3       |                                             |   HJ   |
| Baseline Finetuned | 0.02811074 |       0.03246053      | 27.491393449571405 | 0.8447265682397065 | 0.00026039852772696717 | same as above<br>61-75: 1e-5<br>76-100: 1e-4 |      12      | 0-75: 4<br>76-100: 1 |       3       |                                             |   HJ   |
|  Hist Layer in2he  | 0.04175019 |       0.05246153      | 22.929300082171405 | 0.7788759288964449 |  0.0008090686585943601 |      0-30epoch: 1e-4<br>31-60epoch: 1e-5     |      12      |           4          |       3       |                  hist layer                 |   HJ   |
|  Hist Layer he2he  | 0.02953422 |       0.03511760      | 26.969168140270085 | 0.8392954715976009 | 0.00036061441545966536 |      0-30epoch: 1e-4<br>31-60epoch: 1e-5     |      12      |           4          |       3       |                  hist layer                 |   HJ   |
|       Depth 2      | 0.03208632 |       0.03496898      |  26.30993395911323 | 0.8239608135488299 | 0.00015883494476721382 |      0-30epoch: 1e-4<br>31-60epoch: 1e-5     |      12      |           4          |       2       |         down-sampling<br>number = 2         |   HJ   |
|       Depth 4      |            |                       |                    |                    |                        |                                              |              |                      |               |                                             |   HH   |
| Initial Channel 16 |            |                       |                    |                    |                        |                                              |              |                      |               |                                             |   HH   |
|     FrameFreq2     | 0.02718642 |       0.03153746      |  27.72178942362468 | 0.8486048811011844 | 0.00030751586973910323 |      0-30epoch: 1e-4<br>31-60epoch: 1e-5     |      12      |           2          |       3       |             frame frequency = 2             |   HJ   |
|       ResNet       | 0.02898007 |       0.03173088      |  27.43755528485333 | 0.8414171627274266 | 0.0001870047494116059  |      0-30epoch: 1e-4<br>31-60epoch: 1e-5     |      12      |           4          |       3       |           conv -> Residual blocks           |   FC   |
|  train by L1 Loss  | 0.02962459 |       0.03367069      |  27.26798362731934 | 0.840491048053459  | 0.00018188494183424305 |      0-30epoch: 1e-4<br>31-50epoch: 1e-5     |      10      |                      |       3       |                                             |   BW   |
| train by str Loss  | 0.18124115 |       0.20149784      |  26.70764254817257 | 0.853678109910753  | 0.0015242191872764136  |      0-30epoch: 1e-4<br>31-50epoch: 1e-5     |      10      |           4          |       3       |                                             |   BW   |
|train by region Loss| 0.15896447 |       0.17681161      | 27.234630761323157 | 0.8402597001305335 | 0.00048029046326743263 |      0-30epoch: 1e-4<br>31-50epoch: 1e-5     |      10      |           4          |       3       |                                             |   BW   |
|  train by VGG Loss |139.36647229|      150.83625793     | 27.022222095065647 | 0.8321757709538495 | 0.00016276957652055672 |      0-30epoch: 1e-4<br>31-50epoch: 1e-5     |      10      |           4          |       3       |                                             |   BW   |
|     Multi Loss1    | 1.75286838 |       1.91280270      | 27.228771979720506 | 0.8475041495429146 | 0.00037438217094211244 |      0-30epoch: 1e-4<br>31-50epoch: 1e-5     |      10      |           4          |       3       |                                             |   BW   |
|     Multi Loss2    | 0.35868726 |       0.40669683      | 27.064698537190754 | 0.8522180590364669 | 0.0023925343577309807  |      0-30epoch: 1e-4<br>31-50epoch: 1e-5     |      10      |           4          |       3       |                                             |   BW   |
|    Batch size>1    |            |                       |                    |                    |                        |                                              |              |                      |               |                                             |   YQ   |
|         GAN        | 0.02907291 |       0.03209123      | 27.383646908512816 | 0.8408598800500232 | 0.00014062573447008142 |      0-30epoch: 1e-4<br>31-60epoch: 1e-5     |      12      |           4          |       3       | 4 conv block<br>3 FC layer<br>discriminator |   HJ   |

### Detailed baseline results: 
https://drive.google.com/file/d/1DLVLR_MRh65gQV7GYVfXjNUma0v7J0PI/view?usp=sharing

