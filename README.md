​

## Prerequisites

    Pytorch 1.1
    cuda 9.0
    python 3.6
    GPU Memory=24G

## Getting Started

Dataset Preparation

1.CUHK-PEDES
Organize them in dataset folder as follows:

  

```
> --data   
>  ---CUHK-PEDES      
>    -----imgs
> --------cam_a
> --------cam_b
> --------...   
>   --- reid_raw.json
```

Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)  
2.ICFG-PEDES
Organize them in dataset folder as follows:

```
-- data
--- RSTPReid
---- imgs
----processed_data
---------test_save.pkl
---------train_save.pkl
---------val_save.pkl
----  RSTPReid.json
```

We evaluate our method on RSTPReid. Please check the data.

## Training and Testing
python train.py
 --max-length 64 --batch-size 64 --num-epoches  --adam-lr 0.001 --gpus 0


## Evaluation
Import the 99checkpoint.pth file in the log file into the test_ Model.py, and then run test_ model.py

## E-mail
If you have any questions, please contact us: zhhylysys@yeah.net






​
