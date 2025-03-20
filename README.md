# WormVIO 

## Installation

      $git clone https://github.com/Vench1/wormvio.git
      $conda env create --file environment.yaml
      $conda activate py310_wormVIO

## Data Preparation

The code in this repository is tested on [KITTI](https://www.cvlibs.net/datasets/kitti/) Odometry dataset. The IMU data after pre-processing is provided under `imus`. You should place the `imus` folder under the `kitti_odometry_color` folder. 

The dataset file structure should look like this:


```
kitti_odometry_color
├── imus
│   ├── 00.mat
│   ├── 01.mat
│   ...
│   └── 10.mat
├── poses
│   ├── 00.txt
│   ├── 01.txt
│   ...
│   └── 10.txt
└── sequences
    ├── 00
    │   ├── calib.txt
    │   ├── image_2
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    |   |   | ...
    │   ├── image_3
    │   └── times.txt
    ├── 01
    ...
    └── 10
```
## Training

Before training, you should check whether the dataset path is correctly set in `config.py`. You can adjust different parameters for training (e.g., selecting different fuse methods). 

Then, run the following script to start training:

      $python train.py

## Evaluation and Visualization

Run the following command to test our pretrained model:

      $python test.py

You can also run `eval/OTHERS/test_others.py` to evaluate other methods. For example, to evaluate ULVIO, run:

      $python eval/ULVIO/test_ulvio.py

The figures and error records will be generated under `./results/experiment_name/files`.


## Acknowledgements

This code is borrowed heavily from [ULVIO](https://github.com/jp4327/ulvio) , [Visual-Selective-VIO](https://github.com/mingyuyng/Visual-Selective-VIO) and [selective_sensor_fusion](https://github.com/changhao-chen/selective_sensor_fusion). We thank the authors for sharing their codes.

## Reference

If you find our work useful in your research, please consider citing our paper:
#TODO
