# raw-pc-darts-anti-spoofing
This repository includes the code to reproduce our paper [Raw Differentiable Architecture Search for Speech Deepfake and Spoofing Detection](https://arxiv.org/abs/2107.12212) published in the ASVspoof 2021 workshop.

### Dependencies
```
pip install -r requirements.txt
```

### Dataset
The ASVspoof2019 database can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336)

The extracted data should be orginased as:
* LA/
   * ASVspoof2019_LA_dev/flac/...
   * ASVspoof2019_LA_eval/flac/...
   * ASVspoof2019_LA_train/flac/...
   * ASVspoof2019.LA.cm.dev.trl.txt
   * ASVspoof2019.LA.cm.eval.trl.txt
   * ASVspoof2019.LA.cm.train.trn.txt
   * ASVspoof2019.LA.cm.train.trn_h.txt (uploaded in /split_protocols)
   * ASVspoof2019.LA.cm.train.trn_t.txt (uploaded in /split_protocols)
   * ...


For convience, you can change the codes' default `--data` argument to `'/path/to/your/LA'`, instead of typing it for each run.

### Usage
#### Architecture Search
To search with 8 layers with 64 initial channels, and with fixed sinc layer initialised using mel scale:
```
python train_search.py --layers=8 --init_channels=64 --sinc_scale=mel
```
You can also try with other two scales (linear scale: `linear` and inverse-mel scale: `lem`), and set the scale to learnable:
```
python train_search.py --layers=8 --init_channels=64 --sinc_scale=linear --trainable
```
#### Train from Scratch
To train with the reported best architecture in the paper, using 8 layers, 64 initial channels and masked mel scale sinc layer:
```
python train_model.py --arch=ARCH --layers=8 --init_channels=64 --sinc_scale=mel
```
replace `ARCH` with `"Genotype(normal=[('dil_conv_5', 1), ('dil_conv_3', 0), ('dil_conv_5', 1), ('dil_conv_5', 2), ('std_conv_5', 2), ('skip_connect', 3), ('std_conv_5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3', 0), ('std_conv_3', 1), ('dil_conv_3', 0), ('dil_conv_3', 2), ('skip_connect', 0), ('dil_conv_5', 2), ('dil_conv_3', 0), ('avg_pool_3', 1)], reduce_concat=range(2, 6))"`

To train with the reported best architecture in the paper, using 8 layers, 64 initial channels and masked, also learned linear scale sinc layer:
```
python train_model.py --arch=ARCH --layers=8 --init_channels=64 --sinc_scale=linear --pre_trained=pre_trained_models/trainable_linear_in_search.pth
```
replace `ARCH` with `"Genotype(normal=[('dil_conv_5', 0), ('dil_conv_5', 1), ('dil_conv_5', 0), ('max_pool_3', 2), ('std_conv_5', 1), ('dil_conv_5', 0), ('dil_conv_3', 0), ('std_conv_3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5', 0), ('dil_conv_5', 1), ('dil_conv_3', 2), ('dil_conv_3', 0), ('dil_conv_3', 1), ('dil_conv_5', 3), ('dil_conv_3', 1), ('dil_conv_3', 0)], reduce_concat=range(2, 6))"`

#### Evaluate
To evaluate the saved model using the same architecture in train from scratch on LA Evaluation partition:
```
python evaluate.py --arch=ARCH --model=/path/to/your/saved/models/epoch_x.pth --layers=8 --init_channels=64 --frontend=SCALE
```
also replace `ARCH` with the corresponding architecture, and `SCALE` with the corresponding scale.

#### Pre-trained models
The pre-trained models are too large for uploading, will find a way to share latter.

#### Citation
If you find this repository useful, please consider citing:
```
@inproceedings{ge21_asvspoof,
  author={Wanying Ge and Jose Patino and Massimiliano Todisco and Nicholas Evans},
  title={{Raw Differentiable Architecture Search for Speech Deepfake and Spoofing Detection}},
  year=2021,
  booktitle={Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
  pages={22--28},
}
```
#### Acknowledgement
Codes are based on the implementations of [pc-darts-anti-spoofing
](https://github.com/eurecom-asp/pc-darts-anti-spoofing), [rawnet2-antispoofing
](https://github.com/eurecom-asp/rawnet2-antispoofing) and and [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts).