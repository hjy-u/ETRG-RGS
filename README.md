# ETRG-A-RGS

## Introduction
This is the official implementation code base for [A Parameter-Efficient Tuning Framework for Language-guided Object Grounding and Robot Grasping](https://arxiv.org/pdf/2409.19457) accepted for ICRA 2025. Our project produces **three separated** github repos for ETOG, ETRG-A, ETRG-B models. Stay tuned for code release. Here is our [Project Page](https://sites.google.com/umn.edu/etog-etrg/home).

This git repo includes the ETRG-A model in our paper, which is designed for parameter-efficient tuning on the Referring Grasp Synthesis (RGS) task.

![Pipeline Image](pipeline.png)

## Implementations for RGS and RGA Robotics Tasks

1. The ETOG model designed for ```Referring Expresion Segmentation (RES)``` task can be found [here](https://github.com/hjy-u/ETOG).

2. The ETRG-B model designed for ```Referring Grasp Affordance (RGA)``` task can be found here.


## Preparation
1. Conda env: We used Pytorch (2.1.0+cu118), other packages are in ```requirements.txt``` similar in our [ETOG](https://github.com/hjy-u/ETOG) repo
2. OCID-VLG dataset
   - Please follow the official [OCID-VLG](https://github.com/gtziafas/OCID-VLG) dataset preparation guidline.
   - The folder arrangement after preparation should be like this:

```
$ETRG
├── config
├── model
├── engine
├── pretrain (manually download from CLIP -> R50, R101, ViT-B-16)
├── tools
│     ├── data_process.py
│     └── ...
├── ...
└── datasets
    ├── ARID10
    ├── arid20
    ├── ...
    └── catalog.csv

```

## Pretrianed model wegihts and training/testing logs
Performance (mIoU) on Refcoco dataset:

| Backbone | IoU | J@1 | J@Any | Weights| Train log | Test log |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| ETRG-A-R50 | 79.82 | 89.06 | 92.17 | [models](https://drive.google.com/drive/folders/1jH-EgWZdZnpATMzYEUlDRCARLXy9zXwn?usp=sharing) | [log](https://drive.google.com/file/d/1zvSdIV2ssvWeSS_9grGkMk016RvV0x-d/view?usp=drive_link) | [log](https://drive.google.com/file/d/1fYJoWYLK3cHgUtzFCXxB7Faq9kl2VTvu/view?usp=drive_link) |
| ETRG-A-R101 | 80.11 | 89.38 | 93.49 | [models](https://drive.google.com/drive/folders/1nllSCFrE_d4Eh1_uXMLmUk1FeTrqKbkA?usp=drive_link) | [log](https://drive.google.com/file/d/1ooVgrL4TQ-_6zUQHibmDkxMJ-jFKPeda/view?usp=drive_link) | [log](https://drive.google.com/file/d/107iPGfxFEbjySM7eblLeMYEhFghBMhKT/view?usp=drive_link) |

## Train ETOG:

Directly run the```train.py``` file. Please modify the config files ```(e.g. config/OCID-VLG/etrg_r50.ymal)``` to change the batch_size, directory and test-split etc. values.

Our defualt setup: bs=11 on 1 NVIDIA RTX 2080 TI GPU.


## Test ETOG:
Directly run the```test.py``` file. Please modify the config files ```(e.g. config/OCID-VLG/etrg_r50.ymal)``` to change the directory specifics, model weights selection (```resume``` key).


## Acknowledgment

The code is adapted from [ETIRS](https://github.com/kkakkkka/ETRIS/tree/main) and [CROG](https://github.com/HilbertXu/CROG). We appreciate the authors for their wonderful codebase.

## Citation

If ETOG-ETRG is useful for your research, please consider citing:

```
@article{yu2024parameter,
  title={A Parameter-Efficient Tuning Framework for Language-guided Object Grounding and Robot Grasping},
  author={Yu, Houjian and Li, Mingen and Rezazadeh, Alireza and Yang, Yang and Choi, Changhyun},
  journal={arXiv preprint arXiv:2409.19457},
  year={2024}
}
```



